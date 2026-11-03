import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Rover_GTOHP domain.

    Bottleneck Analysis:
    The core bottleneck in this domain is the multi-stage data acquisition pipeline: 
    a rover must be equipped, navigate to a site, ensure its store is empty (for samples), 
    perform the physical sampling or imaging, navigate to a transmission point, 
    and finally communicate the data. This serialized flow means that for every goal 
    or high-level task, there is a clear, unavoidable sequence of primitive actions.

    Admissibility Sketches:
    1. Goal-Based Data Cost: Every 'communicated_data' goal not yet true requires at least
       one 'communicate' operator. If no rover has the required data in memory yet,
       it also requires 'sample' (1) or 'calibrate'+'take_image' (2). This is a 
       sum of independent mandatory actions, making it a lower bound.
    2. Task-Network Decomposition Cost: Every grounded task in the network must be 
       reduced to operators. Summing the minimum state-aware reduction cost of all 
       active tasks (e.g., a 'get_soil_data' task always needing at least 2 actions)
       is admissible because these tasks must be executed to reach the goal.
    """

    def __init__(self, rover_gemini3flash_heuristic_19=None, name="RoverGTOHPGemini3FlashHeuristic19"):
        super().__init__(name=name)
        self.goal_data = []
        self.predicate_map = defaultdict(dict)
        self.rover_info = {}
        self.store_to_rover = {}
        self.lander_locations = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        self._preprocess()
        h_value = self._compute_heuristic(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        # 1. Index Predicates
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at_lander":
                self.lander_locations.append(args[1])
            elif pred == "store_of":
                self.store_to_rover[args[0]] = args[1]
                if args[1] not in self.rover_info: self.rover_info[args[1]] = {'soil': False, 'rock': False, 'imaging': False, 'store': args[0]}
            elif pred == "equipped_for_soil_analysis":
                if args[0] not in self.rover_info: self.rover_info[args[0]] = {'soil': True, 'rock': False, 'imaging': False}
                self.rover_info[args[0]]['soil'] = True
            elif pred == "equipped_for_rock_analysis":
                if args[0] not in self.rover_info: self.rover_info[args[0]] = {'soil': False, 'rock': True, 'imaging': False}
                self.rover_info[args[0]]['rock'] = True
            elif pred == "equipped_for_imaging":
                if args[0] not in self.rover_info: self.rover_info[args[0]] = {'soil': False, 'rock': False, 'imaging': True}
                self.rover_info[args[0]]['imaging'] = True
            
            # Bit indexes for state checking
            key = tuple(args)
            self.predicate_map[pred][key] = fact.global_id

        # 2. Identify Goals from Bitmask
        if hasattr(self.model, 'goals'):
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_names:
                pred, args = self._parse_fact_name(g_name)
                if pred in ["communicated_soil_data", "communicated_rock_data", "communicated_image_data"]:
                    self.goal_data.append((pred, args, self.predicate_map[pred][tuple(args)]))

    def _compute_heuristic(self, state: int, task_network) -> int:
        # Component 1: Goal-based progress
        h_goals = 0
        for pred, args, g_bit in self.goal_data:
            if not (state >> g_bit & 1):
                # Goal not communicated yet
                # Check if data is already gathered by ANY rover
                data_gathered = False
                if pred == "communicated_soil_data":
                    for r in self.rover_info:
                        bit = self.predicate_map["have_soil_analysis"].get((r, args[0]))
                        if bit is not None and (state >> bit & 1):
                            data_gathered = True; break
                    h_goals += 1 if data_gathered else 2
                elif pred == "communicated_rock_data":
                    for r in self.rover_info:
                        bit = self.predicate_map["have_rock_analysis"].get((r, args[0]))
                        if bit is not None and (state >> bit & 1):
                            data_gathered = True; break
                    h_goals += 1 if data_gathered else 2
                elif pred == "communicated_image_data":
                    for r in self.rover_info:
                        bit = self.predicate_map["have_image"].get((r, args[0], args[1]))
                        if bit is not None and (state >> bit & 1):
                            data_gathered = True; break
                    h_goals += 1 if data_gathered else 3

        # Component 2: Task-network requirements (P2 pattern)
        h_tasks = 0
        for task in task_network:
            t_pred, t_args = self._parse_fact_name(task.name)
            if not t_pred: continue

            if t_pred == "get_soil_data" or t_pred == "get_rock_data":
                h_tasks += 2
            elif t_pred == "get_image_data":
                h_tasks += 3
            elif t_pred in ["send_soil_data", "send_rock_data", "send_image_data", "do_calibrate"]:
                h_tasks += 1
            elif t_pred in ["do_navigate1", "do_navigate2"]:
                h_tasks += 1
            elif hasattr(task, 'cost'):
                h_tasks += task.cost
            else:
                h_tasks += 1

        # Tie-breaker: penalized state if lander channel is busy
        penalty = 0
        # lander channel check logic could go here, but let's keep it fast.

        return max(h_goals, h_tasks)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
