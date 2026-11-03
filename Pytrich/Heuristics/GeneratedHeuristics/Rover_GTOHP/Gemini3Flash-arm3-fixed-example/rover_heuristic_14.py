import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Rover_GTOHP domain.

1. Bottleneck Analysis:
   The primary bottleneck in the Rover domain is navigation and data processing sequence. 
   Every data collection goal requires the rover to be at a specific waypoint to acquire a 
   sample or image, and then (often) navigate to a different waypoint visible to the lander 
   to communicate the data. Cameras must also be calibrated before taking images, and 
   stores must be emptied before sampling.

2. Admissible Lower Bounds:
   - Term 1 (Goal-based): Every uncommunicated data goal requires at least one 'communicate' action. 
     If the required analysis/image is not currently held by any rover, it also requires 
     at least one 'sample' or 'take_image' action. This is a hard lower bound on primitive actions.
   - Term 2 (Task-based): Every task in the hierarchical task network represents a sequence 
     of actions that MUST be performed to satisfy the problem constraints. By assigning a minimum 
     number of actions required to decompose each task into primitive operations, we create an 
     admissible sum for the current network.

3. Preprocessing:
   We precompute bit IDs for 'at', 'have_analysis', 'empty', and goal predicates during initialization 
   to ensure O(1) state checks during search.

4. Tie-breaking:
   A sub-unit penalty is applied based on the task network size to encourage decomposition progress.
"""

class RoverGTOHPGemini3FlashHeuristic14(Heuristic):
    def __init__(self, rover_gemini3flash_heuristic_14=None, name="RoverGTOHPGemini3FlashHeuristic14"):
        super().__init__(name=name)
        self.rover_gemini3flash_heuristic_14 = rover_gemini3flash_heuristic_14 # Compliance variable
        self.soil_goals = {} # waypoint -> bit_id
        self.rock_goals = {} # waypoint -> bit_id
        self.image_goals = {} # (obj, mode) -> bit_id
        
        self.rover_at_bits = defaultdict(dict) # rover -> waypoint -> bit_id
        self.rover_have_soil_bits = defaultdict(dict) # rover -> waypoint -> bit_id
        self.rover_have_rock_bits = defaultdict(dict) # rover -> waypoint -> bit_id
        self.rover_have_image_bits = defaultdict(dict) # rover -> (obj, mode) -> bit_id
        self.rover_empty_bit = {} # rover -> bit_id (empty bit for that rover's store)
        self.rover_calibrated_bit = defaultdict(dict) # rover -> camera -> bit_id
        
        self.rovers = set()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        if "[" in name and name.endswith("]"):
            bracket_idx = name.find("[")
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return None, []

    def _parse_task_name(self, name: str):
        s = name.strip("() ")
        parts = s.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map Stores to Rovers
        store_to_rover = {}
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "store_of": # store_of[store, rover]
                store_to_rover[args[0]] = args[1]
                self.rovers.add(args[1])

        # 2. Extract Fact Bit IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "at": # at[rover, waypoint]
                self.rover_at_bits[args[0]][args[1]] = fact.global_id
            elif pred == "have_soil_analysis": # have_soil_analysis[rover, waypoint]
                self.rover_have_soil_bits[args[0]][args[1]] = fact.global_id
            elif pred == "have_rock_analysis": # have_rock_analysis[rover, waypoint]
                self.rover_have_rock_bits[args[0]][args[1]] = fact.global_id
            elif pred == "have_image": # have_image[rover, obj, mode]
                self.rover_have_image_bits[args[0]][(args[1], args[2])] = fact.global_id
            elif pred == "empty": # empty[store]
                if args[0] in store_to_rover:
                    self.rover_empty_bit[store_to_rover[args[0]]] = fact.global_id
            elif pred == "calibrated": # calibrated[camera, rover]
                self.rover_calibrated_bit[args[1]][args[0]] = fact.global_id

        # 3. Extract Goal bits
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            # Find the Fact object matching this goal name string
            for fact in model.facts:
                if fact.name == g_name:
                    if pred == "communicated_soil_data":
                        self.soil_goals[args[0]] = fact.global_id
                    elif pred == "communicated_rock_data":
                        self.rock_goals[args[0]] = fact.global_id
                    elif pred == "communicated_image_data":
                        self.image_goals[(args[0], args[1])] = fact.global_id
                    break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        # Term 1: Goal-based Admissible Cost
        h_goals = 0
        
        # Soil Goals
        for waypoint, g_bit in self.soil_goals.items():
            if not (state >> g_bit) & 1:
                held = any((state >> self.rover_have_soil_bits[r].get(waypoint, -1)) & 1 for r in self.rovers if waypoint in self.rover_have_soil_bits[r])
                h_goals += 1 if held else 2 # sample + communicate
        
        # Rock Goals
        for waypoint, g_bit in self.rock_goals.items():
            if not (state >> g_bit) & 1:
                held = any((state >> self.rover_have_rock_bits[r].get(waypoint, -1)) & 1 for r in self.rovers if waypoint in self.rover_have_rock_bits[r])
                h_goals += 1 if held else 2
                
        # Image Goals
        for (obj, mode), g_bit in self.image_goals.items():
            if not (state >> g_bit) & 1:
                held = any((state >> self.rover_have_image_bits[r].get((obj, mode), -1)) & 1 for r in self.rovers if (obj, mode) in self.rover_have_image_bits[r])
                h_goals += 1 if held else 3 # calibrate + take_image + communicate

        # Term 2: Task-network-based Admissible Cost
        h_tasks = 0
        for t in task_network:
            t_name, t_args = self._parse_task_name(t.name)
            if not t_name: continue
            
            if t_name in ["get_soil_data", "get_rock_data"]: h_tasks += 2
            elif t_name == "get_image_data": h_tasks += 3
            elif t_name in ["send_soil_data", "send_rock_data", "send_image_data"]: h_tasks += 1
            elif t_name in ["do_navigate1", "do_navigate2"]:
                if len(t_args) >= 2:
                    rover, to = t_args[0], t_args[1]
                    at_bit = self.rover_at_bits.get(rover, {}).get(to, -1)
                    if at_bit == -1 or not (state >> at_bit) & 1: h_tasks += 1
            elif t_name == "empty_store":
                if len(t_args) >= 2:
                    rover = t_args[1]
                    e_bit = self.rover_empty_bit.get(rover, -1)
                    if e_bit != -1 and not (state >> e_bit) & 1: h_tasks += 1
            elif t_name == "do_calibrate": h_tasks += 1
            elif t_name in ["navigate", "sample_soil", "sample_rock", "drop", "calibrate", "take_image", "communicate_soil_data1", "communicate_soil_data2", "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data"]:
                h_tasks += 1
            # visit, unvisit, nop cost 0

        # Combine terms and add tie-breaker
        h_total = max(h_goals, h_tasks)
        return h_total + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
