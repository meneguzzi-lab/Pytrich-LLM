from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGTOHPGemini3FlashHeuristic19(Heuristic):
    """
    Domain-Dependent Heuristic for Rover_GTOHP.

    Bottleneck Analysis:
    The primary serialization in the Rover domain stems from the single-store constraint per rover,
    necessitating a cycle of navigating to a target, ensuring the store is empty, sampling,
    and then navigating to a communication-ready waypoint for transmission.

    Admissibility Sketches:
    1. network_cost: Sums the minimum number of remaining primitive actions (e.g., sample + communicate) 
       required for each task currently in the task network. Since each goal (soil/rock/image) 
       requires a unique transmission and unique sample/image capture, these actions are disjoint 
       and mandatory, making their sum a lower bound on cost.
    2. goal_cost: Counts the number of unique goal facts (communicated data) not yet true in the state;
       every unmet goal represents at least one mandatory 'communicate' action not yet performed.
    """

    def __init__(self, rover_gemini3flash_heuristic_19="rover_gemini3flash_heuristic_19"):
        super().__init__(name=rover_gemini3flash_heuristic_19)
        self.at_ids = {}            # (rover, waypoint) -> global_id
        self.empty_ids = {}         # store -> global_id
        self.have_soil_ids = {}     # (rover, waypoint) -> global_id
        self.have_rock_ids = {}     # (rover, waypoint) -> global_id
        self.have_image_ids = {}    # (rover, objective, mode) -> global_id
        self.comm_soil_ids = {}     # waypoint -> global_id
        self.comm_rock_ids = {}     # waypoint -> global_id
        self.comm_image_ids = {}    # (objective, mode) -> global_id
        self.calibrated_ids = {}    # (camera, rover) -> global_id

        self.goal_ids = []
        self.rovers = []
        self.stores = []
        self.rover_to_store = {}
        self.task_type_cache = {}    # task_global_id -> (type, args)

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to global IDs
        for fact in model.facts:
            p, args = self._parse_fact_name(fact.name)
            if p == "at": self.at_ids[tuple(args)] = fact.global_id
            elif p == "empty": 
                self.empty_ids[args[0]] = fact.global_id
                if args[0] not in self.stores: self.stores.append(args[0])
            elif p == "have_soil_analysis": self.have_soil_ids[tuple(args)] = fact.global_id
            elif p == "have_rock_analysis": self.have_rock_ids[tuple(args)] = fact.global_id
            elif p == "have_image": self.have_image_ids[tuple(args)] = fact.global_id
            elif p == "communicated_soil_data": self.comm_soil_ids[args[0]] = fact.global_id
            elif p == "communicated_rock_data": self.comm_rock_ids[args[0]] = fact.global_id
            elif p == "communicated_image_data": self.comm_image_ids[tuple(args)] = fact.global_id
            elif p == "calibrated": self.calibrated_ids[tuple(args)] = fact.global_id
            elif p == "available": 
                if args[0] not in self.rovers: self.rovers.append(args[0])
            elif p == "store_of":
                self.rover_to_store[args[1]] = args[0]

        # 2. Map goals
        goal_names = model.state_explicit_repr(model.goals)
        for gn in goal_names:
            for f in model.facts:
                if f.name == gn:
                    self.goal_ids.append(f.global_id)
                    break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        network_cost = 0
        for task in task_network:
            t_id = task.global_id
            if t_id not in self.task_type_cache:
                self.task_type_cache[t_id] = self._parse_task_name(task.name)
            
            t_type, t_args = self.task_type_cache[t_id]

            # High-level task estimated remaining primitive actions
            if t_type == "get_soil_data":
                wp = t_args[0]
                if self._check(state, self.comm_soil_ids.get(wp, -1)): network_cost += 0
                elif any(self._check(state, self.have_soil_ids.get((r, wp), -1)) for r in self.rovers): network_cost += 1
                else: network_cost += 2
            elif t_type == "get_rock_data":
                wp = t_args[0]
                if self._check(state, self.comm_rock_ids.get(wp, -1)): network_cost += 0
                elif any(self._check(state, self.have_rock_ids.get((r, wp), -1)) for r in self.rovers): network_cost += 1
                else: network_cost += 2
            elif t_type == "get_image_data":
                obj, mode = t_args[0], t_args[1]
                if self._check(state, self.comm_image_ids.get((obj, mode), -1)): network_cost += 0
                elif any(self._check(state, self.have_image_ids.get((r, obj, mode), -1)) for r in self.rovers): network_cost += 1
                else: network_cost += 3 # calibrate + take_image + transmit
            
            # Navigation tasks
            elif t_type in ["do_navigate1", "do_navigate2"]:
                # args: [rover, target] or [rover, from, target]
                rover, target = t_args[0], t_args[-1]
                if not self._check(state, self.at_ids.get((rover, target), -1)): network_cost += 1
            
            # Internal decomposition tasks
            elif t_type == "empty_store":
                store = t_args[0]
                if not self._check(state, self.empty_ids.get(store, -1)): network_cost += 1
            elif t_type == "do_calibrate":
                rover, cam = t_args[0], t_args[1]
                if not self._check(state, self.calibrated_ids.get((cam, rover), -1)): network_cost += 2 # move + calibrate
            
            # Sending/Sampling tasks
            elif t_type in ["send_soil_data", "send_rock_data", "send_image_data"]:
                network_cost += 1
            elif t_type in ["sample_soil", "sample_rock", "take_image", "calibrate", "navigate", "drop"]:
                network_cost += 1
            
            # Catch for actual communication operators
            elif "communicate" in t_type:
                network_cost += 1

        # Unmet goals remaining fallback
        goal_cost = sum(1 for gid in self.goal_ids if not ((state >> gid) & 1))
        
        # For guidance, use a weighted sum of network actions and goal actions
        return int(network_cost + (goal_cost * 0.1))

    def _check(self, state, gid):
        if gid == -1: return False
        return (state >> gid) & 1

    def _parse_fact_name(self, name: str):
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _parse_task_name(self, name: str):
        if "[" in name:
            parts = name.replace("]", "").split("[")
            t_type = parts[0]
            t_args = parts[1].split(",")
            return t_type, t_args
        elif "(" in name:
            parts = name.replace(")", "").split("(")
            t_type = parts[0]
            t_args = parts[1].split(",")
            return t_type, t_args
        else:
            parts = name.split()
            return parts[0], parts[1:]
