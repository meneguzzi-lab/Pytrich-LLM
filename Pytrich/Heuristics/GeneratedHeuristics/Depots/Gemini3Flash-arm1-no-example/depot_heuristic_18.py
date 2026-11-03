from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The primary constraint in Depots is the coordination of hoists and trucks at specific locations. 
    A hoist is required for every lift/drop/load/unload, and a truck is required to move crates between locations. 
    Stacks (towers) of crates create a serialization bottleneck, as buried crates or target surfaces must be cleared 
    before a 'do_put_on' task can proceed.

    Admissible Lower Bounds:
    1. Task Action Lower Bound: Each abstract task (e.g., do_put_on, do_clear) has a minimum number 
       of primitive actions (Lift, Drop, Load, etc.) required to decompose based on the current state.
    2. Crate Displacement: For every crate not yet on its goal surface, a minimum sequence of 
       (Lift, [Load, Drive, Unload], Drop) is required. 
    3. Surface Clearing: If a surface (pallet or crate) is not clear and is a target for a crate, 
       at least 2 actions (Lift + Drop/Load) are required for each obstructing crate.
    """

    def __init__(self, depot_gemini3flash_heuristic_18="DepotsGemini3FlashHeuristic18"):
        super().__init__(depot_gemini3flash_heuristic_18)
        self.on_ids = {}         # (crate, surface) -> global_id
        self.at_ids = {}         # (locatable, place) -> global_id
        self.in_ids = {}         # (crate, truck) -> global_id
        self.lifting_ids = {}    # (hoist, crate) -> global_id
        self.clear_ids = {}      # surface -> global_id
        self.available_ids = {}  # hoist -> global_id
        
        self.place_names = []
        self.truck_names = []
        self.fixed_places = {}   # obj_name -> place_name (for pallets/hoists)
        self.task_data = {}      # task.global_id -> (type, args)

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
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        init_facts = model.state_explicit_repr(model.initial_state)
        all_places = set()
        all_trucks = set()
        
        # Map facts to global IDs for fast bitwise checks
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            if pred == "on": self.on_ids[(args[0], args[1])] = fact.global_id
            elif pred == "at": self.at_ids[(args[0], args[1])] = fact.global_id
            elif pred == "in": self.in_ids[(args[0], args[1])] = fact.global_id
            elif pred == "lifting": self.lifting_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clear": self.clear_ids[args[0]] = fact.global_id
            elif pred == "available": self.available_ids[args[0]] = fact.global_id

        # Identify object types and fixed locations from initial state
        for f_name in init_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == "at":
                all_places.add(args[1])
                # Pallets and Hoists are effectively static in Depots
                if "pallet" in args[0] or "hoist" in args[0]:
                    self.fixed_places[args[0]] = args[1]
            if pred == "in":
                all_trucks.add(args[1])

        self.place_names = list(all_places)
        self.truck_names = list(all_trucks)

        # Precompute task types and arguments for efficiency in __call__
        for t in model.abstract_tasks:
            parts = t.name.replace('(', '').replace(')', '').split()
            if parts: self.task_data[t.global_id] = (parts[0], parts[1:])
        for o in model.operators:
            parts = o.name.replace('(', '').replace(')', '').split()
            if parts: self.task_data[o.global_id] = (parts[0], parts[1:])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _is_true(self, state, fact_id):
        return (state >> fact_id) & 1 if fact_id is not None else False

    def _get_loc(self, state, obj_name):
        if obj_name in self.fixed_places: return self.fixed_places[obj_name]
        for p in self.place_names:
            if self._is_true(state, self.at_ids.get((obj_name, p))): return p
        return None

    def _compute(self, state, task_network) -> int:
        h = 0
        for task in task_network:
            t_info = self.task_data.get(task.global_id)
            if not t_info: 
                if task.name != "nop": h += 1
                continue
            
            t_type, t_args = t_info
            
            if t_type == "do_put_on":
                crate, surface = t_args[0], t_args[1]
                if not self._is_true(state, self.on_ids.get((crate, surface))):
                    h += 2 # Lift + Drop
                    # Check relocation necessity
                    p_target = self._get_loc(state, surface)
                    p_crate = self._get_loc(state, crate)
                    if p_crate and p_target and p_crate != p_target:
                        h += 3 # Load + Drive + Unload
                    elif not p_crate: # Check if in truck
                        in_any_truck = False
                        for truck in self.truck_names:
                            if self._is_true(state, self.in_ids.get((crate, truck))):
                                h += 1 # Unload
                                p_truck = self._get_loc(state, truck)
                                if p_truck and p_truck != p_target: h += 1 # Drive
                                in_any_truck = True
                                break
                    # Stacking costs
                    if not self._is_true(state, self.clear_ids.get(surface)): h += 2
                    if not self._is_true(state, self.clear_ids.get(crate)): h += 2
            
            elif t_type == "do_clear":
                if not self._is_true(state, self.clear_ids.get(t_args[0])): h += 2
            
            elif t_type == "do_get_truck":
                truck, place = t_args[0], t_args[1]
                if not self._is_true(state, self.at_ids.get((truck, place))): h += 1
            
            elif t_type == "do_lift_crate":
                crate, hoist = t_args[0], t_args[2]
                if not self._is_true(state, self.lifting_ids.get((hoist, crate))): h += 1

            elif t_type in ["Lift", "Drop", "Load", "Unload", "Drive"]: h += 1
            
            elif t_type != "nop": h += 1
            
        return h
