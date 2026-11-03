from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic17(Heuristic):
    """
    Heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The bottleneck in the Depots domain is the coordination of hoists and trucks to manipulate crates 
    across different locations while managing stack constraints (Blocksworld logic) at each location. 
    Only one hoist per location exists, and trucks must be positioned correctly to transport crates.
    
    Admissible Lower Bounds:
    1. Crate Transit Cost: Minimum actions to move a crate from its current status (at place, in truck, 
       or being lifted) to its target surface. If locations differ, additional Load/Drive/Unload actions are required.
    2. Clearance Cost: Actions required to clear target surfaces and the crates themselves. If a surface is not 
       clear, at least one Lift+Drop/Load (2 actions) must occur for the obstructing crate.
    """
    
    def __init__(self, depot_gemini3flash_heuristic_17="DepotsGemini3FlashHeuristic17"):
        super().__init__(name=depot_gemini3flash_heuristic_17)
        self.at_id = {}  # locatable -> {place -> global_id}
        self.on_id = {}  # crate -> {surface -> global_id}
        self.in_id = {}  # crate -> {truck -> global_id}
        self.clear_id = {}  # surface -> global_id
        self.lifting_any_id = {} # crate -> {hoist -> global_id}

    def initialize(self, model: Model, initial_node: HTNNode):
        for fact in model.facts:
            prefix, pred, args = self._parse_fact_name(fact.name)
            if prefix == "+":
                if pred == "at":
                    loc, p = args[0], args[1]
                    if loc not in self.at_id: self.at_id[loc] = {}
                    self.at_id[loc][p] = fact.global_id
                elif pred == "on":
                    c, s = args[0], args[1]
                    if c not in self.on_id: self.on_id[c] = {}
                    self.on_id[c][s] = fact.global_id
                elif pred == "in":
                    c, t = args[0], args[1]
                    if c not in self.in_id: self.in_id[c] = {}
                    self.in_id[c][t] = fact.global_id
                elif pred == "clear":
                    self.clear_id[args[0]] = fact.global_id
                elif pred == "lifting":
                    h, c = args[0], args[1]
                    if c not in self.lifting_any_id: self.lifting_any_id[c] = {}
                    self.lifting_any_id[c][h] = fact.global_id
        
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state: int, task_network) -> int:
        h = 0
        for task in task_network:
            t_name, t_args = self._parse_task_name(task.name)
            if t_name == "nop":
                continue
            
            if t_name == "do_put_on":
                c, s = t_args[0], t_args[1]
                # If goal not reached
                if not self._check(state, self.on_id.get(c, {}).get(s)):
                    h += 2 # Base: Lift + Drop
                    p_c = self._find_at(state, c)
                    p_s = self._find_at(state, s)
                    if p_c and p_s and p_c != p_s:
                        h += 3 # Load + Drive + Unload
                    elif self._is_in_truck(state, c):
                        t = self._find_truck_containing(state, c)
                        p_t = self._find_at(state, t)
                        if p_t and p_s and p_t != p_s:
                            h += 2 # Drive + Unload
                        else:
                            h += 1 # Unload
                    
                    # Obstruction penalty
                    if not self._check(state, self.clear_id.get(c)):
                        h += 1
                    if not self._check(state, self.clear_id.get(s)):
                        h += 1
            elif t_name == "do_clear":
                s = t_args[0]
                if not self._check(state, self.clear_id.get(s)):
                    h += 2 # Lift + Drop/Load some crate
            elif t_name == "do_get_truck":
                t, p = t_args[0], t_args[1]
                if not self._check(state, self.at_id.get(t, {}).get(p)):
                    h += 1 # Drive
            elif t_name in ["Lift", "Drop", "Load", "Unload", "Drive"]:
                h += 1
            elif t_name in ["do_lift_crate", "do_load_truck", "do_unload_truck"]:
                h += 2

        return h

    def _check(self, state, fact_id):
        if fact_id is None: return False
        return (state >> fact_id) & 1

    def _find_at(self, state, locatable):
        if locatable in self.at_id:
            for p, f_id in self.at_id[locatable].items():
                if (state >> f_id) & 1:
                    return p
        # Check if crate is in a truck or being lifted
        if locatable in self.in_id:
            for t, f_id in self.in_id[locatable].items():
                if (state >> f_id) & 1:
                    return self._find_at(state, t)
        if locatable in self.lifting_any_id:
            for h, f_id in self.lifting_any_id[locatable].items():
                if (state >> f_id) & 1:
                    return self._find_at(state, h)
        return None

    def _is_in_truck(self, state, crate):
        if crate not in self.in_id: return False
        for f_id in self.in_id[crate].values():
            if (state >> f_id) & 1: return True
        return False

    def _find_truck_containing(self, state, crate):
        if crate not in self.in_id: return None
        for t, f_id in self.in_id[crate].items():
            if (state >> f_id) & 1: return t
        return None

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return prefix, predicate, args
        return None, None, []

    def _parse_task_name(self, name: str):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []
