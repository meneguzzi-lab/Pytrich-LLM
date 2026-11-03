from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class DepotsGemini3FlashHeuristic16(Heuristic):
    """
    Heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The main bottleneck is the single hoist available per place (depot or distributor) and the stack structure 
    of crates on pallets. Only one action (Lift/Drop/Load/Unload) can happen at a time per hoist, and crates 
    must be moved in a specific order if they are stacked.
    
    Admissible Lower Bounds:
    1. Task Cost: Each task in the network (e.g., do_put_on) requires a minimum number of physical actions 
       (Lift, Load, Drive, Unload, Drop) depending on the crate's current state (at location, in truck, or lifting).
    2. Stack Clearing: If a crate or pallet is not clear, it needs at least 2 actions (Lift and Drop/Load) 
       for every crate currently on top of it to clear it.
    3. Goal Progress: Any goal 'on' fact not yet satisfied and not currently being handled by the task network 
       requires at least 2 actions.
    """

    def __init__(self, depot_gemini3flash_heuristic_16="DepotsHeuristic"):
        super().__init__(name=depot_gemini3flash_heuristic_16)
        self.at_fact_ids = defaultdict(dict)
        self.on_fact_ids = defaultdict(dict)
        self.in_fact_ids = defaultdict(dict)
        self.clear_fact_ids = {}
        self.available_fact_ids = {}
        self.lifting_fact_ids = defaultdict(dict)
        self.lifting_by_crate = defaultdict(list)
        self.all_places = set()
        self.all_trucks = set()
        self.all_hoists = set()
        self.goal_on_facts = []

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

    def _parse_task_name(self, name: str):
        parts = name.split()
        if not parts: return "", []
        return parts[0], parts[1:]

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at":
                l, p = args
                self.at_fact_ids[l][p] = f.global_id
                self.all_places.add(p)
            elif pred == "on":
                c, s = args
                self.on_fact_ids[c][s] = f.global_id
            elif pred == "in":
                c, t = args
                self.in_fact_ids[c][t] = f.global_id
                self.all_trucks.add(t)
            elif pred == "clear":
                self.clear_fact_ids[args[0]] = f.global_id
            elif pred == "lifting":
                h, c = args
                self.lifting_fact_ids[h][c] = f.global_id
                self.lifting_by_crate[c].append((h, f.global_id))
                self.all_hoists.add(h)
            elif pred == "available":
                self.available_fact_ids[args[0]] = f.global_id
                self.all_hoists.add(args[0])

        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                c, s = args
                if c in self.on_fact_ids and s in self.on_fact_ids[c]:
                    self.goal_on_facts.append((c, s, self.on_fact_ids[c][s]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_location(self, obj, state, memo, depth=0):
        if obj in memo: return memo[obj]
        if depth > 3: return None
        if obj in self.at_fact_ids:
            for p, fid in self.at_fact_ids[obj].items():
                if (state >> fid) & 1:
                    memo[obj] = p
                    return p
        if obj in self.in_fact_ids:
            for t, fid in self.in_fact_ids[obj].items():
                if (state >> fid) & 1:
                    loc = self._get_location(t, state, memo, depth + 1)
                    memo[obj] = loc
                    return loc
        if obj in self.lifting_by_crate:
            for h, fid in self.lifting_by_crate[obj]:
                if (state >> fid) & 1:
                    loc = self._get_location(h, state, memo, depth + 1)
                    memo[obj] = loc
                    return loc
        return None

    def _compute(self, state, task_network) -> int:
        h_val = 0
        tn_names = set()
        memo = {}

        for task in task_network:
            tn_names.add(task.name)
            if hasattr(task, 'cost') and task.cost is not None:
                base_cost = task.cost
            else:
                base_cost = 1

            t_type, t_args = self._parse_task_name(task.name)
            if t_type == "do_put_on" and len(t_args) >= 2:
                c, s = t_args[0], t_args[1]
                if c in self.on_fact_ids and s in self.on_fact_ids[c]:
                    if (state >> self.on_fact_ids[c][s]) & 1: continue
                
                # Minimum work to put c on s
                cost = 2 # Default Lift + Drop
                loc_s = self._get_location(s, state, memo)
                
                lifted_by = None
                if c in self.lifting_by_crate:
                    for h, fid in self.lifting_by_crate[c]:
                        if (state >> fid) & 1: lifted_by = h; break
                
                if lifted_by:
                    cost = 1 # Just Drop
                    loc_c = self._get_location(lifted_by, state, memo)
                    if loc_c and loc_s and loc_c != loc_s: cost += 3
                else:
                    in_truck = None
                    if c in self.in_fact_ids:
                        for t, fid in self.in_fact_ids[c].items():
                            if (state >> fid) & 1: in_truck = t; break
                    if in_truck:
                        cost = 2 # Unload + Drop
                        loc_c = self._get_location(in_truck, state, memo)
                        if loc_c and loc_s and loc_c != loc_s: cost += 1
                    else:
                        cost = 2 # Lift + Drop
                        loc_c = self._get_location(c, state, memo)
                        if loc_c and loc_s and loc_c != loc_s: cost += 3
                h_val += cost
            elif t_type == "do_clear" and len(t_args) >= 1:
                s = t_args[0]
                if s in self.clear_fact_ids and not ((state >> self.clear_fact_ids[s]) & 1):
                    h_val += 2
            elif t_type == "do_get_truck" and len(t_args) >= 2:
                t, p = t_args[0], t_args[1]
                if t in self.at_fact_ids and p in self.at_fact_ids[t]:
                    if not ((state >> self.at_fact_ids[t][p]) & 1): h_val += 1
            elif t_type == "do_load_truck": h_val += 2
            elif t_type == "do_unload_truck": h_val += 2
            else: h_val += base_cost

        for c, s, fid in self.goal_on_facts:
            if not ((state >> fid) & 1):
                if f"do_put_on {c} {s}" not in tn_names:
                    h_val += 2

        return h_val
