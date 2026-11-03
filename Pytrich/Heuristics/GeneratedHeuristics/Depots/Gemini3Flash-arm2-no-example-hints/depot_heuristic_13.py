from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic13(Heuristic):
    """
    Domain-Dependent Heuristic for the Depots Domain.
    
    Bottleneck Analysis:
    The primary bottleneck in Depots is the availability of Hoists and the serialization of work 
    required to clear crate stacks before they can be moved via trucks to target surfaces. 
    The domain combines logistics (truck transport) with blocksworld (stacking/clearing).
    
    Admissibility Sketches:
    1. Goal-Based Cost: For each 'on(crate, surface)' goal, we estimate the minimum actions (Lift + Drop = 2, 
       plus transport if locations differ). This is a lower bound on the plan length.
    2. Task-Based Cost: Each non-satisfied HTN task requires a minimum number of operators to resolve 
       (e.g., do_put_on requires at least 4 actions if moving locations). 
    3. Nop Bonus: As per hints, tasks that cannot take the 'nop' path incur a +2 penalty, reflecting 
       the minimum decomposition cost into non-trivial actions.
    """

    def __init__(self, depot_gemini3flash_heuristic_13="DepotsHeuristic"):
        super().__init__(depot_gemini3flash_heuristic_13)
        self.fact_map = {'on': {}, 'at': {}, 'in': {}, 'lifting': {}, 'clear': {}}
        self.goal_ons = []
        self.at_ids = {} # locatable -> {place: id}
        self.in_ids = {} # crate -> {truck: id}
        self.lifting_ids = {} # crate -> {hoist: id}
        self.clear_ids = {} # surface -> id
        self.on_ids = {} # (crate, surface) -> id
        self.task_data = {} # task_id -> (name, args)
        self.places = set()
        self.trucks = set()
        self.hoists = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            if pred == 'at':
                obj, p = args[0], args[1]
                if obj not in self.at_ids: self.at_ids[obj] = {}
                self.at_ids[obj][p] = f.global_id
                self.places.add(p)
            elif pred == 'on':
                c, s = args[0], args[1]
                self.on_ids[(c, s)] = f.global_id
            elif pred == 'in':
                c, t = args[0], args[1]
                if c not in self.in_ids: self.in_ids[c] = {}
                self.in_ids[c][t] = f.global_id
                self.trucks.add(t)
            elif pred == 'lifting':
                h, c = args[0], args[1]
                if c not in self.lifting_ids: self.lifting_ids[c] = {}
                self.lifting_ids[c][h] = f.global_id
                self.hoists.add(h)
            elif pred == 'clear':
                self.clear_ids[args[0]] = f.global_id

        # 2. Identify Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on" and len(args) == 2:
                c, s = args[0], args[1]
                fid = self.on_ids.get((c, s))
                if fid is not None: self.goal_ons.append((c, s, fid))

        # 3. Task Metadata
        for t in model.abstract_tasks:
            name, args = self._parse_grounded_name(t.name)
            self.task_data[t.global_id] = (name, args)
        for op in model.operators:
            name, args = self._parse_grounded_name(op.name)
            self.task_data[op.global_id] = (name, args)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        h_tn = 0
        for task in task_network:
            t_type, t_args = self.task_data.get(task.global_id, (None, []))
            if not t_type: continue
            if t_type == 'do_put_on':
                c, s = t_args[0], t_args[1]
                on_fid = self.on_ids.get((c, s))
                if on_fid is None or not ((state >> on_fid) & 1):
                    cost = 4 # Lift, Load, Unload, Drop base
                    c_loc = self._get_location(c, state)
                    s_loc = self._get_location(s, state)
                    if c_loc and s_loc and c_loc != s_loc: cost += 1 # Drive
                    if not self._is_clear(c, state): cost += 2
                    if not self._is_clear(s, state): cost += 2
                    h_tn += cost + 2 # Nop bonus
            elif t_type == 'do_clear':
                s = t_args[0]
                if not self._is_clear(s, state): h_tn += 2 + 2
            elif t_type == 'do_get_truck':
                t, p = t_args[0], t_args[1]
                at_fid = self.at_ids.get(t, {}).get(p)
                if at_fid is None or not ((state >> at_fid) & 1): h_tn += 1 + 2
            elif t_type in ['do_lift_crate', 'do_load_truck', 'do_unload_truck']: h_tn += 2 + 2

        h_goal = 0
        for c, s, fid in self.goal_ons:
            if not ((state >> fid) & 1):
                cost = 4
                c_loc = self._get_location(c, state)
                s_loc = self._get_location(s, state)
                if c_loc and s_loc and c_loc != s_loc: cost += 1
                h_goal += cost

        return max(h_tn, h_goal)

    def _get_location(self, obj, state):
        if obj in self.places: return obj
        if obj in self.at_ids:
            for p, fid in self.at_ids[obj].items():
                if (state >> fid) & 1: return p
        if obj in self.in_ids:
            for t, fid in self.in_ids[obj].items():
                if (state >> fid) & 1:
                    if t in self.at_ids:
                        for p, tfid in self.at_ids[t].items():
                            if (state >> tfid) & 1: return p
        if obj in self.lifting_ids:
            for h, fid in self.lifting_ids[obj].items():
                if (state >> fid) & 1:
                    if h in self.at_ids:
                        for p, hfid in self.at_ids[h].items():
                            if (state >> hfid) & 1: return p
        return None

    def _is_clear(self, obj, state):
        fid = self.clear_ids.get(obj)
        return (state >> fid) & 1 if fid is not None else True

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                return predicate, args
        return None, []

    def _parse_grounded_name(self, name: str):
        name = name.strip()
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []
