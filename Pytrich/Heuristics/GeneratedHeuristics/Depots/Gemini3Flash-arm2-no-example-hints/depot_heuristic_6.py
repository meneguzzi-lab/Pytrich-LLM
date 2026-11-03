from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Depots domain is the sequential access to hoists for lifting and 
    dropping crates, combined with the clearing of stacked crates. Only one hoist can lift one 
    crate at a time, and a surface must be clear before dropping another crate onto it. 
    Additionally, trucks serve as a physical bottleneck for moving crates between locations.

    Admissibility Sketches:
    1. htn_cost: Each task in the network requires a minimum number of actions to satisfy.
       - do_put_on(c, s): Requires at least a Lift and a Drop (2). If at different places,
         requires Load, Drive, and Unload (+3). If top isn't clear, requires clearing (+2 per surface).
       - do_clear(s, p): Requires moving at least one crate (2).
    2. goal_cost: Each unsatisfied goal (on c s) requires at least one Lift and one Drop action (2).

    Heuristic is the max of these two components, ensuring admissibility while providing guidance.
    """
    
    def __init__(self, depot_gemini3flash_heuristic_6="DepotsGemini3FlashHeuristic6"):
        super().__init__(depot_gemini3flash_heuristic_6)
        self.on_ids = {}  # crate -> surface -> fact_id
        self.at_ids = {}  # locatable -> place -> fact_id
        self.clear_ids = {} # surface -> fact_id
        self.in_ids = {} # crate -> truck -> fact_id
        self.lifting_ids_by_crate = {} # crate -> hoist -> fact_id
        self.lifting_ids_by_hoist = {} # hoist -> crate -> fact_id
        self.task_data = {} # task_id -> parsed_info
        self.goal_ids = []
        self.crates = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()

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
        # 1. Map predicates to fact IDs
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on':
                c, s = args[0], args[1]
                self.on_ids.setdefault(c, {})[s] = fact.global_id
                self.crates.add(c)
            elif pred == 'at':
                loc, p = args[0], args[1]
                self.at_ids.setdefault(loc, {})[p] = fact.global_id
                self.places.add(p)
            elif pred == 'clear':
                self.clear_ids[args[0]] = fact.global_id
            elif pred == 'in':
                c, t = args[0], args[1]
                self.in_ids.setdefault(c, {})[t] = fact.global_id
                self.trucks.add(t)
            elif pred == 'lifting':
                h, c = args[0], args[1]
                self.lifting_ids_by_crate.setdefault(c, {})[h] = fact.global_id
                self.lifting_ids_by_hoist.setdefault(h, {})[c] = fact.global_id
                self.hoists.add(h)

        # 2. Parse Goal IDs
        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on':
                gid = self.on_ids.get(args[0], {}).get(args[1])
                if gid is not None: self.goal_ids.append(gid)

        # 3. Pre-parse task info
        for task in model.abstract_tasks:
            self.task_data[task.global_id] = self._parse_name(task.name)
        for op in model.operators:
            self.task_data[op.global_id] = self._parse_name(op.name)

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _parse_name(self, name):
        if '[' not in name: return {'type': 'nop', 'args': []}
        base = name[:name.find('[')]
        args = name[name.find('[')+1:name.find(']')].split(',')
        return {'type': base, 'args': args}

    def _get_place(self, state, item):
        for p, fid in self.at_ids.get(item, {}).items():
            if (state >> fid) & 1: return p
        if item in self.crates:
            for t, fid in self.in_ids.get(item, {}).items():
                if (state >> fid) & 1: return self._get_place(state, t)
            for h, fid in self.lifting_ids_by_crate.get(item, {}).items():
                if (state >> fid) & 1: return self._get_place(state, h)
        return None

    def _is_lifting(self, state, crate):
        for h, fid in self.lifting_ids_by_crate.get(crate, {}).items():
            if (state >> fid) & 1: return True
        return False

    def _is_in_truck(self, state, crate):
        for t, fid in self.in_ids.get(crate, {}).items():
            if (state >> fid) & 1: return True
        return False

    def _estimate_task_cost(self, state, tid):
        info = self.task_data.get(tid)
        if not info: return 0
        t_type, args = info['type'], info['args']
        if t_type == 'do_put_on':
            c, s = args[0], args[1]
            on_id = self.on_ids.get(c, {}).get(s)
            if on_id is not None and (state >> on_id) & 1: return 0
            cost = 2
            p_c, p_s = self._get_place(state, c), self._get_place(state, s)
            if p_c and p_s and p_c != p_s: cost += 3
            if not self._is_lifting(state, c) and not self._is_in_truck(state, c):
                cid = self.clear_ids.get(c)
                if cid is not None and not ((state >> cid) & 1): cost += 2
            sid = self.clear_ids.get(s)
            if sid is not None and not ((state >> sid) & 1): cost += 2
            return cost
        elif t_type == 'do_clear':
            sid = self.clear_ids.get(args[0])
            if sid is not None and (state >> sid) & 1: return 0
            return 2
        elif t_type == 'do_get_truck':
            aid = self.at_ids.get(args[0], {}).get(args[1])
            if aid is not None and (state >> aid) & 1: return 0
            return 1
        elif t_type in ['Lift', 'Unload', 'Load', 'Drop', 'Drive']: return 1
        return 0

    def _compute(self, state, tn):
        htn_cost = sum(self._estimate_task_cost(state, t.global_id) for t in tn)
        goal_cost = 0
        for gid in self.goal_ids:
            if not ((state >> gid) & 1): goal_cost += 2
        return max(htn_cost, goal_cost)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
