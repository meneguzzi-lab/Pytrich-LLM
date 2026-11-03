from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

class DepotsGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    Bottleneck Analysis:
    The main serialization in the Depots domain arises from stacking constraints (Blocksworld logic) 
    and location-based transport (Logistics logic). Crates must be cleared before moving, and 
    target surfaces must be cleared before dropping. Hoists and trucks are finite resources 
    at locations.

    Admissible Lower Bounds:
    1. Transport Cost: A crate moving between different locations requires a minimum of 5 actions 
       (Lift, Load, Drive, Unload, Drop). Within the same location, it requires at least 2 (Lift, Drop).
    2. Clear Cost: Every crate above a target or a crate to be moved must be lifted and moved out of the way, 
       representing at least 2 actions (Lift + Load or Lift + Drop) per blocking crate.
    3. Task Network Progress: The remaining tasks in the HTN network represent the unavoidable sequence 
       of decompositions required to reach the goal state.
    """

    def __init__(self, depot_gemini3flash_heuristic_18=None, name="DepotsGemini3FlashHeuristic18"):
        super().__init__(name=name)
        self.on_ids = {}  # (crate, surface) -> fact_id
        self.at_by_locatable = defaultdict(list) # locatable -> [(place, fact_id)]
        self.in_by_crate = defaultdict(list) # crate -> [(truck, fact_id)]
        self.lifting_by_crate = defaultdict(list) # crate -> [(hoist, fact_id)]
        self.clear_ids = {} # surface -> fact_id
        self.task_cache = {} # global_id -> (name, args)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                self.on_ids[(args[0], args[1])] = f.global_id
            elif pred == 'at':
                self.at_by_locatable[args[0]].append((args[1], f.global_id))
            elif pred == 'in':
                self.in_by_crate[args[0]].append((args[1], f.global_id))
            elif pred == 'lifting':
                self.lifting_by_crate[args[1]].append((args[0], f.global_id))
            elif pred == 'clear':
                self.clear_ids[args[0]] = f.global_id

        # Precompute reverse 'on' mapping for height calculations
        self.on_by_base = defaultdict(list)
        for (crate, surface), fid in self.on_ids.items():
            self.on_by_base[surface].append((crate, fid))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"): name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def _get_loc_info(self, item, state):
        # Returns (location_place, in_truck_flag, is_lifting_flag)
        if item in self.at_by_locatable:
            for p, fid in self.at_by_locatable[item]:
                if (state >> fid) & 1: return p, False, False
        if item in self.in_by_crate:
            for t, fid in self.in_by_crate[item]:
                if (state >> fid) & 1:
                    p, _, _ = self._get_loc_info(t, state)
                    return p, True, False
        if item in self.lifting_by_crate:
            for h, fid in self.lifting_by_crate[item]:
                if (state >> fid) & 1:
                    p, _, _ = self._get_loc_info(h, state)
                    return p, False, True
        return None, False, False

    def _get_height(self, s, state):
        if s not in self.clear_ids: return 0
        if (state >> self.clear_ids[s]) & 1: return 0
        h, curr = 0, s
        for _ in range(50):  # Depth safety
            found = False
            for crate, fid in self.on_by_base.get(curr, []):
                if (state >> fid) & 1:
                    h += 1
                    curr = crate
                    found = True
                    break
            if not found: break
        return h

    def _get_task_cost(self, task, state):
        gid = task.global_id
        if gid not in self.task_cache:
            self.task_cache[gid] = self._parse_fact_name(task.name)
        tname, args = self.task_cache[gid]

        # Operator logic
        if gid >= self.model.iop_init and gid <= self.model.iop_end:
            return 0 if tname == 'nop' else 1

        # Abstract Task logic
        if tname == 'do_put_on': # args: crate, target_surface
            c, s = args[0], args[1]
            on_fid = self.on_ids.get((c, s))
            if on_fid is not None and (state >> on_fid) & 1: return 0
            loc_c, in_truck, lifting = self._get_loc_info(c, state)
            loc_s, _, _ = self._get_loc_info(s, state)
            if lifting: dist = 1 if loc_c == loc_s else 4
            elif in_truck: dist = 2 if loc_c == loc_s else 3
            else: dist = 2 if loc_c == loc_s else 5
            return dist + (self._get_height(c, state) * 2) + (self._get_height(s, state) * 2)
        elif tname == 'do_clear':
            return self._get_height(args[0], state) * 2
        elif tname == 'do_load_truck':
            return 2 + (self._get_height(args[0], state) * 2)
        elif tname == 'do_unload_truck':
            return 2 + (self._get_height(args[1], state) * 2)
        elif tname == 'do_get_truck':
            t, p = args[0], args[1]
            loc_t, _, _ = self._get_loc_info(t, state)
            return 0 if loc_t == p else 1
        elif tname == 'do_lift_crate':
            return 1
        return 1

    def _compute(self, state, task_network):
        return sum(self._get_task_cost(task, state) for task in task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
