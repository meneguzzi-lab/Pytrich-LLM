from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Depots domain.

    Bottleneck Analysis:
    The bottleneck in the Depots domain is the multi-step process required to move a crate: 
    clearing the crate, lifting it, loading it into a truck, driving the truck, unloading, 
    clearing the destination, and finally dropping it. This serializes actions around hoists 
    (one crate at a time) and trucks.

    Admissible Lower Bounds:
    1. Goal-Crate Bound (h_goal): Every crate not on its goal surface requires at least 2 actions (Lift, Drop). 
       If the goal surface is at a different location, it requires 4 actions (Lift, Load, Unload, Drop).
    2. Task Network Bound (h_tn): Every non-satisfied task in the network requires a minimum decomposition cost.
       Tasks that cannot take their 'nop' path (e.g., do_put_on when on(c, s) is false) receive a 'nop bonus' penalty.
    """

    def __init__(self, depot_gemini3flash_heuristic_8="depots_gemini3_flash_8"):
        super().__init__(depot_gemini3flash_heuristic_8)
        self.goal_on = {} # crate_id -> surface_id
        self.obj_to_id = {}
        self.id_to_name = {}
        self.on_ids = {} # (c_id, s_id) -> fact_id
        self.at_ids = {} # (l_id, p_id) -> fact_id
        self.in_ids = {} # (c_id, t_id) -> fact_id
        self.clear_ids = {} # s_id -> fact_id
        self.lifting_ids = {} # (h_id, c_id) -> fact_id
        self.pallet_to_place = {} # pallet_id -> place_id
        self.hoist_to_place = {} # hoist_id -> place_id
        self.place_ids = set()
        self.crate_ids = set()
        self.truck_ids = set()
        self.hoist_ids = set()
        self.task_meta = {} # task_global_id -> (type_idx, arg_ids)

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
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map Objects
        obj_names = set()
        for f in model.facts:
            _, args = self._parse_fact_name(f.name)
            obj_names.update(args)
        
        self.obj_to_id = {name: i for i, name in enumerate(sorted(list(obj_names)))}
        self.id_to_name = {i: name for name, i in self.obj_to_id.items()}

        # 2. Parse Facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            arg_ids = [self.obj_to_id[a] for a in args]
            if pred == "on":
                self.on_ids[(arg_ids[0], arg_ids[1])] = f.global_id
                self.crate_ids.add(arg_ids[0])
            elif pred == "at":
                self.at_ids[(arg_ids[0], arg_ids[1])] = f.global_id
                self.place_ids.add(arg_ids[1])
            elif pred == "in":
                self.in_ids[(arg_ids[0], arg_ids[1])] = f.global_id
                self.truck_ids.add(arg_ids[1])
            elif pred == "clear":
                self.clear_ids[arg_ids[0]] = f.global_id
            elif pred == "lifting":
                self.lifting_ids[(arg_ids[0], arg_ids[1])] = f.global_id

        # 3. Recover Static Locations from Operator/Method Names
        for op in model.operators:
            pred, args = self._parse_fact_name(op.name)
            if pred in ["Lift", "Drop", "Load", "Unload"]:
                # Args: hoist, crate, surface/truck, place
                h_id, c_id, s_id, p_id = self.obj_to_id[args[0]], self.obj_to_id[args[1]], self.obj_to_id[args[2]], self.obj_to_id[args[3]]
                self.hoist_to_place[h_id] = p_id
                if "pallet" in args[2]:
                    self.pallet_to_place[s_id] = p_id

        # 4. Parse Goals
        goal_facts = model.state_explicit_repr(model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                self.goal_on[self.obj_to_id[args[0]]] = self.obj_to_id[args[1]]

        # 5. Pre-map Abstract Tasks
        for abt in model.abstract_tasks:
            pred, args = self._parse_fact_name(abt.name)
            arg_ids = [self.obj_to_id[a] for a in args]
            if pred == "do_put_on":
                self.task_meta[abt.global_id] = (1, arg_ids)
            elif pred == "do_clear":
                self.task_meta[abt.global_id] = (2, arg_ids)
            elif pred == "do_get_truck":
                self.task_meta[abt.global_id] = (3, arg_ids)
            else:
                self.task_meta[abt.global_id] = (4, arg_ids)

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _get_loc(self, state, obj_id):
        if obj_id in self.pallet_to_place: return self.pallet_to_place[obj_id]
        if obj_id in self.hoist_to_place: return self.hoist_to_place[obj_id]
        
        # Check at(obj, place)
        for p_id in self.place_ids:
            fid = self.at_ids.get((obj_id, p_id))
            if fid is not None and (state >> fid) & 1:
                return p_id
        
        # Check in(obj, truck)
        for t_id in self.truck_ids:
            fid = self.in_ids.get((obj_id, t_id))
            if fid is not None and (state >> fid) & 1:
                return self._get_loc(state, t_id)
        
        # Check lifting(hoist, obj)
        for h_id in self.hoist_ids:
            fid = self.lifting_ids.get((h_id, obj_id))
            if fid is not None and (state >> fid) & 1:
                return self.hoist_to_place.get(h_id, -1)
        
        return -1

    def _compute(self, state, task_network):
        # Term 1: Goal-based bound
        h_goal = 0
        memo_locs = {}

        def get_loc_memo(oid):
            if oid not in memo_locs:
                memo_locs[oid] = self._get_loc(state, oid)
            return memo_locs[oid]

        for c_id, s_id in self.goal_on.items():
            fid = self.on_ids.get((c_id, s_id))
            if fid is None or not (state >> fid) & 1:
                h_goal += 2
                c_loc = get_loc_memo(c_id)
                s_loc = get_loc_memo(s_id)
                if c_loc != -1 and s_loc != -1 and c_loc != s_loc:
                    h_goal += 2

        # Term 2: Task-based bound with Nop Bonus
        h_tn = 0
        for task in task_network:
            if task.global_id >= self.model.iabt_init and task.global_id <= self.model.iabt_end:
                meta = self.task_meta.get(task.global_id)
                if not meta: 
                    h_tn += 1
                    continue
                t_type, args = meta
                if t_type == 1: # do_put_on(c, s)
                    fid = self.on_ids.get((args[0], args[1]))
                    if fid is None or not (state >> fid) & 1:
                        h_tn += 4 # 2 actions + 2 nop bonus
                elif t_type == 2: # do_clear(s, p)
                    f_clear = self.clear_ids.get(args[0])
                    f_at = self.at_ids.get((args[0], args[1]))
                    is_clear = f_clear is not None and (state >> f_clear) & 1
                    is_at = f_at is not None and (state >> f_at) & 1
                    if not (is_clear and is_at):
                        h_tn += 3 # 1 action + 2 nop bonus
                elif t_type == 3: # do_get_truck(t, p)
                    fid = self.at_ids.get((args[0], args[1]))
                    if fid is None or not (state >> fid) & 1:
                        h_tn += 3 # 1 action + 2 nop bonus
                else:
                    h_tn += 1
            else: # Operator
                h_tn += getattr(task, 'cost', 1)

        return max(h_goal, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
