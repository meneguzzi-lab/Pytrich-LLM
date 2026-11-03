from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class DepotsGemini3FlashHeuristic1(Heuristic):
    """
    Heuristic for the Depots domain.
    
    Bottleneck Analysis:
    The main serialization in the Depots domain is the stack-based access to crates (Blocksworld style)
    coupled with the requirement of hoists and trucks to move crates between locations (Logistics style).
    Accessing a crate requires clearing everything above it; moving it requires a hoist and a truck if
    changing depots, and dropping it requires the destination surface to be clear.

    Admissibility Sketches:
    1. do_put_on(c, s) cost: A crate c requires at least a Lift and a Drop (2 actions) to be placed on s
       if they are at the same location. If at different locations, it requires Lift, Load, Unload, Drop
       (4 actions) plus at least one Drive action for the truck.
    2. Clear(s) cost: To clear a surface s, every crate x in the stack above s must be moved. 
       Each such crate x requires at least a Lift and a Drop/Load elsewhere (2 actions).
    3. These estimates are independent for each unique crate commitment in the task network,
       making the sum of these costs a lower bound on the remaining work.
    """

    def __init__(self, depot_gemini3flash_heuristic_1="depot_gemini3flash_heuristic_1"):
        super().__init__(name=depot_gemini3flash_heuristic_1)
        self.at_lookup = {}       # obj -> {place: global_id}
        self.on_lookup = {}       # bottom -> {top: global_id}
        self.in_lookup = {}       # crate -> {truck: global_id}
        self.lifting_lookup = {}  # hoist -> {crate: global_id}
        self.fixed_at = {}        # hoist/pallet -> place
        self.places = set()
        self.goal_on_crates = []

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

    def _parse_task_name(self, name):
        name = name.replace('(', '').replace(')', '')
        if '[' in name:
            match = re.match(r"([^\[]+)\[([^\]]+)\]", name)
            if match:
                return match.group(1), match.group(2).split(',')
        parts = name.split()
        if len(parts) > 1:
            return parts[0], parts[1:]
        return name, []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'at':
                obj, place = args[0], args[1]
                if obj not in self.at_lookup: self.at_lookup[obj] = {}
                self.at_lookup[obj][place] = f.global_id
                self.places.add(place)
                if 'hoist' in obj or 'pallet' in obj:
                    self.fixed_at[obj] = place
            elif pred == 'on':
                top, bottom = args[0], args[1]
                if bottom not in self.on_lookup: self.on_lookup[bottom] = {}
                self.on_lookup[bottom][top] = f.global_id
            elif pred == 'in':
                crate, truck = args[0], args[1]
                if crate not in self.in_lookup: self.in_lookup[crate] = {}
                self.in_lookup[crate][truck] = f.global_id
            elif pred == 'lifting':
                hoist, crate = args[0], args[1]
                if hoist not in self.lifting_lookup: self.lifting_lookup[hoist] = {}
                self.lifting_lookup[hoist][crate] = f.global_id

        goal_names = model.state_explicit_repr(model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on':
                self.goal_on_crates.append((args[0], args[1]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _get_loc(self, obj, state):
        if obj in self.places: return obj
        if obj in self.fixed_at: return self.fixed_at[obj]
        for place, fid in self.at_lookup.get(obj, {}).items():
            if (state >> fid) & 1: return place
        for truck, fid in self.in_lookup.get(obj, {}).items():
            if (state >> fid) & 1:
                for tp, tfid in self.at_lookup.get(truck, {}).items():
                    if (state >> tfid) & 1: return tp
        for hoist, crates in self.lifting_lookup.items():
            for crate, fid in crates.items():
                if (state >> fid) & 1: return self.fixed_at.get(hoist)
        return None

    def _check_on(self, top, bottom, state):
        fid = self.on_lookup.get(bottom, {}).get(top)
        return fid is not None and (state >> fid) & 1

    def _count_above(self, surface, state):
        count = 0
        curr = surface
        while True:
            found = False
            for top, fid in self.on_lookup.get(curr, {}).items():
                if (state >> fid) & 1:
                    count += 1
                    curr = top
                    found = True
                    break
            if not found: break
        return count

    def _estimate_puton(self, c, s, state):
        if self._check_on(c, s, state): return 0
        loc_c = self._get_loc(c, state)
        loc_s = self._get_loc(s, state)
        cost = 2 if loc_c == loc_s else 4
        cost += 2 * self._count_above(c, state)
        cost += 2 * self._count_above(s, state)
        return cost

    def _compute(self, state, task_network):
        h = 0
        active_putons = set()
        for task in task_network:
            t_name, t_args = self._parse_task_name(task.name)
            if t_name == 'do_put_on' and len(t_args) >= 2:
                c, s = t_args[0], t_args[1]
                active_putons.add((c, s))
                h += self._estimate_puton(c, s, state)
            elif t_name == 'do_clear' and len(t_args) >= 1:
                h += 2 * self._count_above(t_args[0], state)
            elif t_name == 'do_get_truck' and len(t_args) >= 2:
                t, p = t_args[0], t_args[1]
                fid = self.at_lookup.get(t, {}).get(p)
                if fid is not None and not ((state >> fid) & 1): h += 1
            elif t_name in ['do_load_truck', 'do_unload_truck']: h += 2
            elif t_name in ['Drive', 'Lift', 'Drop', 'Load', 'Unload']: h += 1
        
        for c, s in self.goal_on_crates:
            if (c, s) not in active_putons and not self._check_on(c, s, state):
                h += self._estimate_puton(c, s, state)
        return h
