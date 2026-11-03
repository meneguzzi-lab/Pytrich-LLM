"""
Domain-dependent heuristic for the Depots domain.

Bottleneck Analysis:
    The bottleneck in the Depots domain is the sequential stacking and transport of crates.
    Each crate movement requires clearing all crates above it, lifting it with a local hoist,
    potentially transporting it via truck (if the target place is different), and then
    clearing the destination surface before dropping it. This is a hybrid of Blocksworld
    and Logistics constraints.

Admissibility Sketches:
    1. PutOn Cost: The minimum number of actions to place a crate 'c' on surface 's' is:
       - 0 if already (on c s).
       - 2 if at same place (Lift, Drop) + costs to clear 'c' and 's'.
       - 5 if at different places (Lift, Load, Drive, Unload, Drop) + clearance costs.
    2. Clearance Cost: Every crate 'c' on top of a surface 's' must be moved at least once (2 actions: Lift, Drop/Load)
       to make 's' clear. This is a hard lower bound on the number of actions required.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic6(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_6=None, name="DepotsGemini3FlashHeuristic6"):
        super().__init__(name=name)
        self.on_map = defaultdict(list) # crate -> [(fact_id, surface)]
        self.on_top_of_map = defaultdict(list) # surface -> [(fact_id, crate)]
        self.at_map = defaultdict(list) # locatable -> [(fact_id, place)]
        self.in_map = defaultdict(list) # crate -> [(fact_id, truck)]
        self.lifting_map = defaultdict(list) # crate -> [(fact_id, hoist)]
        self.pallet_to_place = {} # pallet -> place (static)
        self.all_trucks = set()
        
    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "on":
                self.on_map[args[0]].append((fact.global_id, args[1]))
                self.on_top_of_map[args[1]].append((fact.global_id, args[0]))
            elif pred == "at":
                self.at_map[args[0]].append((fact.global_id, args[1]))
                if "pallet" in args[0].lower() or "depot" in args[1].lower() or "distributor" in args[1].lower():
                    if "pallet" in args[0].lower():
                        self.pallet_to_place[args[0]] = args[1]
                if "truck" in args[0].lower():
                    self.all_trucks.add(args[0])
            elif pred == "in":
                self.in_map[args[0]].append((fact.global_id, args[1]))
            elif pred == "lifting":
                self.lifting_map[args[1]].append((fact.global_id, args[0]))

        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

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

    def _get_place(self, state, locatable):
        if locatable in self.pallet_to_place:
            return self.pallet_to_place[locatable]
        for fid, place in self.at_map.get(locatable, []):
            if (state >> fid) & 1: return place
        return None

    def _get_loc_info(self, state, crate):
        for fid, truck in self.in_map.get(crate, []):
            if (state >> fid) & 1: return 'truck', truck, self._get_place(state, truck)
        for fid, hoist in self.lifting_map.get(crate, []):
            if (state >> fid) & 1: return 'hoist', hoist, self._get_place(state, hoist)
        for fid, surf in self.on_map.get(crate, []):
            if (state >> fid) & 1:
                p = self._get_place(state, surf)
                return 'place', surf, p
        return None, None, None

    def _get_depth(self, state, surface):
        depth = 0
        curr = surface
        visited = {curr}
        while True:
            found = False
            for fid, crate in self.on_top_of_map.get(curr, []):
                if (state >> fid) & 1:
                    depth += 1
                    curr = crate
                    if curr in visited: break
                    visited.add(curr)
                    found = True
                    break
            if not found: break
        return depth

    def _estimate_abstract_task(self, state, task_name, args):
        if task_name == "nop": return 0
        
        if task_name == "do_put_on":
            if not args or len(args) < 2: return 0
            crate, target_surf = args[0], args[1]
            
            # Check if goal already met
            for fid, s in self.on_map.get(crate, []):
                if (state >> fid) & 1 and s == target_surf: return 0
            
            # Estimate movement cost
            type_c, cur_c, place_c = self._get_loc_info(state, crate)
            place_s = self._get_place(state, target_surf)
            
            base_cost = 0
            if type_c == 'hoist': base_cost = 1 # Just Drop
            elif type_c == 'truck':
                base_cost = 2 # Unload, Drop
                if place_c != place_s: base_cost += 1 # Drive
            else:
                if place_c == place_s: base_cost = 2 # Lift, Drop
                else: base_cost = 5 # Lift, Load, Drive, Unload, Drop
            
            clear_c = self._get_depth(state, crate) * 2
            clear_s = self._get_depth(state, target_surf) * 2
            return base_cost + clear_c + clear_s

        if task_name == "do_clear":
            return self._get_depth(state, args[0]) * 2 if args else 0
        
        if task_name == "do_load_truck":
            return 2 + self._get_depth(state, args[0]) * 2 if args else 2
            
        if task_name == "do_unload_truck":
            return 2 + self._get_depth(state, args[1]) * 2 if args else 2

        return 1

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                h += task.cost
            else:
                # Abstract Task grounded name: name[arg1,arg2]
                t_name, args = self._parse_grounded_name(task.name)
                h += self._estimate_abstract_task(state, t_name, args)
        return h

    def _parse_grounded_name(self, name):
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            return name[:bracket_idx], name[bracket_idx+1:-1].split(",")
        return name, []

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
