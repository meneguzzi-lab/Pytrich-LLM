"""
Domain-dependent heuristic for the Depots domain.

Bottleneck Analysis:
The Depots domain is a combination of Logistics and Blocksworld. The primary bottleneck is the serial constraints on hoists 
at each location and the stack dependencies of crates on surfaces. Moving any crate requires it to be clear and a hoist 
to be available at its current location.

Admissible Lower Bounds:
1. Movement Lower Bound: Any crate not at its goal position (not on its goal surface or on an unstable base) 
   requires at least 2 actions (Lift and Drop) if at the correct location, or 4 actions (Lift, Load, Unload, Drop) 
   if a transport to another location is required.
2. Clearing Lower Bound: If a crate is satisfied (at its goal) but is currently blocking an unsatisfied crate 
   below it, it must be moved out and eventually back, costing at least 2 actions.
3. Transport Lower Bound: Each location with at least one crate needing transport to another location 
   requires at least 1 Drive action from a truck.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic9(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_9=None, name="DepotsGemini3FlashHeuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Mapping information
        self.crates = set()
        self.pallets = set()
        self.trucks = set()
        self.hoists = set()
        
        self.on_facts = {}            # (crate, surface) -> global_id
        self.at_place_facts = defaultdict(dict) # locatable -> {place: global_id}
        self.clear_facts = {}         # surface -> global_id
        self.in_truck_facts = defaultdict(dict) # crate -> {truck: global_id}
        self.lifting_facts = defaultdict(dict)  # crate -> {hoist: global_id}
        
        self.goal_on = {}             # crate -> target surface
        self.goal_place = {}          # crate -> target place
        self.pallet_to_place = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # 1. Parse all facts to identify objects and map IDs
        surfaces = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            
            if pred == "on":
                self.crates.add(args[0])
                surfaces.add(args[1])
                self.on_facts[(args[0], args[1])] = f.global_id
            elif pred == "at":
                self.at_place_facts[args[0]][args[1]] = f.global_id
            elif pred == "clear":
                surfaces.add(args[0])
                self.clear_facts[args[0]] = f.global_id
            elif pred == "in":
                self.crates.add(args[0])
                self.trucks.add(args[1])
                self.in_truck_facts[args[0]][args[1]] = f.global_id
            elif pred == "lifting":
                self.hoists.add(args[0])
                self.crates.add(args[1])
                self.lifting_facts[args[1]][args[0]] = f.global_id
            elif pred == "available":
                self.hoists.add(args[0])

        self.pallets = surfaces - self.crates

        # 2. Precompute Pallet Locations (Fixed in Depots)
        for p in self.pallets:
            for place, fid in self.at_place_facts.get(p, {}).items():
                if (initial_node.state >> fid) & 1:
                    self.pallet_to_place[p] = place
                    break

        # 3. Extract Goals and Precompute Goal Places
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # Propagation of goal places
        changed = True
        while changed:
            changed = False
            for c, target_s in self.goal_on.items():
                if c in self.goal_place: continue
                if target_s in self.pallet_to_place:
                    self.goal_place[c] = self.pallet_to_place[target_s]
                    changed = True
                elif target_s in self.goal_place:
                    self.goal_place[c] = self.goal_place[target_s]
                    changed = True

        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

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

    def _get_at_place(self, state, locatable):
        # Direct location
        for place, fid in self.at_place_facts.get(locatable, {}).items():
            if (state >> fid) & 1: return place
        # In truck?
        for truck, fid in self.in_truck_facts.get(locatable, {}).items():
            if (state >> fid) & 1: return self._get_at_place(state, truck)
        # Being lifted?
        for hoist, fid in self.lifting_facts.get(locatable, {}).items():
            if (state >> fid) & 1: return self._get_at_place(state, hoist)
        return None

    def _compute(self, state):
        # 1. Map current relationships
        on_now = {}
        for (c, s), fid in self.on_facts.items():
            if (state >> fid) & 1: on_now[c] = s

        # 2. Fixed-point satisfaction check
        satisfied = set(self.pallets)
        memo_sat = {}
        def is_satisfied(c):
            if c in satisfied: return True
            if c in memo_sat: return memo_sat[c]
            memo_sat[c] = False
            curr_s = on_now.get(c)
            goal_s = self.goal_on.get(c)
            if curr_s and goal_s and curr_s == goal_s:
                if is_satisfied(curr_s):
                    memo_sat[c] = True
            return memo_sat[c]

        for c in self.crates: is_satisfied(c)
        final_satisfied = {c for c, sat in memo_sat.items() if sat} | self.pallets

        # 3. Heuristic Calculation
        h = 0
        places_needing_truck = set()
        for c in self.crates:
            if c not in final_satisfied:
                # Crate must be lifted and dropped
                h += 2
                curr_p = self._get_at_place(state, c)
                goal_p = self.goal_place.get(c)
                if curr_p and goal_p and curr_p != goal_p:
                    # Must be loaded and unloaded
                    h += 2
                    places_needing_truck.add(curr_p)
                
                # If not clear, we must move something off it
                # However, if that 'something' is also unsatisfied, we've already counted it.
                # We only add cost if the blocker is satisfied.
                cfid = self.clear_facts.get(c)
                if cfid is not None and not ((state >> cfid) & 1):
                    # Find blocker
                    blocker = None
                    for possible_c in self.crates:
                        if on_now.get(possible_c) == c:
                            blocker = possible_c
                            break
                    if blocker and blocker in final_satisfied:
                        h += 2
            else:
                # Crate is satisfied, but is it blocking an unsatisfied crate?
                curr_base = on_now.get(c)
                while curr_base and curr_base in self.crates:
                    if curr_base not in final_satisfied:
                        h += 2 # Must move c out of way
                        break
                    curr_base = on_now.get(curr_base)
        
        h += len(places_needing_truck)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
