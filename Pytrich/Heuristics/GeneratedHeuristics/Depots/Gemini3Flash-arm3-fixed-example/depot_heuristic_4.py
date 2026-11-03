"""
Domain-dependent heuristic for the Depots domain.

Bottleneck analysis:
The bottleneck in the Depots domain is the serialization of crate movement at each location 
due to limited hoist availability and the clearing of stacks, combined with the 
transportation overhead between different locations using a limited number of trucks.

Admissibility sketches:
1. Crate Manipulation: Every crate not currently at its goal stack must be Lifted and Dropped. 
   At a minimum, this requires 2 actions (Lift + Drop).
2. Transportation: If a crate's current location differs from its goal location, it must be 
   Loaded into a truck and Unloaded at the destination. This adds at least 2 more actions 
   (Load + Unload), excluding the shared Drive actions.
3. Stability: If a crate is correctly placed on its target surface but that surface is itself 
   an unstable crate, the top crate must eventually be moved to allow the stack underneath 
   to be corrected, then placed back. This adds at least 2 actions.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic4(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_4=None, name="DepotsGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.depot_gemini3flash_heuristic_4 = depot_gemini3flash_heuristic_4
        self.preprocessing_time = 0
        
        # Precomputed maps
        self.crates = set()
        self.pallets = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()
        
        self.goal_on = {} # crate -> goal_surface
        self.goal_place = {} # crate -> goal_place_name
        
        self.at_facts = defaultdict(dict) # obj -> {place: fact_id}
        self.in_facts = defaultdict(dict) # crate -> {truck: fact_id}
        self.lifting_facts = defaultdict(dict) # crate -> {hoist: fact_id}
        self.on_facts = defaultdict(dict) # crate -> {surface: fact_id}
        self.clear_facts = {} # surface -> fact_id
        self.pallet_at = {} # pallet -> place_name

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # 1. Parse goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gf in goal_fact_names:
                pred, args = self._parse_fact_name(gf)
                if pred == "on" and len(args) == 2:
                    self.goal_on[args[0]] = args[1]

        # 2. Map all facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "at":
                obj, loc = args[0], args[1]
                self.at_facts[obj][loc] = f.global_id
                self.places.add(loc)
                # Rough classification by name if types aren't explicit
                if "pallet" in obj: self.pallets.add(obj)
                elif "truck" in obj: self.trucks.add(obj)
                elif "hoist" in obj: self.hoists.add(obj)
                elif "crate" in obj: self.crates.add(obj)
            elif pred == "on":
                self.on_facts[args[0]][args[1]] = f.global_id
                self.crates.add(args[0])
            elif pred == "in":
                self.in_facts[args[0]][args[1]] = f.global_id
                self.crates.add(args[0])
                self.trucks.add(args[1])
            elif pred == "lifting":
                self.lifting_facts[args[1]][args[0]] = f.global_id
                self.crates.add(args[1])
                self.hoists.add(args[0])
            elif pred == "clear":
                self.clear_facts[args[0]] = f.global_id

        # 3. Fix pallet locations (static)
        for p in self.pallets:
            for loc, fid in self.at_facts[p].items():
                if (initial_node.state >> fid) & 1:
                    self.pallet_at[p] = loc
                    break

        # 4. Determine goal place for every crate
        for c in self.crates:
            curr = c
            visited = {curr}
            while curr in self.goal_on:
                curr = self.goal_on[curr]
                if curr in self.pallet_at:
                    self.goal_place[c] = self.pallet_at[curr]
                    break
                if curr in visited: break
                visited.add(curr)

        self.preprocessing_time = time.time() - start_time
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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

    def _compute(self, state, task_network):
        # 1. Map current truck and hoist locations
        truck_at = {}
        for t in self.trucks:
            for p, fid in self.at_facts[t].items():
                if (state >> fid) & 1: truck_at[t] = p; break
        
        hoist_at = {}
        for h in self.hoists:
            for p, fid in self.at_facts[h].items():
                if (state >> fid) & 1: hoist_at[h] = p; break

        # 2. Identify stable crates (already on goal and goal is supported by pallet)
        stable = set()
        while True:
            added = False
            for c in self.crates:
                if c in stable or c not in self.goal_on: continue
                gs = self.goal_on[c]
                if gs in self.on_facts[c]:
                    fid = self.on_facts[c][gs]
                    if (state >> fid) & 1:
                        if gs in self.pallets or gs in stable:
                            stable.add(c)
                            added = True
            if not added: break

        # 3. Sum up admissible costs per crate
        h = 0
        for c in self.crates:
            if c not in self.goal_place or c in stable:
                continue
            
            tp = self.goal_place[c]
            cp = None # Current place of crate
            is_in_truck = False
            is_lifting = False
            
            # Check direct at
            for p, fid in self.at_facts[c].items():
                if (state >> fid) & 1: cp = p; break
            
            if cp is None:
                # Check trucks
                for t, fid in self.in_facts[c].items():
                    if (state >> fid) & 1:
                        cp = truck_at.get(t)
                        is_in_truck = True
                        break
            
            if cp is None:
                # Check hoists
                for h_name, fid in self.lifting_facts[c].items():
                    if (state >> fid) & 1:
                        cp = hoist_at.get(h_name)
                        is_lifting = True
                        break
            
            # Apply admissible cost logic
            if is_lifting:
                h += 1 if cp == tp else 3
            elif is_in_truck:
                h += 2 # Unload + Drop
            else: # at a place
                h += 2 # Lift + Drop
                if cp != tp: h += 2 # Load + Unload

        # Tie-breaker: task network complexity
        h += 0.001 * len(task_network)
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
