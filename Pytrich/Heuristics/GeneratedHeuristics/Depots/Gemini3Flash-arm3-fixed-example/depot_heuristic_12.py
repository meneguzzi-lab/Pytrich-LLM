"""
Domain-dependent heuristic for the Depots domain.

1. Bottleneck Analysis:
   The Depots domain is a fusion of Logistics and Blocksworld. The primary bottleneck is the serial execution 
   of stack clearing and transportation. A hoist can only handle one crate at a time, and a crate must be 
   cleared of everything above it before it can be moved. Similarly, the destination surface must be clear.

2. Admissible Lower Bounds:
   - Term 1: Transportation Cost. For each crate 'c' not at its goal surface 's', if 'c' is at a different location 
     than 's', it requires at least 4 actions (Lift, Load, Unload, Drop). If they are at the same location, 
     it requires at least 2 actions (Lift, Drop).
   - Term 2: Clearing Cost. For every crate 'd' currently blocking a crate 'c' that needs to move, or blocking 
     the target surface 's' of a goal, 'd' must be lifted and moved elsewhere, requiring at least 2 actions.
   - Term 3: Goal Stability. A crate is only 'correct' if it's on its target and its target is also correct 
     (pallets are always correct). This recursive definition captures necessary re-stacking.

These terms are combined additively (where independent) or via max to maintain admissibility while 
providing strong guidance.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic12(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_12=None, name="DepotsGemini3FlashHeuristic12"):
        super().__init__(name=name)
        self.goal_on = {}  # crate -> target_surface
        self.pallets = set()
        self.crates = set()
        self.trucks = set()
        self.hoists = set()
        self.places = set()
        
        self.on_info = []      # list of (crate, surface, global_id)
        self.at_info = []      # list of (locatable, place, global_id)
        self.in_info = []      # list of (crate, truck, global_id)
        self.lifting_info = [] # list of (hoist, crate, global_id)
        
        self.pallet_locs = {}  # pallet -> place (static)
        self.hoist_locs = {}   # hoist -> place (static)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse goals
        if hasattr(model, 'goals') and model.goals:
            goal_fact_names = model.state_explicit_repr(model.goals)
            for goal_name in goal_fact_names:
                pred, args = self._parse_fact_name(goal_name)
                if pred == "on":
                    self.goal_on[args[0]] = args[1]

        # 2. Extract static info and map facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at":
                # Hoists and Pallets don't move in this domain usually
                if args[0].startswith("pallet"):
                    self.pallets.add(args[0])
                    self.pallet_locs[args[0]] = args[1]
                elif args[0].startswith("hoist"):
                    self.hoists.add(args[0])
                    self.hoist_locs[args[0]] = args[1]
                elif args[0].startswith("truck"):
                    self.trucks.add(args[0])
                elif args[0].startswith("crate"):
                    self.crates.add(args[0])
                self.places.add(args[1])
                self.at_info.append((args[0], args[1], fact.global_id))
            elif pred == "on":
                self.on_info.append((args[0], args[1], fact.global_id))
                self.crates.add(args[0])
                if args[1].startswith("crate"):
                    self.crates.add(args[1])
            elif pred == "in":
                self.in_info.append((args[0], args[1], fact.global_id))
            elif pred == "lifting":
                self.lifting_info.append((args[0], args[1], fact.global_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
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

    def _compute(self, state, task_network):
        # 1. Build current state lookups
        curr_on = {}     # surface -> crate on it
        curr_on_rev = {} # crate -> what it is on
        curr_at = {}     # locatable -> place
        
        # Static locations
        curr_at.update(self.pallet_locs)
        curr_at.update(self.hoist_locs)
        
        for obj, loc, fid in self.at_info:
            if (state >> fid) & 1: curr_at[obj] = loc
        for c, s, fid in self.on_info:
            if (state >> fid) & 1:
                curr_on[s] = c
                curr_on_rev[c] = s
        
        # Trucks are already in curr_at. Now handle crates in trucks or hoists
        for c, t, fid in self.in_info:
            if (state >> fid) & 1: curr_at[c] = curr_at.get(t)
        for h, c, fid in self.lifting_info:
            if (state >> fid) & 1: curr_at[c] = curr_at.get(h)

        # 2. Identify stable vs unstable crates
        # A crate is stable if it's on its goal surface and that surface is stable
        stable = {}
        def check_stable(c):
            if c in stable: return stable[c]
            target_s = self.goal_on.get(c)
            if not target_s: # Crate has no goal? Consider it stable if on its initial surface
                stable[c] = True
                return True
            if curr_on_rev.get(c) != target_s:
                stable[c] = False
                return False
            if target_s in self.pallets:
                stable[c] = True
                return True
            is_s_stable = check_stable(target_s)
            stable[c] = is_s_stable
            return is_s_stable

        unstable_crates = [c for c in self.crates if not check_stable(c)]

        # 3. Calculate Heuristic Value
        h = 0
        processed_clearing = set()

        for c in unstable_crates:
            target_s = self.goal_on.get(c)
            if not target_s: continue
            
            # Base move cost
            p1 = curr_at.get(c)
            p2 = curr_at.get(target_s)
            
            if p1 and p2:
                if p1 == p2: h += 2 # Lift, Drop
                else: h += 4        # Lift, Load, Unload, Drop
            else: h += 2 # Default if location unknown (lifting/in truck)

            # Clearing costs (2 actions per crate above c and target_s)
            # We track processed clearing to avoid double counting crates in the same stack
            cursor = curr_on.get(c)
            while cursor:
                if cursor not in processed_clearing:
                    h += 2
                    processed_clearing.add(cursor)
                cursor = curr_on.get(cursor)
            
            cursor = curr_on.get(target_s)
            while cursor:
                if cursor not in processed_clearing:
                    h += 2
                    processed_clearing.add(cursor)
                cursor = curr_on.get(cursor)

        # 4. Tie-breaker for task network progress
        h += len(task_network) * 0.1

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
