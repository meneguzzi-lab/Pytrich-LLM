"""
Domain-dependent heuristic for the Depots domain.

Bottleneck Analysis:
The Depots domain is a fusion of Logistics and Blocksworld. The primary bottleneck is the serial nature of stacking and transporting crates.
1. Stacking: Hoists are limited resources (one per place), and crates must be cleared to move them or to access surfaces.
2. Logistics: Crates must be transported between depots/distributors via trucks, which can only be in one place at a time.
3. Sequentiality: To satisfy a goal on(c, s), s must be at the correct place, s must be clear, and c must be transported to s's location.

Admissible Lower Bounds:
1. Transport Cost: Every crate not at its goal location requires at least a Lift, Load, Unload, and Drop sequence (4 actions).
2. Placement Cost: Every crate at its goal location but on the wrong surface requires at least a Lift and Drop sequence (2 actions).
3. Clearance Cost: Any crate blocking a goal surface or blocking another crate that needs to move must be lifted and moved (at least 2 actions).

This heuristic implements a 'must_move' set logic that propagates requirements from the bottom of stacks upwards, ensuring all blocking and misplaced crates are counted.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotsGemini3FlashHeuristic11(Heuristic):
    def __init__(self, depot_gemini3flash_heuristic_11=None, name="DepotsGemini3FlashHeuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Preprocessed domain info
        self.crates = []
        self.pallets = []
        self.trucks = []
        self.hoists = []
        self.places = []
        
        self.goal_on = {}        # crate -> surface
        self.goal_place = {}     # crate -> place
        self.goal_clear = set()  # surface
        self.is_goal_surface = {}# surface -> crate (the crate that wants to be on it)
        
        self.pallet_to_place = {} # pallet -> place
        
        # Fact Maps
        self.on_fact_ids = {}      # (crate, surface) -> id
        self.at_fact_ids = {}      # (locatable, place) -> id
        self.in_fact_ids = {}      # (crate, truck) -> id
        self.lifting_fact_ids = {} # (hoist, crate) -> id
        
        self.on_facts_by_crate = {}
        self.on_facts_by_surface = {}
        self.at_facts_by_locatable = {}
        self.in_facts_by_crate = {}
        self.lifting_facts_by_crate = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # 1. Identify all objects and types
        all_locatables = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'at':
                all_locatables.add(args[0])
                self.places.append(args[1])
            elif pred == 'on':
                self.crates.append(args[0])
            elif pred == 'in':
                self.crates.append(args[0])
                self.trucks.append(args[1])
            elif pred == 'lifting':
                self.hoists.append(args[0])
                self.crates.append(args[1])
        
        self.crates = sorted(list(set(self.crates)))
        self.places = sorted(list(set(self.places)))
        self.trucks = sorted(list(set(self.trucks)))
        self.hoists = sorted(list(set(self.hoists)))
        self.pallets = sorted(list(all_locatables - set(self.crates) - set(self.trucks) - set(self.hoists)))
        
        # 2. Build Fact Maps
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                self.on_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.on_facts_by_crate: self.on_facts_by_crate[args[0]] = []
                self.on_facts_by_crate[args[0]].append((args[1], f.global_id))
                if args[1] not in self.on_facts_by_surface: self.on_facts_by_surface[args[1]] = []
                self.on_facts_by_surface[args[1]].append((args[0], f.global_id))
            elif pred == 'at':
                self.at_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.at_facts_by_locatable: self.at_facts_by_locatable[args[0]] = []
                self.at_facts_by_locatable[args[0]].append((args[1], f.global_id))
            elif pred == 'in':
                self.in_fact_ids[(args[0], args[1])] = f.global_id
                if args[0] not in self.in_facts_by_crate: self.in_facts_by_crate[args[0]] = []
                self.in_facts_by_crate[args[0]].append((args[1], f.global_id))
            elif pred == 'lifting':
                self.lifting_fact_ids[(args[0], args[1])] = f.global_id
                if args[1] not in self.lifting_facts_by_crate: self.lifting_facts_by_crate[args[1]] = []
                self.lifting_facts_by_crate[args[1]].append((args[0], f.global_id))

        # 3. Parse initial state for static pallet locations
        initial_state_facts = model.state_explicit_repr(initial_node.state)
        for f_name in initial_state_facts:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'at' and args[0] in self.pallets:
                self.pallet_to_place[args[0]] = args[1]

        # 4. Parse Goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == 'on':
                    self.goal_on[args[0]] = args[1]
                    self.is_goal_surface[args[1]] = args[0]
                elif pred == 'clear':
                    self.goal_clear.add(args[0])

        # 5. Precompute Goal Places for Crates
        for crate in self.crates:
            curr = crate
            visited = set()
            while curr in self.goal_on and curr not in visited:
                visited.add(curr)
                curr = self.goal_on[curr]
            if curr in self.pallet_to_place:
                self.goal_place[crate] = self.pallet_to_place[curr]

        self.preprocessing_time = time.time() - start_time
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
        current_on = {}        # crate -> surface
        current_on_top_of = {} # surface -> crate
        in_truck = {}          # crate -> truck
        being_lifted = {}      # crate -> hoist
        at_place = {}          # locatable -> place

        # Extract current state structure efficiently
        for c in self.crates:
            # Check where crate is
            for p in self.places:
                if (state >> self.at_fact_ids.get((c, p), 0)) & 1:
                    at_place[c] = p; break
            if c not in at_place:
                for t in self.trucks:
                    if (state >> self.in_fact_ids.get((c, t), 0)) & 1:
                        in_truck[c] = t; break
            if c not in at_place and c not in in_truck:
                for h in self.hoists:
                    if (state >> self.lifting_fact_ids.get((h, c), 0)) & 1:
                        being_lifted[c] = h; break
            
            # Check what crate is on
            if c in self.on_facts_by_crate:
                for s, fid in self.on_facts_by_crate[c]:
                    if (state >> fid) & 1:
                        current_on[c] = s
                        current_on_top_of[s] = c; break

        # Propagate Must-Move set
        must_move = set()
        # Crates in transit must be moved to destination
        for c in in_truck: must_move.add(c)
        for c in being_lifted: must_move.add(c)

        # Traverse stacks from bottom up at every place
        for p in self.places:
            p_pallets = [pal for pal in self.pallets if self.pallet_to_place.get(pal) == p]
            for pal in p_pallets:
                curr = pal
                found_misplaced = False
                while curr in current_on_top_of:
                    c = current_on_top_of[curr]
                    # Reason 1: This crate belongs elsewhere
                    if not found_misplaced:
                        if c in self.goal_on and self.goal_on[c] != curr:
                            found_misplaced = True
                        # Reason 2: This surface is needed by someone else
                        elif curr in self.is_goal_surface and self.is_goal_surface[curr] != c:
                            found_misplaced = True
                        # Reason 3: This surface must be clear
                        elif curr in self.goal_clear:
                            found_misplaced = True
                    
                    if found_misplaced:
                        must_move.add(c)
                    curr = c

        # Sum costs for must_move crates
        h = 0
        for c in must_move:
            if c in being_lifted: 
                h += 1
            elif c in in_truck:
                h += 2
            else:
                # Misplaced crate at a place
                c_place = at_place.get(c)
                g_place = self.goal_place.get(c)
                if g_place and c_place != g_place:
                    h += 4 # Lift, Load, Unload, Drop
                else:
                    h += 2 # Lift, Drop (within same place or blocking)
        
        # Tie-breaker for task progress
        h += 0.01 * len(task_network)
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
