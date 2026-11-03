import sys

class DepotsGemini3ProHeuristic4(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_4, model):
        super().__init__(depots_gemini3pro_heuristic_4, model)
        self.model = model
        
        # --- Preprocessing ---
        self.on_facts = []        # list of (crate, surface, global_id)
        self.at_facts = []        # list of (obj, place, global_id)
        self.in_facts = []        # list of (crate, truck, global_id)
        self.lifting_facts = []   # list of (hoist, crate, global_id)
        self.clear_facts = []     # list of (surface, global_id)
        
        self.crates = set()
        self.pallets = set()
        self.trucks = set()
        self.places = set()
        
        self.goal_on = {}         # crate -> target_surface
        
        # 1. Parse all facts to categorize them and identify objects
        for fact in self.model.facts:
            name = fact.name
            # Fact format is typically +predicate[arg1,arg2] or -predicate...
            # We assume Pytrich facts are grounded strings like '+on[crate1,pallet1]'
            
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
                
            if pred == 'on':
                if len(args) == 2:
                    self.on_facts.append((args[0], args[1], fact.global_id))
                    self.crates.add(args[0])
                    # args[1] could be crate or pallet
            elif pred == 'at':
                if len(args) == 2:
                    self.at_facts.append((args[0], args[1], fact.global_id))
                    self.places.add(args[1])
            elif pred == 'in':
                if len(args) == 2:
                    self.in_facts.append((args[0], args[1], fact.global_id))
                    self.crates.add(args[0])
                    self.trucks.add(args[1])
            elif pred == 'lifting':
                if len(args) == 2:
                    # lifting ?h ?c
                    self.lifting_facts.append((args[0], args[1], fact.global_id))
                    self.crates.add(args[1])
            elif pred == 'clear':
                if len(args) == 1:
                    self.clear_facts.append((args[0], fact.global_id))

        # 2. Identify Pallets (Surfaces that are not Crates)
        # We iterate known surfaces from 'on' facts and subtract known crates
        all_surfaces = set()
        for c, s, fid in self.on_facts:
            all_surfaces.add(s)
            
        # Any surface that is not a crate is treated as a pallet (static location)
        for s in all_surfaces:
            if s not in self.crates:
                self.pallets.add(s)
                
        # 3. Static Locations for Pallets
        # Pallets don't move. We find their location from 'at' facts in the initial state.
        self.pallet_loc = {}
        # We need to check initial state bits
        init_state = self.model.initial_state
        # But simpler: scan all 'at' facts. If arg0 is a pallet, that's its fixed location.
        # This assumes pallets don't move (standard Depots domain).
        for obj, place, fid in self.at_facts:
            if obj in self.pallets:
                self.pallet_loc[obj] = place

        # 4. Parse Goals
        # model.goals is a bitwise integer
        goal_mask = self.model.goals
        if isinstance(goal_mask, set):
             # Compatibility if goals are passed as set of global_ids (older pytrich versions sometimes)
             # But based on context, it is bitwise or we check explicit representation
             pass
        
        # We iterate all facts to check if they are in the goal mask
        for fact in self.model.facts:
            if (goal_mask >> fact.global_id) & 1:
                pred, args = self._parse_fact_name(fact.name)
                if pred == 'on' and len(args) == 2:
                    self.goal_on[args[0]] = args[1]
        
        # 5. Precompute Final Locations for Goal Crates
        # Trace goal chain: Crate -> Crate -> ... -> Pallet -> Place
        self.final_pos = {}
        self._compute_final_positions()

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

    def _compute_final_positions(self):
        # Fix point iteration or recursion to propagate locations up the goal stacks
        # Pallets have fixed locations.
        # Crate C on S -> FinalPos(C) = FinalPos(S)
        
        # Seed with pallets
        for p, loc in self.pallet_loc.items():
            self.final_pos[p] = loc
            
        # Propagate
        changed = True
        while changed:
            changed = False
            for c, s in self.goal_on.items():
                if c not in self.final_pos and s in self.final_pos:
                    self.final_pos[c] = self.final_pos[s]
                    changed = True

    def h(self, node):
        state = node.state
        
        # --- 1. Decode State ---
        # We need efficient lookups: 
        # map: crate -> current_surface (on)
        # map: obj -> current_place (at)
        # map: truck -> current_place
        
        current_on = {}      # top -> bottom
        current_loc = {}     # obj -> place (for those explicitly at a place)
        in_truck = {}        # crate -> truck
        lifting = {}         # crate -> hoist
        
        # Check 'on' facts
        for c, s, fid in self.on_facts:
            if (state >> fid) & 1:
                current_on[c] = s
        
        # Check 'at' facts
        for obj, p, fid in self.at_facts:
            if (state >> fid) & 1:
                current_loc[obj] = p
        
        # Check 'in' facts
        for c, t, fid in self.in_facts:
            if (state >> fid) & 1:
                in_truck[c] = t
                
        # Check 'lifting' facts
        for h, c, fid in self.lifting_facts:
            if (state >> fid) & 1:
                lifting[c] = h

        # Resolve locations for all crates
        # Priority: explicit 'at' > in truck > lifting
        # Note: In depots, 'at' is usually removed when 'in' or 'lifting'.
        
        final_obj_loc = {} # obj -> place_name
        
        # 1. Objects explicitly at locations
        for obj, p in current_loc.items():
            final_obj_loc[obj] = p
            
        # 2. Pallets (static fallback)
        for p, loc in self.pallet_loc.items():
            if p not in final_obj_loc:
                final_obj_loc[p] = loc
                
        # 3. Trucks (should be in current_loc)
        # 4. Crates in Trucks
        for c, t in in_truck.items():
            if t in final_obj_loc:
                final_obj_loc[c] = final_obj_loc[t]
                
        # 5. Crates being lifted
        for c, h in lifting.items():
            # Hoist location is needed. Hoists are at places.
            if h in final_obj_loc:
                final_obj_loc[c] = final_obj_loc[h]

        # 6. Crates on Surfaces
        # If C is on S, and S is at P, then C is at P.
        # Need to traverse 'on' chains if 'at' is not explicit for stacked items.
        # In Depots, typically only bottom crate has 'at' or all have 'at'? 
        # To be safe, propagate locations up the stacks.
        # Build adjacency for propagation: bottom -> list of tops
        holding = {} # surface -> top_crate
        for top, bottom in current_on.items():
            holding[bottom] = top
            
        # Iteratively resolve locations
        # This handles stacks A on B on C... where C has a location.
        # Since max stack depth is finite, simple loop works.
        # Objects to resolve: crates in current_on
        pending = set(current_on.keys())
        iterations = 0
        while pending and iterations < 20: # Limit loop just in case
            resolved_this_round = set()
            for c in pending:
                if c in final_obj_loc:
                    resolved_this_round.add(c)
                    continue
                s = current_on[c]
                if s in final_obj_loc:
                    final_obj_loc[c] = final_obj_loc[s]
                    resolved_this_round.add(c)
            
            if not resolved_this_round:
                break # Cannot resolve remaining (partial state or detached)
            pending -= resolved_this_round
            iterations += 1

        # --- 2. Calculate Heuristic ---
        # Check correctness of placement recursively
        # Memoization for is_correct
        memo_correct = {}

        def is_correct(obj):
            if obj in memo_correct:
                return memo_correct[obj]
            
            # Pallets are always correct
            if obj in self.pallets:
                memo_correct[obj] = True
                return True
                
            # Check if obj is a crate with a goal
            if obj not in self.goal_on:
                # If obj is not in goal, it is effectively "incorrect" if it sits on top of something important
                # but for the purpose of recursion, we treat it as neutral or False.
                # Usually treating as False (needs moving if it's in the way) is safer.
                memo_correct[obj] = False
                return False
            
            target = self.goal_on[obj]
            
            # Must be on the target
            if current_on.get(obj) != target:
                memo_correct[obj] = False
                return False
            
            # And the target must be correct
            res = is_correct(target)
            memo_correct[obj] = res
            return res

        h_val = 0
        
        # Iterate over all crates to assess penalties
        for c in self.crates:
            # If correctly placed, no cost
            if is_correct(c):
                continue
                
            # Base cost for being misplaced (implies at least one move/action)
            h_val += 1
            
            # Logistics Penalty: If crate is at wrong place (City/Depot)
            # Only relevant if we know where it SHOULD be.
            target_pos = self.final_pos.get(c)
            curr_pos = final_obj_loc.get(c)
            
            if target_pos and curr_pos and target_pos != curr_pos:
                # Cost to move between locations: Load (1) + Drive (1) + Unload (1) = 3
                # We already added 1 base cost, so add 2 more.
                h_val += 2
            
            # Obstruction Penalty (Buried)
            # If this crate is incorrect, but it is COVERED by another crate,
            # that covering crate effectively adds to the cost.
            # We iterate ALL crates. The covering crate will also be visited.
            # If the covering crate is correct, it wouldn't be here (impossible if below is incorrect).
            # If the covering crate is incorrect, it adds +1 to h_val.
            # So simply summing 1 for every misplaced crate inherently accounts for stack depth.
            # E.g., Stack A(top)-B-C(bot). Goal: C elsewhere. 
            # A is incorrect (needs move). B incorrect. C incorrect.
            # Total 3. Actions: Move A, Move B, Move C. Matches.
            
            # What if C is in a truck? 
            # in_truck[c] -> True. current_on[c] -> None.
            # is_correct(c) -> False (unless goal is in truck, which is invalid).
            # Cost += 1. Plus logistics penalty if truck is at wrong place.
            # Matches.
            
            # What if C is being lifted?
            # lifting[c] -> True.
            # Cost += 1. Plus logistics if hoist at wrong place.
            # Matches.
            
        return h_val
