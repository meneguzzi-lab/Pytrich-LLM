from typing import List, Set, Dict, Tuple, Optional
import sys
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGemini3ProHeuristic6(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_6=None):
        super().__init__(model)
        self.model = model
        
        # --- Preprocessing ---
        # Maps for fast O(1) state lookups
        self.supports = defaultdict(list)    # surface_name -> [(crate_name, global_id)]
        self.is_on = defaultdict(list)       # crate_name -> [(surface_name, global_id)]
        self.at_facts = defaultdict(list)    # obj_name -> [(place_name, global_id)]
        self.in_facts = defaultdict(list)    # crate_name -> [(truck_name, global_id)]
        self.lifting_facts = defaultdict(list) # crate_name -> [(hoist_name, global_id)]
        
        self.goal_triplets = [] # List of (crate, surface, global_id)

        self._analyze_domain()

    def _analyze_domain(self):
        # 1. Build lookup maps from static facts
        for fact in self.model.facts:
            name = fact.name
            fid = fact.global_id
            
            # Parse format: +predicate[arg1,arg2] or -predicate...
            if name.startswith('-'): continue
            clean_name = name[1:] if name.startswith('+') else name
            
            if '[' not in clean_name:
                continue
                
            pred, args_s = clean_name[:-1].split('[')
            args = args_s.split(',')
            
            if pred == 'on':
                # on[crate, surface]
                if len(args) == 2:
                    c, s = args
                    self.supports[s].append((c, fid))
                    self.is_on[c].append((s, fid))
            
            elif pred == 'at':
                # at[obj, place]
                if len(args) == 2:
                    o, p = args
                    self.at_facts[o].append((p, fid))
            
            elif pred == 'in':
                # in[crate, truck]
                if len(args) == 2:
                    c, t = args
                    self.in_facts[c].append((t, fid))
            
            elif pred == 'lifting':
                # lifting[hoist, crate]
                if len(args) == 2:
                    h, c = args
                    self.lifting_facts[c].append((h, fid))

        # 2. Decode Goals
        # model.goals is a bitwise integer mask
        goals = self.model.goals
        if isinstance(goals, int):
            # Iterate 0..len(facts) to find set bits
            # (Assuming global_id corresponds to bit index)
            for i in range(len(self.model.facts)):
                if (goals >> i) & 1:
                    self._add_goal(self.model.facts[i])
        else:
            # Fallback for set-based goals
            for i in goals:
                self._add_goal(self.model.facts[i])

    def _add_goal(self, fact):
        name = fact.name
        clean_name = name[1:] if name.startswith('+') else name
        if '[' not in clean_name: return
        pred, args_s = clean_name[:-1].split('[')
        
        # We only care about 'on' goals for Depots logic
        if pred == 'on':
            args = args_s.split(',')
            if len(args) == 2:
                self.goal_triplets.append((args[0], args[1], fact.global_id))

    def compute(self, state, task_network=None):
        """
        Estimate cost-to-go for Depots.
        Logic:
        1. For each unachieved goal on(C, S):
           - Cost to clear C (unstack items above it)
           - Cost to clear S (unstack items above it, if S is a crate)
           - Movement cost (1 if same place, >1 if different place)
        """
        h_val = 0
        
        # Per-state caches to handle recursion/shared stacks efficiently
        self._cache_loc = {}
        self._cache_clear = {}
        
        for c, s, fid in self.goal_triplets:
            # Check if goal is already satisfied in bitwise state
            if (state >> fid) & 1:
                continue
            
            # --- Cost 1: Clear the crate C ---
            cost_clear_c = self._get_clear_cost(c, state)
            
            # --- Cost 2: Clear the target surface S ---
            # In depots, Drop action requires (clear ?surface).
            cost_clear_s = self._get_clear_cost(s, state)
            
            # --- Cost 3: Transport C to S ---
            loc_c = self._get_location(c, state)
            loc_s = self._get_location(s, state)
            
            dist_cost = 0
            if loc_c is not None and loc_s is not None:
                if loc_c == loc_s:
                    # Same location: Lift + Drop = 2 actions roughly.
                    # We count '1' as base heuristic unit for 'move'.
                    dist_cost = 1
                else:
                    # Different location: Lift + Load + Drive + Unload + Drop
                    dist_cost = 3 
            else:
                # Penalty if location unknown (e.g. holding/in truck logic intermediate)
                dist_cost = 2
            
            h_val += cost_clear_c + cost_clear_s + dist_cost
            
        return h_val

    def _get_clear_cost(self, surface, state):
        """Recursive cost to clear everything above a surface."""
        if surface in self._cache_clear:
            return self._cache_clear[surface]
        
        # Find what is immediately ON this surface
        above_crate = None
        if surface in self.supports:
            for c, fid in self.supports[surface]:
                if (state >> fid) & 1:
                    above_crate = c
                    break
        
        cost = 0
        if above_crate:
            # Cost = 1 (to move 'above_crate') + cost to clear 'above_crate'
            cost = 1 + self._get_clear_cost(above_crate, state)
        
        self._cache_clear[surface] = cost
        return cost

    def _get_location(self, obj, state):
        """Find place where obj is located (recursive)."""
        if obj in self._cache_loc:
            return self._cache_loc[obj]
        
        # 1. Direct location (at obj place)
        if obj in self.at_facts:
            for p, fid in self.at_facts[obj]:
                if (state >> fid) & 1:
                    self._cache_loc[obj] = p
                    return p
        
        # 2. Indirect location via 'on' (crate on surface)
        if obj in self.is_on:
            for s, fid in self.is_on[obj]:
                if (state >> fid) & 1:
                    loc = self._get_location(s, state)
                    self._cache_loc[obj] = loc
                    return loc
                    
        # 3. Indirect location via 'in' (crate in truck)
        if obj in self.in_facts:
            for t, fid in self.in_facts[obj]:
                if (state >> fid) & 1:
                    loc = self._get_location(t, state)
                    self._cache_loc[obj] = loc
                    return loc

        # 4. Indirect location via 'lifting' (hoist lifting crate)
        if obj in self.lifting_facts:
            for h, fid in self.lifting_facts[obj]:
                if (state >> fid) & 1:
                    loc = self._get_location(h, state)
                    self._cache_loc[obj] = loc
                    return loc
                    
        self._cache_loc[obj] = None
        return None
