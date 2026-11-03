import sys

# Pytrich/Heuristics/Heuristic.py
class Heuristic:
    def __init__(self, heuristic_name, model):
        self.heuristic_name = heuristic_name
        self.model = model

    def h(self, state, task_network):
        return 0

class DepotsGemini3ProHeuristic12(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_12, model):
        super().__init__(depots_gemini3pro_heuristic_12, model)
        self.facts = model.facts
        self.goal_state_mask = model.goals
        
        # Object Sets
        self.crates = set()
        self.pallets = set()
        self.trucks = set()
        self.hoists = set()
        
        # Fact Lookups (indices)
        self.f_on = []          # (id, c_name, s_name)
        self.f_in = []          # (id, c_name, t_name)
        self.f_lifting = []     # (id, h_name, c_name)
        self.f_at_truck = []    # (id, t_name, p_name)
        self.f_clear = {}       # s_name -> id
        
        # Static Locations
        self.static_pallet_loc = {} # p_name -> place_name
        self.static_hoist_loc = {}  # h_name -> place_name
        
        # Goal Configuration
        self.goal_map = {} # c_name -> s_name
        
        self._initialize_domain()
        
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

    def _initialize_domain(self):
        """
        Parses facts to categorize objects and identify static vs dynamic elements.
        Also parses the goal state mask into a structured map.
        """
        raw_at = []
        
        # 1. Categorize predicates and collect args
        for f in self.facts:
            fid = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == 'on':
                # on(crate, surface)
                c, s = args
                self.crates.add(c)
                self.f_on.append((fid, c, s))
            elif pred == 'in':
                # in(crate, truck)
                c, t = args
                self.crates.add(c)
                self.trucks.add(t)
                self.f_in.append((fid, c, t))
            elif pred == 'lifting':
                # lifting(hoist, crate)
                h, c = args
                self.hoists.add(h)
                self.crates.add(c)
                self.f_lifting.append((fid, h, c))
            elif pred == 'at':
                # at(obj, place)
                obj, p = args
                raw_at.append((fid, obj, p))
            elif pred == 'clear':
                # clear(surface)
                s = args[0]
                self.f_clear[s] = fid
        
        # 2. Identify Pallets
        # Pallets are objects in 'at' that are not Trucks, Hoists, or Crates.
        potential_pallets = set()
        for fid, obj, p in raw_at:
            if obj not in self.trucks and obj not in self.hoists and obj not in self.crates:
                potential_pallets.add(obj)
        self.pallets = potential_pallets
        
        # 3. Store Location Info
        for fid, obj, p in raw_at:
            if obj in self.trucks:
                self.f_at_truck.append((fid, obj, p))
            elif obj in self.hoists:
                self.static_hoist_loc[obj] = p
            elif obj in self.pallets:
                self.static_pallet_loc[obj] = p
            
        # 4. Parse Goals from Bitmask
        # We iterate relevant 'on' facts to see which are set in the goal mask.
        for fid, c, s in self.f_on:
            if (self.goal_state_mask >> fid) & 1:
                self.goal_map[c] = s

    def h(self, state, task_network):
        """
        Calculates the heuristic cost based on:
        1. Goal achievement (using stability check)
        2. Transport costs (Depot distance)
        3. Clearing costs (Obstructions)
        """
        # --- 1. Decode Current State Configuration ---
        
        # Truck Locations
        truck_loc = {}
        for fid, t, p in self.f_at_truck:
            if (state >> fid) & 1:
                truck_loc[t] = p
        
        # Object Relations
        parent = {}     # crate -> surface
        container = {}  # crate -> truck/hoist
        
        for fid, c, s in self.f_on:
            if (state >> fid) & 1:
                parent[c] = s
                
        for fid, c, t in self.f_in:
            if (state >> fid) & 1:
                container[c] = t
                
        for fid, h, c in self.f_lifting:
            if (state >> fid) & 1:
                container[c] = h

        # --- 2. Location Helper with Memoization ---
        loc_cache = {}
        # Seed static locations
        for p_obj, p_loc in self.static_pallet_loc.items():
            loc_cache[p_obj] = p_loc
            
        def get_loc(obj):
            if obj in loc_cache: return loc_cache[obj]
            
            # If truck
            if obj in truck_loc: return truck_loc[obj]
            
            # If hoist
            if obj in self.static_hoist_loc: return self.static_hoist_loc[obj]
            
            # If crate held by container
            if obj in container:
                holder = container[obj]
                l = get_loc(holder)
                if l:
                    loc_cache[obj] = l
                return l
            
            # If crate on surface
            if obj in parent:
                s = parent[obj]
                l = get_loc(s)
                if l:
                    loc_cache[obj] = l
                return l
            
            return None

        # --- 3. Stability Check ---
        # Verifies if a crate is part of a bottom-up correct stack matching the goals.
        stable_cache = {p: True for p in self.pallets}
        
        def check_stability(c):
            if c in stable_cache: return stable_cache[c]
            
            if c not in parent:
                stable_cache[c] = False
                return False
                
            s = parent[c]
            
            # Condition: Must be on the goal surface
            if c in self.goal_map:
                if self.goal_map[c] != s:
                    stable_cache[c] = False
                    return False
            
            # Condition: Surface beneath must be stable
            res = check_stability(s)
            stable_cache[c] = res
            return res

        # --- 4. Cost Calculation ---
        cost = 0
        
        for c, s_target in self.goal_map.items():
            # If correctly placed on stable base, cost is 0
            if c in parent and parent[c] == s_target:
                if check_stability(s_target):
                    continue
            
            # Basic cost for unsatisfied goal (min operations to move)
            cost += 1
            
            # Distance Estimation
            loc_c = get_loc(c)
            loc_s = get_loc(s_target)
            
            if loc_c and loc_s:
                if loc_c != loc_s:
                    # Needs truck transport: Load + Drive + Unload
                    cost += 3
                else:
                    # Same location: Lift + Drop
                    cost += 1
            else:
                # Penalize missing information
                cost += 5
                
            # Obstruction Penalty: Crate 'c' must be clear to move
            if c in self.f_clear:
                fid = self.f_clear[c]
                if not ((state >> fid) & 1):
                    cost += 1 # Penalty for being buried
            
            # Obstruction Penalty: Target surface must be clear to receive
            if s_target in self.f_clear:
                fid = self.f_clear[s_target]
                if not ((state >> fid) & 1):
                    cost += 1 # Penalty for target being buried
                    
        return cost
