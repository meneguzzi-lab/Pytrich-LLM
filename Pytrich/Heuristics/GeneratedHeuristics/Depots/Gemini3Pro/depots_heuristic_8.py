from Pytrich.Heuristics.heuristic import Heuristic
import sys

class DepotsGemini3ProHeuristic8(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_8, model):
        super().__init__(model)
        self.model = model
        self.fact_ids = {} 
        self.id_to_name = {}
        
        # Storage for parsed structural facts: (arg1_id, arg2_id, global_fact_idx)
        self.at_facts = [] 
        self.on_facts = [] 
        self.in_facts = [] 
        self.lifting_facts = [] 
        
        # Goal mappings: crate_id -> surface_id
        self.goals = {}
        
        # Static location cache (for pallets/rigid objects)
        self.static_locations = {}
        
        self._preprocess_facts()
        self._extract_goals()
        self._extract_static_locations()

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

    def _get_obj_id(self, name):
        if name not in self.fact_ids:
            uid = len(self.fact_ids)
            self.fact_ids[name] = uid
            self.id_to_name[uid] = name
        return self.fact_ids[name]

    def _preprocess_facts(self):
        # Single pass to parse all facts and organize by predicate type
        for i, fact in enumerate(self.model.facts):
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            arg_ids = [self._get_obj_id(a) for a in args]
            
            if pred == 'at':
                self.at_facts.append((arg_ids[0], arg_ids[1], i))
            elif pred == 'on':
                self.on_facts.append((arg_ids[0], arg_ids[1], i))
            elif pred == 'in':
                self.in_facts.append((arg_ids[0], arg_ids[1], i))
            elif pred == 'lifting':
                self.lifting_facts.append((arg_ids[0], arg_ids[1], i))

    def _extract_goals(self):
        # Goals are stored as a bitmask in model.goals
        # We iterate our identified 'on' facts to see which are required
        goals_mask = self.model.goals
        for crate_id, surf_id, idx in self.on_facts:
            if (goals_mask >> idx) & 1:
                self.goals[crate_id] = surf_id

    def _extract_static_locations(self):
        # Extract initial locations for pallets (roots) from initial state
        # Assuming pallets don't move, their 'at' facts are constant
        init_state = self.model.initial_state
        for obj_id, place_id, idx in self.at_facts:
            if (init_state >> idx) & 1:
                self.static_locations[obj_id] = place_id

    def __call__(self, node):
        state = node.state
        
        # 1. Reconstruct current state graph
        # parent_of[child] = (parent, type)
        # children_of[parent] = child (since Depots is stack-based, mostly 1 child on top)
        parent_of = {}
        children_of = {}
        at_loc = self.static_locations.copy() # Start with static, update with dynamic
        
        # Process dynamic 'at' (trucks, hoists, maybe moved items)
        for obj_id, place_id, idx in self.at_facts:
            if (state >> idx) & 1:
                at_loc[obj_id] = place_id
        
        # Process 'on'
        for top, bot, idx in self.on_facts:
            if (state >> idx) & 1:
                parent_of[top] = (bot, 'surface')
                children_of[bot] = top
        
        # Process 'in'
        for crate, truck, idx in self.in_facts:
            if (state >> idx) & 1:
                parent_of[crate] = (truck, 'truck')
                # Trucks contain things, but 'on' relation is specific to surfaces.
                # 'in' is containment. We don't map children_of for truck for obstruction purposes 
                # because things in trucks don't block each other in Depots domain logic usually.
                
        # Process 'lifting'
        for hoist, crate, idx in self.lifting_facts:
            if (state >> idx) & 1:
                parent_of[crate] = (hoist, 'hoist')
        
        # 2. Helper: Get current location (Place)
        loc_cache = {}
        def get_place(obj_id):
            if obj_id in loc_cache: return loc_cache[obj_id]
            
            if obj_id in at_loc:
                res = at_loc[obj_id]
                loc_cache[obj_id] = res
                return res
                
            if obj_id in parent_of:
                parent, ptype = parent_of[obj_id]
                res = get_place(parent)
                loc_cache[obj_id] = res
                return res
                
            loc_cache[obj_id] = None
            return None

        # 3. Determine Well-Placed Status
        # status: 0=unknown, 1=well-placed, 2=misplaced
        wp_status = {}
        
        def is_well_placed(c_id):
            if c_id in wp_status: 
                return wp_status[c_id] == 1
            
            # If c_id is not in goals, it is well-placed (neutral) unless it blocks something.
            # We handle blocking separately. Here we check strictly goal compliance.
            if c_id not in self.goals:
                wp_status[c_id] = 1
                return True
            
            target_s = self.goals[c_id]
            
            # Must be on the target surface
            if c_id not in parent_of:
                wp_status[c_id] = 2
                return False
                
            parent, ptype = parent_of[c_id]
            if ptype != 'surface' or parent != target_s:
                wp_status[c_id] = 2
                return False
                
            # Recursive check: is the target surface well-placed?
            # If target is a crate, check it. If pallet, it's a root (no parent in parent_of usually).
            if parent not in parent_of: 
                # Likely a pallet/root
                wp_status[c_id] = 1
                return True
            
            if is_well_placed(parent):
                wp_status[c_id] = 1
                return True
            else:
                wp_status[c_id] = 2
                return False

        # 4. Identify Moving Blocks
        moving_blocks = set()
        
        # A: Blocks that are misplaced
        all_crates = set(self.goals.keys())
        # Add other crates found in state (obstructions)
        for c in parent_of.keys():
            all_crates.add(c)
            
        for c in all_crates:
            if not is_well_placed(c):
                moving_blocks.add(c)
                
        # B: Blocks that are obstructing destinations
        for c, target_s in self.goals.items():
            # If target_s has something on it that is NOT c, that thing must move
            if target_s in children_of:
                blocker = children_of[target_s]
                if blocker != c:
                    # Blocker must move
                    moving_blocks.add(blocker)
        
        # C: Propagate movement upwards (if X moves, everything on X moves)
        # Use a queue for closure
        queue = list(moving_blocks)
        idx = 0
        while idx < len(queue):
            curr = queue[idx]
            idx += 1
            if curr in children_of:
                child = children_of[curr]
                if child not in moving_blocks:
                    moving_blocks.add(child)
                    queue.append(child)
                    
        # 5. Calculate Cost
        total_cost = 0
        for c in moving_blocks:
            # Determine current state
            state_type = 'on'
            if c in parent_of:
                _, state_type = parent_of[c]
            
            # Phase 1: Get to holding at destination
            transport_needed = False
            
            loc_c = get_place(c)
            # Destination place?
            if c in self.goals:
                target_s = self.goals[c]
                loc_dest = get_place(target_s)
            else:
                # If just moving out of way, local move is fine
                loc_dest = loc_c
            
            if loc_c is not None and loc_dest is not None and loc_c != loc_dest:
                transport_needed = True
            
            cost_to_hold_at_dest = 0
            
            if state_type == 'hoist':
                # Already holding. 
                if transport_needed:
                    # Load(1) + Drive(1) + Unload(1)
                    cost_to_hold_at_dest = 3
                else:
                    # Already at dest holding
                    cost_to_hold_at_dest = 0
            
            elif state_type == 'truck':
                # In truck
                if transport_needed:
                    # Drive(1) + Unload(1)
                    cost_to_hold_at_dest = 2
                else:
                    # Unload(1)
                    cost_to_hold_at_dest = 1
            
            else: # On surface or unknown
                # Need to lift first
                # Lift(1)
                if transport_needed:
                    # Lift(1) + Load(1) + Drive(1) + Unload(1)
                    cost_to_hold_at_dest = 4
                else:
                    # Lift(1)
                    cost_to_hold_at_dest = 1
            
            # Phase 2: Drop on target (or table)
            # Drop(1)
            drop_cost = 1
            
            total_cost += cost_to_hold_at_dest + drop_cost
            
        return total_cost
