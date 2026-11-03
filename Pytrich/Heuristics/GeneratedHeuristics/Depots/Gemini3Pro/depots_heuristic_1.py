from pytrich.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic1(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_1=None):
        super().__init__(model)
        self.obj_to_id = {}
        self.id_to_obj = []
        self.goal_relations = []
        self.relevant_facts_map = {} 
        self.at_facts_map = {}

        def get_obj_id(name):
            if name not in self.obj_to_id:
                self.obj_to_id[name] = len(self.id_to_obj)
                self.id_to_obj.append(name)
            return self.obj_to_id[name]

        # Parse all facts to build lookup tables
        for i in range(self.model.ifacts_end + 1):
            fact = self.model.facts[i]
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            arg_ids = [get_obj_id(a) for a in args]
            fid = fact.global_id
            
            if pred in ['on', 'in', 'lifting']:
                # Relationships: arg0 is 'child', arg1 is 'parent' (surface, truck, or hoist)
                child = arg_ids[0]
                parent = arg_ids[1]
                
                # Map child -> potential parents (for location finding)
                if child not in self.relevant_facts_map:
                    self.relevant_facts_map[child] = []
                self.relevant_facts_map[child].append((fid, 'parent', parent))
                
                # Map parent -> potential children (for burial calculation)
                # Only 'on' implies burial/stacking
                if pred == 'on':
                    if parent not in self.relevant_facts_map:
                        self.relevant_facts_map[parent] = []
                    self.relevant_facts_map[parent].append((fid, 'child', child))

            elif pred == 'at':
                obj = arg_ids[0]
                loc = arg_ids[1]
                if obj not in self.at_facts_map:
                    self.at_facts_map[obj] = []
                self.at_facts_map[obj].append((fid, loc))

        # Identify goal conditions
        goal_mask = self.model.goals
        for i in range(goal_mask.bit_length()):
            if (goal_mask >> i) & 1:
                if i <= self.model.ifacts_end:
                    fact = self.model.facts[i]
                    pred, args = self._parse_fact_name(fact.name)
                    if pred == 'on':
                        self.goal_relations.append((get_obj_id(args[0]), get_obj_id(args[1]), fact.global_id))

    def _parse_fact_name(self, name):
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

    def compute(self, node):
        state = node.state
        
        # Quick check if goals are fully satisfied
        if (self.model.goals & state) == self.model.goals:
            return 0
        
        # Caches for current state traversal to avoid re-scanning facts
        parent_cache = {}
        child_cache = {}
        loc_cache = {}
        
        def get_parent(obj_id):
            if obj_id in parent_cache: return parent_cache[obj_id]
            if obj_id in self.relevant_facts_map:
                # Check which of the possible parent facts is true
                for fid, rtype, target in self.relevant_facts_map[obj_id]:
                    if rtype == 'parent' and ((state >> fid) & 1):
                        parent_cache[obj_id] = target
                        return target
            parent_cache[obj_id] = None
            return None

        def get_child(obj_id):
            if obj_id in child_cache: return child_cache[obj_id]
            if obj_id in self.relevant_facts_map:
                # Check which crate is currently on this object
                for fid, rtype, target in self.relevant_facts_map[obj_id]:
                    if rtype == 'child' and ((state >> fid) & 1):
                        child_cache[obj_id] = target
                        return target
            child_cache[obj_id] = None
            return None

        def get_loc(obj_id):
            if obj_id in loc_cache: return loc_cache[obj_id]
            
            # 1. Check direct 'at' facts (Trucks, Hoists, Pallets)
            if obj_id in self.at_facts_map:
                for fid, loc in self.at_facts_map[obj_id]:
                    if (state >> fid) & 1:
                        loc_cache[obj_id] = loc
                        return loc
            
            # 2. Recursive check via parent (Crates in Trucks or on Surfaces)
            p = get_parent(obj_id)
            if p is not None:
                l = get_loc(p)
                loc_cache[obj_id] = l
                return l
            
            loc_cache[obj_id] = None
            return None

        def count_above(obj_id):
            c = get_child(obj_id)
            if c is None: return 0
            return 1 + count_above(c)

        h = 0
        for c, s, gid in self.goal_relations:
            # Skip if goal already met
            if (state >> gid) & 1:
                continue
            
            # Base cost: Must drop the crate eventually
            h += 1
            
            loc_c = get_loc(c)
            loc_s = get_loc(s)
            
            # Location mismatch cost
            if loc_c is None or loc_s is None:
                h += 10 # Heavy penalty for lost/buried information
            elif loc_c != loc_s:
                h += 3 # Requires Load + Drive + Unload
            else:
                h += 1 # Intra-location move (Lift + Drop)
            
            # Burial Costs (Stacking constraints)
            # 1. Cost to unbury C (move everything above it)
            h += count_above(c)
            
            # 2. Cost to clear S (move everything above it)
            occupant = get_child(s)
            if occupant is not None and occupant != c:
                h += 1 + count_above(occupant)
                
        return h