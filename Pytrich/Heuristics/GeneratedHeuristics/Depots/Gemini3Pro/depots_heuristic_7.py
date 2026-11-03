from Pytrich.Heuristics.heuristic import Heuristic
import sys

class DepotsGemini3ProHeuristic7(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_7=None):
        super().__init__(model)
        self.model = model
        
        # Define constants for fact types
        self.FACT_AT = 1
        self.FACT_ON = 2
        self.FACT_IN = 3
        self.FACT_LIFTING = 4
        
        # 1. Parse objects and facts for fast lookup
        self.name_to_id = {}
        self.id_to_name = []
        
        # Table to map global_id -> (type, arg1_id, arg2_id)
        # Using a fixed size list is faster than dict for integer keys 0..N
        self.fact_data = [None] * len(self.model.facts)
        
        for fact in self.model.facts:
            # Fact name format: +predicate[arg1,arg2] or -predicate...
            name = fact.name.strip()
            if name.startswith('+'): name = name[1:]
            if name.startswith('-'): continue # Ignore negative facts for state reconstruction
            
            if '[' not in name or not name.endswith(']'):
                continue
                
            pred, args_str = name[:-1].split('[', 1)
            args = args_str.split(',') if args_str else []
            
            # Map args to internal IDs
            arg_ids = []
            for arg in args:
                arg = arg.strip()
                if arg not in self.name_to_id:
                    self.name_to_id[arg] = len(self.id_to_name)
                    self.id_to_name.append(arg)
                arg_ids.append(self.name_to_id[arg])
            
            type_code = 0
            # Identify key predicates for Depots domain
            if pred == 'at' and len(arg_ids) == 2: type_code = self.FACT_AT
            elif pred == 'on' and len(arg_ids) == 2: type_code = self.FACT_ON
            elif pred == 'in' and len(arg_ids) == 2: type_code = self.FACT_IN
            # lifting ?hoist ?crate
            elif pred == 'lifting' and len(arg_ids) == 2: type_code = self.FACT_LIFTING
            
            if type_code != 0:
                self.fact_data[fact.global_id] = (type_code, arg_ids[0], arg_ids[1])

        # 2. Identify Goals
        self.goal_relations = []
        goals_mask = self.model.goals
        while goals_mask:
            # Extract lowest bit
            t = goals_mask & -goals_mask
            gid = t.bit_length() - 1
            goals_mask ^= t
            
            if gid < len(self.fact_data):
                data = self.fact_data[gid]
                if data and data[0] == self.FACT_ON:
                    self.goal_relations.append((data[1], data[2]))

    def h(self, node):
        state = node.state
        
        # Reconstruct state graphs
        # parent: crate -> surface (ON)
        # children: surface -> list of crates (ON - inverse)
        # at_map: object -> place (AT)
        # in_map: crate -> truck (IN)
        # lifting_map: crate -> hoist (LIFTING)
        
        parent = {}
        children = {}
        at_map = {}
        in_map = {}
        lifting_map = {}
        
        # Efficient bit iteration to rebuild state representation
        curr = state
        while curr:
            t = curr & -curr
            gid = t.bit_length() - 1
            curr ^= t
            
            data = self.fact_data[gid]
            if data:
                ftype, a1, a2 = data
                if ftype == self.FACT_ON:
                    parent[a1] = a2
                    if a2 not in children: children[a2] = []
                    children[a2].append(a1)
                elif ftype == self.FACT_AT:
                    at_map[a1] = a2
                elif ftype == self.FACT_IN:
                    in_map[a1] = a2
                elif ftype == self.FACT_LIFTING:
                    # lifting ?hoist ?crate -> crate(a2) is held by hoist(a1)
                    lifting_map[a2] = a1

        # Cache for recursive location lookups
        loc_cache = {}

        def get_location(obj_id):
            if obj_id in loc_cache: return loc_cache[obj_id]
            
            loc = -1
            if obj_id in at_map:
                loc = at_map[obj_id]
            elif obj_id in parent:
                # If on something, location is same as parent
                loc = get_location(parent[obj_id])
            elif obj_id in in_map:
                # If in truck, location is truck's location
                truck = in_map[obj_id]
                loc = get_location(truck)
            elif obj_id in lifting_map:
                # If held by hoist, location is hoist's location
                hoist = lifting_map[obj_id]
                loc = get_location(hoist)
            
            loc_cache[obj_id] = loc
            return loc

        def count_stack_above(obj_id):
            if obj_id not in children: return 0
            count = 0
            # Traverse up the stack
            stack = list(children[obj_id])
            while stack:
                child = stack.pop()
                count += 1
                if child in children:
                    stack.extend(children[child])
            return count

        cost = 0
        
        for c, s in self.goal_relations:
            # Check if goal is satisfied
            if parent.get(c) == s:
                continue
            
            # Goal is NOT satisfied
            
            # 1. Cost to clear C (the item to be moved)
            # C needs to be clear to be lifted.
            # We count everything stacked on top of C.
            cost += count_stack_above(c)
            
            # If C is ON something (not held/in truck), we need to pick it up (Lift)
            if c in parent:
                cost += 1 
            elif c in in_map:
                # In truck, needs unload
                cost += 1
            # If lifting, no pick up cost (already held)

            # 2. Cost to clear S (the destination surface)
            # S needs to be clear to receive C.
            # If S is a crate, we count everything on top of it.
            cost += count_stack_above(s)
            
            # 3. Transportation / Move cost
            loc_c = get_location(c)
            loc_s = get_location(s)
            
            if loc_c != -1 and loc_s != -1:
                if loc_c != loc_s:
                    # Different location: Requires Load + Drive + Unload sequence
                    # Heuristic weight: 3
                    cost += 3
                else:
                    # Same location: Just Drop (if held) or Lift+Drop
                    cost += 1
            else:
                # Unknown location (partial state?), apply generic penalty
                cost += 2
                
        return cost
