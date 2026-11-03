from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic10(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_10, model):
        super().__init__(depots_gemini3pro_heuristic_10, model)
        self.model = model
        self.fact_map = {}
        self.at_facts = []
        self.on_facts = []
        self.in_facts = []
        
        # Pre-process facts to avoid repeated string parsing during heuristic evaluation
        for fact in self.model.facts:
            name = fact.name
            # Parse grounded fact names like +at[crate1,depot0] or +on[crate1,pallet0]
            pred, args = self._parse_fact_name(name)
            if pred:
                self.fact_map[fact.global_id] = (pred, args)
                if pred == 'at':
                    self.at_facts.append((fact.global_id, args))
                elif pred == 'on':
                    self.on_facts.append((fact.global_id, args))
                elif pred == 'in':
                    self.in_facts.append((fact.global_id, args))

    def _parse_fact_name(self, name: str):
        """Parses fact name string into predicate and arguments."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        
        bracket_start = name.find('[')
        if bracket_start != -1 and name.endswith(']'):
            pred = name[:bracket_start]
            args = name[bracket_start+1:-1].split(',')
            return pred, args
        return None, []

    def h(self, model, state, task_network):
        """
        Domain-dependent heuristic for Depots.
        Estimates cost by analyzing current state (crate positions, stacking) and 
        remaining tasks in the network.
        """
        # 1. Reconstruct minimal State Representation from bitmask
        # Map: object -> location (place)
        obj_loc = {}
        # Map: top_object -> bottom_object (what is it on?)
        on_map = {}
        # Map: bottom_object -> top_object (what is above it?)
        rev_on_map = {}
        # Map: object -> truck (is it inside a truck?)
        in_truck = {}
        
        # Efficiently iterate only over relevant pre-filtered facts
        for fid, args in self.at_facts:
            if (state >> fid) & 1:
                obj_loc[args[0]] = args[1]
                
        for fid, args in self.on_facts:
            if (state >> fid) & 1:
                on_map[args[0]] = args[1]
                rev_on_map[args[1]] = args[0]
                
        for fid, args in self.in_facts:
            if (state >> fid) & 1:
                in_truck[args[0]] = args[1]

        # Helper: Memoized location resolver
        # Resolves location for stacked items or items in trucks recursively
        memo_loc = {}
        def get_loc(x):
            if x in memo_loc: return memo_loc[x]
            if x in obj_loc:
                memo_loc[x] = obj_loc[x]
                return obj_loc[x]
            if x in in_truck:
                t = in_truck[x]
                l = get_loc(t)
                if l: 
                    memo_loc[x] = l
                    return l
            if x in on_map:
                under = on_map[x]
                l = get_loc(under)
                if l: 
                    memo_loc[x] = l
                    return l
            return None

        # Helper: Count items stacked above x to estimate 'clear' cost
        def items_above(x):
            cnt = 0
            curr = x
            while curr in rev_on_map:
                curr = rev_on_map[curr]
                cnt += 1
            return cnt

        total_cost = 0

        # 2. Analyze Task Network
        for task in task_network:
            # Heuristic Additivity: Sum estimated costs of all remaining tasks
            
            # Handle Primitive Operators directly
            if hasattr(task, 'del_effects'): 
                # It's an operator (primitive action). Assign unit cost.
                total_cost += 1
                continue

            # Handle Abstract Tasks
            name = task.name
            # Normalize task name string for parsing
            clean_name = name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = clean_name.split()
            if not parts: continue
            
            t_type = parts[0]
            
            if "do_put_on" in t_type and len(parts) >= 3:
                # Task: do_put_on crate surface
                c, s = parts[1], parts[2]
                
                # If already achieved, cost is 0
                if on_map.get(c) == s:
                    continue
                    
                # Cost components:
                # 1. Clear the crate (unstack above)
                c_clear_cost = items_above(c)
                
                # 2. Clear the surface (unstack above)
                s_clear_cost = items_above(s)
                
                # 3. Distance / Movement cost
                loc_c = get_loc(c)
                loc_s = get_loc(s)
                
                dist_cost = 0
                if loc_c and loc_s and loc_c != loc_s:
                    # Needs truck: Load + Drive + Unload
                    dist_cost = 3 
                
                # 4. Base operation cost (Lift + Drop)
                base_cost = 1
                
                total_cost += (c_clear_cost + s_clear_cost + dist_cost + base_cost)
            
            elif "do_clear" in t_type and len(parts) >= 2:
                # Task: do_clear surface
                s = parts[1]
                # Cost is proportional to number of items on top
                total_cost += items_above(s)
                
            elif "do_get_truck" in t_type and len(parts) >= 3:
                # Task: do_get_truck truck place
                t, p = parts[1], parts[2]
                t_loc = get_loc(t)
                if t_loc != p:
                    total_cost += 1 # Drive cost
            
            elif "do_load_truck" in t_type and len(parts) >= 2:
                # Task: do_load_truck crate ...
                c = parts[1]
                # Needs to be clear + lifted + loaded
                total_cost += items_above(c) + 2
                
            elif "do_unload_truck" in t_type and len(parts) >= 3:
                # Task: do_unload_truck crate surface ...
                s = parts[2]
                # Surface needs to be clear + unload + drop
                total_cost += items_above(s) + 2
                
            elif "do_lift_crate" in t_type and len(parts) >= 2:
                # Task: do_lift_crate crate ...
                c = parts[1]
                total_cost += items_above(c) + 1
            
            else:
                # Default fallback for other tasks or malformed names
                total_cost += 1
                
        return total_cost
