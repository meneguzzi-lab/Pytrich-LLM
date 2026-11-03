import sys
from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic2(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_2, model):
        super().__init__(depots_gemini3pro_heuristic_2, model)
        self.model = model
        self.fact_map = {}
        self.pred_map = {}
        
        # Pre-parse facts efficiently in init to avoid repeated string ops
        for f in self.model.facts:
            name = f.name
            # Clean name format +pred[args] -> pred, [args]
            if name.startswith("+") or name.startswith("-"):
                name = name[1:]
            bracket = name.find('[')
            if bracket != -1:
                pred = name[:bracket]
                args = name[bracket+1:-1].split(',')
            else:
                pred = name
                args = []
            
            self.fact_map[f.global_id] = (pred, args)
            if pred not in self.pred_map:
                self.pred_map[pred] = []
            self.pred_map[pred].append(f.global_id)

    def h(self, node):
        state = node.state
        tasks = node.tasks
        
        # 1. Decode State from bitmask
        # We build maps to query the state efficiently: O(1) lookups
        on_map = {}      # crate -> surface
        above_map = {}   # surface -> crate (what is directly on top of surface)
        at_map = {}      # obj -> place
        in_map = {}      # crate -> truck
        lifting_map = {} # hoist -> crate
        is_lifted = {}   # crate -> hoist
        
        # Helper to check bit
        def is_true(gid):
            return (state >> gid) & 1

        # Only iterate over predicates relevant to heuristic logic
        relevant_preds = ['at', 'on', 'in', 'lifting']
        
        for pred in relevant_preds:
            if pred not in self.pred_map: continue
            for gid in self.pred_map[pred]:
                if is_true(gid):
                    _, args = self.fact_map[gid]
                    if pred == 'at':
                        at_map[args[0]] = args[1]
                    elif pred == 'on':
                        # on crate surface
                        c, s = args[0], args[1]
                        on_map[c] = s
                        above_map[s] = c
                    elif pred == 'in':
                        in_map[args[0]] = args[1]
                    elif pred == 'lifting':
                        h, c = args[0], args[1]
                        lifting_map[h] = c
                        is_lifted[c] = h

        # Helper to resolve location recursively
        def get_loc(obj):
            # Direct location (trucks, hoists, pallets, free crates)
            if obj in at_map: return at_map[obj]
            # Inside truck
            if obj in in_map:
                t = in_map[obj]
                return at_map.get(t)
            # Being lifted
            if obj in is_lifted:
                h = is_lifted[obj]
                return at_map.get(h)
            # Stacked on something
            if obj in on_map:
                return get_loc(on_map[obj])
            return None

        # 2. Pre-process tasks
        # Identify which crates are scheduled to move. 
        # This allows us to discount 'clearing' costs for items that are moving anyway.
        will_move = set()
        parsed_tasks = []
        
        for task in tasks:
            # Check if task is a primitive Operator (has cost attribute)
            if hasattr(task, 'cost'): 
                parsed_tasks.append(('primitive', task.cost, []))
                continue
            
            # Abstract Task parsing
            # Name format typically: "do_put_on crate1 pallet1"
            parts = task.name.split()
            t_name = parts[0]
            t_args = parts[1:]
            parsed_tasks.append((t_name, 0, t_args))
            
            if t_name == 'do_put_on':
                c, s = t_args[0], t_args[1]
                # If c is not already on s, it must move
                if on_map.get(c) != s:
                    will_move.add(c)

        # 3. Calculate Heuristic
        h_val = 0
        
        # Helper to calculate cost to clear an object (unstack items above it)
        def get_clear_cost(obj):
            cost = 0
            curr = obj
            # Traverse up the stack using above_map
            while curr in above_map:
                top = above_map[curr]
                if top in will_move:
                    # Optimization: 'top' is moving away due to another task, 
                    # so it (and anything on it) effectively stops blocking 'curr' for free
                    break
                cost += 1
                curr = top
            return cost

        for t_name, cost, args in parsed_tasks:
            if t_name == 'primitive':
                h_val += cost
                continue
            
            if t_name == 'do_put_on':
                c, s = args[0], args[1]
                
                # If already achieved, skip
                if on_map.get(c) == s:
                    continue
                
                # Get current locations
                loc_c = get_loc(c)
                loc_s = get_loc(s)
                
                # Fallback if state logic is incomplete
                if not loc_c or not loc_s:
                    h_val += 1
                    continue
                
                # Base manipulation cost logic
                if loc_c == loc_s:
                    # Same location: unstack/lift + drop
                    if c in in_map: 
                        h_val += 2 # Unload + Drop
                    elif c in is_lifted:
                        h_val += 1 # Drop
                    else:
                        h_val += 2 # Lift + Drop
                else:
                    # Different location: lift + load + drive + unload + drop
                    if c in in_map:
                        h_val += 3 # Drive + Unload + Drop
                    elif c in is_lifted:
                        h_val += 4 # Load + Drive + Unload + Drop
                    else:
                        h_val += 5 # Lift + Load + Drive + Unload + Drop
                
                # Add clearance costs
                # Cost to clear 'c' (must be clear to move)
                if c not in in_map and c not in is_lifted:
                    h_val += get_clear_cost(c)
                
                # Cost to clear 's' (must be clear to receive c)
                h_val += get_clear_cost(s)
                
            elif t_name == 'do_clear':
                s = args[0]
                h_val += get_clear_cost(s)
                
            elif t_name == 'do_get_truck':
                t, p = args[0], args[1]
                if at_map.get(t) != p:
                    h_val += 1 # Drive cost
            
            else:
                # Default for other abstract tasks (e.g., do_load, do_lift)
                # Assume at least 1 primitive action
                h_val += 1

        return h_val