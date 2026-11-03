from Pytrich.Heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic5(Heuristic):
    def __init__(self, model, depots_gemini3pro_heuristic_5=None):
        super().__init__(model)
        self.fact_data = {}
        self.goal_on = {}

        # Precompute fact parsing mapping from global_id to (predicate, args)
        # This optimization avoids repeated string parsing during heuristic evaluation
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_data[fact.global_id] = (pred, args)

        # Parse Goal State from bitmask efficiently
        # self.model.goals is a bitwise integer representing the set of goal facts
        goal_mask = self.model.goals
        temp_mask = goal_mask
        while temp_mask:
            # Extract the lowest set bit
            bit = temp_mask & -temp_mask
            gid = bit.bit_length() - 1
            temp_mask ^= bit
            
            if gid in self.fact_data:
                pred, args = self.fact_data[gid]
                if pred == 'on' and len(args) == 2:
                    # Map crate -> target_surface
                    self.goal_on[args[0]] = args[1]

    def _parse_fact_name(self, name: str):
        """Parses fact strings like '+on[c1,p1]' or '-at[t1,l1]'"""
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

    def h(self, node, task_network, state):
        """
        Calculates the heuristic value based on: 
        1. Unstable goal blocks (blocks not on their final target or whose base is unstable).
        2. Cost to clear blocks above the moving block and the target surface.
        3. Transportation cost if the block and target are at different locations.
        """
        current_on = {}
        current_at = {}
        current_in = {}
        
        # Efficiently decode current state from bitmask
        s = state
        while s:
            t = s & -s
            gid = t.bit_length() - 1
            s ^= t
            
            if gid in self.fact_data:
                pred, args = self.fact_data[gid]
                if pred == 'on':
                    current_on[args[0]] = args[1]
                elif pred == 'at':
                    current_at[args[0]] = args[1]
                elif pred == 'in':
                    current_in[args[0]] = args[1]

        # Build inverted index for upward traversal (surface -> crate on top)
        # In Depots, a surface holds at most one crate directly on it.
        on_inverse = {v: k for k, v in current_on.items()}

        # Memoization for recursive location lookup
        memo_loc = {}
        def get_loc(obj):
            if obj in memo_loc: return memo_loc[obj]
            
            res = None
            if obj in current_at:
                res = current_at[obj]
            elif obj in current_in:
                truck = current_in[obj]
                # If inside truck, location is truck's location
                res = get_loc(truck)
            elif obj in current_on:
                surf = current_on[obj]
                # If on surface, location is surface's location
                res = get_loc(surf)
            
            memo_loc[obj] = res
            return res

        # Memoization for stability check
        # A block is stable if it is on its goal target, and that target is also stable.
        memo_stable = {}
        def is_stable(c):
            if c in memo_stable: return memo_stable[c]
            
            # If not currently on anything (e.g., being lifted or in truck), it's not stable on a surface
            if c not in current_on:
                memo_stable[c] = False
                return False
                
            s = current_on[c]
            
            # Check if current position matches goal requirement
            target = self.goal_on.get(c)
            if target is not None:
                if s != target:
                    memo_stable[c] = False
                    return False
            
            # Check if the base 's' is stable
            # If 's' is not a crate involved in goals (e.g. a pallet), it is considered stable base
            if s not in self.goal_on:
                memo_stable[c] = True
                return True
                
            # Recursive check for the stack below
            res = is_stable(s)
            memo_stable[c] = res
            return res

        # Calculate number of blocks stacked above an object
        def count_above(obj):
            cnt = 0
            curr = obj
            # Limit depth to prevent infinite loops in case of malformed state cycles
            for _ in range(20):
                if curr not in on_inverse:
                    break
                cnt += 1
                curr = on_inverse[curr]
            return cnt

        h_val = 0
        
        # Iterate over all goal requirements
        for c, target in self.goal_on.items():
            # If the crate is already in a stable position, no cost
            if is_stable(c):
                continue
            
            # Base cost for moving the crate
            h_val += 1
            
            # Cost to clear the crate (remove blocks on top)
            h_val += count_above(c)
            
            # Cost to clear the target surface (remove blocks on top)
            # If target is currently supporting something else, that must be moved
            if target != c:
                h_val += count_above(target)
                
            # Transportation cost estimation
            loc_c = get_loc(c)
            loc_t = get_loc(target)
            
            # If locations are known and different, add penalty for truck transport
            # (Load + Drive + Unload overhead)
            if loc_c and loc_t and loc_c != loc_t:
                h_val += 2
                
        return h_val