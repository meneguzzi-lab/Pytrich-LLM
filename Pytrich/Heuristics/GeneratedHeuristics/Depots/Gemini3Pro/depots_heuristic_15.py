from pytrich.heuristics.heuristic import Heuristic

class DepotsGemini3ProHeuristic15(Heuristic):
    def __init__(self, depots_gemini3pro_heuristic_15):
        super().__init__(depots_gemini3pro_heuristic_15)
        self.model = depots_gemini3pro_heuristic_15

        # Pre-process facts to separate them by predicate type for efficient lookup
        self.on_facts = []       # List of (global_id, top_obj, bottom_obj)
        self.at_facts = []       # List of (global_id, obj, location)
        self.in_facts = []       # List of (global_id, crate, truck)
        self.lifting_facts = []  # List of (global_id, hoist, crate)
        self.pallets = set()

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on':
                self.on_facts.append((fact.global_id, args[0], args[1]))
                if 'pallet' in args[1]:
                    self.pallets.add(args[1])
            elif pred == 'at':
                self.at_facts.append((fact.global_id, args[0], args[1]))
            elif pred == 'in':
                self.in_facts.append((fact.global_id, args[0], args[1]))
            elif pred == 'lifting':
                self.lifting_facts.append((fact.global_id, args[0], args[1]))

        # Parse goal state
        # Goals are stored as a bitmask. We extract the required 'on' relationships.
        self.goal_on = {}
        goal_mask = self.model.goals
        for gid, top, bottom in self.on_facts:
            if (goal_mask >> gid) & 1:
                self.goal_on[top] = bottom

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

    def eval(self, node):
        state = node.state
        
        # 1. Reconstruct Partial State Graph from Bitmask
        # We only reconstruct what is necessary for the heuristic
        current_on = {}      # top -> bottom
        current_above = {}   # bottom -> top
        current_at = {}      # obj -> location
        
        # Check ON facts
        for gid, top, bottom in self.on_facts:
            if (state >> gid) & 1:
                current_on[top] = bottom
                current_above[bottom] = top

        # Check AT facts
        for gid, obj, loc in self.at_facts:
            if (state >> gid) & 1:
                current_at[obj] = loc

        # Check IN facts (crate inside truck implies location)
        for gid, crate, truck in self.in_facts:
            if (state >> gid) & 1:
                if truck in current_at:
                    current_at[crate] = current_at[truck]

        # Check LIFTING facts (crate held by hoist implies location)
        for gid, hoist, crate in self.lifting_facts:
            if (state >> gid) & 1:
                if hoist in current_at:
                    current_at[crate] = current_at[hoist]

        # 2. Compute Heuristic Cost
        # We sum costs for all misplaced goal blocks.
        # Cost includes: 1 (move action) + depth (unstacking above) + target depth (clearing target) + distance (truck)
        
        h_val = 0
        
        # Memoization for correctness check
        # A block is 'correct' if it is on its goal target, and that target is also correct (recursive).
        is_correct_memo = {}
        
        def is_placed_correctly(obj):
            if obj in is_correct_memo:
                return is_correct_memo[obj]
            
            # Base cases
            if obj in self.pallets:
                is_correct_memo[obj] = True
                return True
            
            target = self.goal_on.get(obj)
            
            # If obj is not in goal spec, we treat it as 'neutral'. 
            # It is 'correct' in the sense that it doesn't need to move for its own sake.
            if target is None:
                is_correct_memo[obj] = True
                return True
            
            # Check if currently on target
            actual_base = current_on.get(obj)
            if actual_base == target:
                # Recursively check if the base is correct
                val = is_placed_correctly(target)
                is_correct_memo[obj] = val
                return val
            
            is_correct_memo[obj] = False
            return False

        # Iterate through all goals to accumulate cost
        for top, target in self.goal_on.items():
            if is_placed_correctly(top):
                continue
                
            # If not correct, we incur costs.
            h_val += 1  # Base cost: at least one move/lift/drop sequence needed
            
            # Cost 1: Clear the block 'top'
            # Iterate upwards from 'top' to count how many blocks are on top of it
            curr = top
            while curr in current_above:
                h_val += 1
                curr = current_above[curr]
                # Cycle protection (though states shouldn't have cycles)
                if curr == top: break
            
            # Cost 2: Clear the target 'target'
            # Iterate upwards from 'target' to count obstructions
            if target not in self.pallets:
                curr_t = target
                while curr_t in current_above:
                    above = current_above[curr_t]
                    # If the block above is 'top' itself (e.g., on target but wrong stack below target?),
                    # we stop to avoid double counting, though is_placed_correctly usually handles this.
                    if above == top: 
                        break
                    h_val += 1
                    curr_t = above
                    if curr_t == target: break

            # Cost 3: Transportation distance
            # If we know locations, add penalty for being in different places
            loc_top = current_at.get(top)
            loc_target = current_at.get(target)
            
            if loc_top is not None and loc_target is not None:
                if loc_top != loc_target:
                    h_val += 1  # Requires driving

        return h_val
