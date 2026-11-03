from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic7(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_4):
        super().__init__(blocksworld_gemini3pro_heuristic_4)
        self.model = blocksworld_gemini3pro_heuristic_4
        self.fact_info = {}
        self.goal_on = {}
        self.goal_ontable = set()
        self._preprocess()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _preprocess(self):
        # 1. Pre-parse all facts in the model to avoid parsing strings during heuristic evaluation
        # We map global_id -> (predicate_name, args_list)
        for i, fact in enumerate(self.model.facts):
            # fact.global_id should be equal to i in the facts list based on Pytrich structure,
            # but we use enumerate index to be safe for array access.
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_info[i] = (pred, args)

        # 2. Parse Goal State
        # The goal is stored as a bitwise integer in self.model.goals
        # We extract set bits to identify goal facts
        goal_state = self.model.goals
        # Handle case if goals is a set of ints or just an int (Context says bitwise integer)
        if isinstance(goal_state, set):
            # Fallback if provided as set of indices (some variations do this)
            goal_indices = list(goal_state)
        else:
            # Standard bitwise int
            goal_indices = [i for i, c in enumerate(bin(goal_state)[2:][::-1]) if c == '1']

        for idx in goal_indices:
            if idx in self.fact_info:
                pred, args = self.fact_info[idx]
                if pred == 'on' and len(args) == 2:
                    self.goal_on[args[0]] = args[1]
                elif pred == 'ontable' and len(args) == 1:
                    self.goal_ontable.add(args[0])

    def h(self, node):
        # 1. Parse Current State from Bitmask
        # We need to reconstruct the configuration of blocks: 'on', 'ontable', 'holding'
        state_int = node.state
        
        current_on = {}
        current_ontable = set()
        holding = None

        # Fast bit iteration using string manipulation
        # This is generally faster in Python than shifting large integers 1000s of times
        bin_str = bin(state_int)[2:][::-1]
        active_indices = [i for i, bit in enumerate(bin_str) if bit == '1']

        all_blocks = set()
        
        for idx in active_indices:
            info = self.fact_info.get(idx)
            if not info: continue
            pred, args = info
            
            if pred == 'on' and len(args) == 2:
                top, bottom = args
                current_on[top] = bottom
                all_blocks.add(top)
                all_blocks.add(bottom)
            elif pred == 'ontable' and len(args) == 1:
                blk = args[0]
                current_ontable.add(blk)
                all_blocks.add(blk)
            elif pred == 'holding' and len(args) == 1:
                holding = args[0]
                all_blocks.add(holding)
        
        # Ensure all goal blocks are considered even if currently hidden (unlikely in valid states)
        all_blocks.update(self.goal_on.keys())
        all_blocks.update(self.goal_ontable)

        # 2. Calculate Blocks Out of Place (Dependency Chain)
        # A block is 'stable' if it is in its goal position AND the block below it is stable.
        
        # Memoization for stability check to avoid re-traversing chains
        memo_stable = {}

        def is_stable(b):
            if b in memo_stable:
                return memo_stable[b]
            
            # If currently held, it is definitely not stable
            if b == holding:
                memo_stable[b] = False
                return False

            # Determine goal requirement for b
            goal_sup = self.goal_on.get(b)
            goal_is_table = b in self.goal_ontable
            
            # If b has no specific goal position (irrelevant block),
            # it is stable if it sits on a stable base.
            # This assumes irrelevant blocks don't need to move unless they block something.
            # However, for the heuristic count, we usually treat irrelevant blocks as stable 
            # if they are grounded, but we penalize them if they are on top of something unstable below.
            if goal_sup is None and not goal_is_table:
                if b in current_on:
                    res = is_stable(current_on[b])
                    memo_stable[b] = res
                    return res
                elif b in current_ontable:
                    memo_stable[b] = True
                    return True
                else:
                    # Floating or unknown (should not happen in valid state)
                    memo_stable[b] = False
                    return False

            # Block has a goal
            if b in current_on:
                curr_sup = current_on[b]
                if curr_sup == goal_sup:
                    # Correct immediate support, check recursively
                    res = is_stable(curr_sup)
                    memo_stable[b] = res
                    return res
                else:
                    # Wrong support
                    memo_stable[b] = False
                    return False
            
            elif b in current_ontable:
                if goal_is_table:
                    memo_stable[b] = True
                    return True
                else:
                    # Should be on something, is on table
                    memo_stable[b] = False
                    return False
            
            # Block position not found (maybe held or parsing error)
            memo_stable[b] = False
            return False

        # 3. Compute Heuristic Cost
        # Cost = number of blocks that are not stable.
        # This implicitly counts moves: if a block is unstable, it must be moved.
        # If a block is on top of an unstable block, it is also unstable (must be unstacked).
        
        unstable_count = 0
        for b in all_blocks:
            if not is_stable(b):
                unstable_count += 1
        
        # Penalty for holding: getting rid of held block costs at least 1 action
        # and is usually a necessary step to stabilize things.
        if holding:
            unstable_count += 1

        return unstable_count
