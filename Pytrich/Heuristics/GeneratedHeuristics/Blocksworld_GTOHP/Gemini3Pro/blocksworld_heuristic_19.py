from Pytrich.Heuristics.heuristic import Heuristic
import sys

class BlocksworldGemini3ProHeuristic19(Heuristic):
    def __init__(self, model, blocksworld_gemini3pro_heuristic_16=None):
        super().__init__(model)
        self.model = model
        self.blocksworld_gemini3pro_heuristic_16 = blocksworld_gemini3pro_heuristic_16
        
        # Caches for efficient state lookup
        self.all_blocks = []
        self.block_pos_masks = {}  # b -> (mask, {bit_index: target_block_or_keyword})
        self.goal_map = {}         # b -> target_block_or_keyword
        
        self._initialize()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
        
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")] if args_str else []
            return predicate, args
        return name, []

    def _initialize(self):
        """
        Pre-computes bitmasks and mappings to allow O(1) checking of block positions
        and goals during heuristic evaluation.
        """
        # Temporary lists to store facts by type
        on_facts = []      # (global_id, x, y)
        ontable_facts = [] # (global_id, x)
        holding_facts = [] # (global_id, x)
        blocks_set = set()

        # 1. Parse all facts in the model to identify blocks and predicates
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            gid = fact.global_id

            if pred == "on":
                if len(args) == 2:
                    x, y = args
                    on_facts.append((gid, x, y))
                    blocks_set.add(x)
                    blocks_set.add(y)
            elif pred == "ontable":
                if len(args) == 1:
                    x = args[0]
                    ontable_facts.append((gid, x))
                    blocks_set.add(x)
            elif pred == "holding":
                if len(args) == 1:
                    x = args[0]
                    holding_facts.append((gid, x))
                    blocks_set.add(x)
            elif pred == "clear":
                if len(args) == 1:
                    blocks_set.add(args[0])

        self.all_blocks = list(blocks_set)

        # 2. Build Position Masks for each block
        # This creates a bitmask of all possible position facts (on, ontable, holding) for block 'b'
        # and a mapping from bit_index to the semantic position.
        for b in self.all_blocks:
            mask = 0
            mapping = {}

            # Map 'on(b, y)' facts
            for gid, x, y in on_facts:
                if x == b:
                    mask |= (1 << gid)
                    mapping[gid] = y
            
            # Map 'ontable(b)' facts
            for gid, x in ontable_facts:
                if x == b:
                    mask |= (1 << gid)
                    mapping[gid] = "TABLE"

            # Map 'holding(b)' facts
            for gid, x in holding_facts:
                if x == b:
                    mask |= (1 << gid)
                    mapping[gid] = "HOLDING"
            
            self.block_pos_masks[b] = (mask, mapping)

        # 3. Parse Goal State from model.goals bitmask
        # We iterate our identified facts and check if they are in the goal set
        goals_int = self.model.goals
        if goals_int != 0:
            for gid, x, y in on_facts:
                if (goals_int >> gid) & 1:
                    self.goal_map[x] = y
            
            for gid, x in ontable_facts:
                if (goals_int >> gid) & 1:
                    self.goal_map[x] = "TABLE"
            
            for gid, x in holding_facts:
                if (goals_int >> gid) & 1:
                    self.goal_map[x] = "HOLDING"

    def __call__(self, node):
        """
        Calculates the heuristic cost for a given search node.
        The heuristic is based on the 'blocks out of place' logic adapted for HTN.
        It counts how many blocks are not in their goal position (recursively checking stability of support).
        """
        state = node.state
        
        # Cache for recursion within this single evaluation
        # Maps block_name -> True (correctly placed) / False (misplaced)
        status_cache = {}
        misplaced_count = 0

        # Check each block's status
        for b in self.all_blocks:
            if not self._is_correct(b, state, status_cache, set()):
                misplaced_count += 1
        
        # If no blocks are misplaced but there are still tasks, use task count to guide decomposition
        if misplaced_count == 0 and len(node.tasks) > 0:
             return len(node.tasks)

        return misplaced_count

    def _is_correct(self, b, state, cache, path):
        """
        Recursive check to see if block 'b' is in a 'correct' state.
        A block is correct if:
        1. It is in its goal position (or has no goal).
        2. The block/surface it is resting on is also correct (stable).
        """
        if b in cache:
            return cache[b]
        
        # Cycle detection for partial states (should typically not happen in valid BW states)
        if b in path:
            return False
        path.add(b)

        # 1. Determine current position of 'b' efficiently using bitmasks
        mask, mapping = self.block_pos_masks.get(b, (0, {}))
        active_bits = state & mask
        
        current_pos = None
        if active_bits:
            # Find the index of the set bit (assuming logical consistency of one position)
            # (n & -n) isolates the lowest set bit
            lsb = active_bits & -active_bits
            gid = lsb.bit_length() - 1
            current_pos = mapping.get(gid)

        # If position is unknown/undefined in state, assume misplaced
        if current_pos is None:
            path.remove(b)
            cache[b] = False
            return False

        # Special case: Holding
        if current_pos == "HOLDING":
            is_goal_holding = (self.goal_map.get(b) == "HOLDING")
            path.remove(b)
            cache[b] = is_goal_holding
            return is_goal_holding

        # 2. Check Goal Compliance
        target = self.goal_map.get(b)
        
        is_stable = False
        
        if target is not None:
            # Block has a specific goal
            if current_pos != target:
                is_stable = False
            else:
                # Block is on target, check if target is stable
                if current_pos == "TABLE":
                    is_stable = True
                else:
                    # Recursively check the block below
                    is_stable = self._is_correct(current_pos, state, cache, path)
        else:
            # Block has NO specific goal
            # It is considered correct (stable) if it is resting on a stable structure.
            # If it's on a block that needs to move, it blocks it, so it's 'misplaced'.
            if current_pos == "TABLE":
                is_stable = True
            else:
                is_stable = self._is_correct(current_pos, state, cache, path)

        path.remove(b)
        cache[b] = is_stable
        return is_stable
