from typing import List, Dict, Set, Tuple, Optional
from Pytrich.Heuristics.heuristic import Heuristic

class BlocksworldGemini3ProHeuristic4(Heuristic):
    """
    A domain-dependent heuristic for Blocksworld that estimates the number of blocks
    that are not in their correct goal position relative to the table.

    Heuristic Strategy:
    1. Parse the Goal State to build a 'target tower' structure.
    2. For a given state, reconstruct the current tower structure from 'on' and 'ontable' facts.
    3. Perform a bottom-up traversal (starting from the Table) to identify 'fixed' blocks.
       - A block is fixed if it is on a fixed base AND it is in its goal position (or has no goal but isn't blocking one).
       - The Table is always fixed.
    4. Blocks that are not 'fixed' are considered out of place.
    5. H(n) = Count of non-fixed blocks.

    This heuristic is efficient (O(N) per state evaluation) and effectively captures the 
    constructive nature of Blocksworld (building towers from bottom up).
    """

    def __init__(self, blocksworld_gemini3pro_heuristic_1, model):
        super().__init__(blocksworld_gemini3pro_heuristic_1, model)
        self.model = model
        self.block_to_id: Dict[str, int] = {}
        self.id_to_block: List[str] = []
        self.table_id = -1

        # Map global_id -> (type_code, arg1_id, arg2_id)
        # Type codes: 1=on, 2=ontable
        self.fact_info = [None] * (len(model.facts) + 1)

        # 1. Identify Blocks and assign integer IDs for performance
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred in ["on", "ontable", "holding", "clear", "handempty"]:
                for arg in args:
                    if arg not in self.block_to_id:
                        self.block_to_id[arg] = len(self.id_to_block)
                        self.id_to_block.append(arg)

        # 2. Map grounded facts to efficient integer lookup table
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                # on(top, bottom)
                u = self.block_to_id.get(args[0])
                v = self.block_to_id.get(args[1])
                if u is not None and v is not None:
                    self.fact_info[fact.global_id] = (1, u, v)
            elif pred == "ontable":
                # ontable(block)
                u = self.block_to_id.get(args[0])
                if u is not None:
                    self.fact_info[fact.global_id] = (2, u, self.table_id)
        
        # 3. Parse Goal State into Structural Maps
        self.goal_on = {} # block_id -> base_id
        self.goal_base_needed_by = {} # base_id -> block_id (which block wants to be on this base)

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                u = self.block_to_id.get(args[0])
                v = self.block_to_id.get(args[1])
                if u is not None and v is not None:
                    self.goal_on[u] = v
                    self.goal_base_needed_by[v] = u
            elif pred == "ontable":
                u = self.block_to_id.get(args[0])
                if u is not None:
                    self.goal_on[u] = self.table_id
                    # Table is shared, so we don't mark it as uniquely needed in needed_by

    def _parse_fact_name(self, name: str):
        """Parses fact strings like +on[b1,b2] or -clear[b1]."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return name, []

    def __call__(self, node):
        state = node.state
        num_blocks = len(self.id_to_block)
        
        # current_on: map block_id -> base_id. Initialize to -2 (unknown/holding)
        current_on = [-2] * num_blocks
        
        # Efficiently iterate over set bits in the state integer
        # Only processing true facts (sparse iteration)
        temp_state = state
        while temp_state:
            # Get lowest set bit
            lowest_bit = temp_state & -temp_state
            idx = lowest_bit.bit_length() - 1
            temp_state ^= lowest_bit # Clear bit
            
            if idx < len(self.fact_info):
                info = self.fact_info[idx]
                if info:
                    ftype, u, v = info
                    if ftype == 1: # on(u, v)
                        current_on[u] = v
                    elif ftype == 2: # ontable(u)
                        current_on[u] = self.table_id

        # Build adjacency list for topological traversal (base -> [blocks on top])
        # In Blocksworld, a block has max 1 on top, but Table has many.
        on_top_of = {}
        for b in range(num_blocks):
            base = current_on[b]
            if base != -2:
                if base not in on_top_of:
                    on_top_of[base] = []
                on_top_of[base].append(b)

        # BFS/Traversal to find fixed blocks starting from Table
        fixed_count = 0
        queue = []
        
        # Add blocks currently on table to queue
        if self.table_id in on_top_of:
            queue.extend(on_top_of[self.table_id])
            
        # Traverse
        q_idx = 0
        while q_idx < len(queue):
            b = queue[q_idx]
            q_idx += 1
            
            base = current_on[b]
            
            # A block b is fixed if:
            # 1. Its base is fixed (Implicit because we traverse from fixed bases up)
            # 2. b matches its goal position (or has no specific goal)
            # 3. b is NOT blocking a goal (i.e., base is not needed by a different block)
            
            # Check Goal Alignment
            goal_base = self.goal_on.get(b)
            is_pos_correct = False
            
            if goal_base is None:
                # No specific goal for this block -> correct by default
                is_pos_correct = True
            elif goal_base == base:
                # Matches goal
                is_pos_correct = True
                
            is_blocking = False
            if is_pos_correct:
                # Check if this position blocks another block's goal
                # Only applies if base is a block (Table can hold everyone)
                if base != self.table_id:
                    needed_by = self.goal_base_needed_by.get(base)
                    if needed_by is not None and needed_by != b:
                        # Base is needed by 'needed_by', but 'b' is sitting on it
                        is_blocking = True
            
            if is_pos_correct and not is_blocking:
                fixed_count += 1
                # Continue traversal up the tower
                if b in on_top_of:
                    queue.extend(on_top_of[b])
            
            # If not fixed, we stop traversing this branch (blocks above are also not fixed)

        # Heuristic: Count of blocks that are NOT fixed
        # These blocks definitely need to be moved.
        h_val = num_blocks - fixed_count
        return h_val