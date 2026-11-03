import sys

# Import Heuristic from the appropriate module provided by the framework
from Pytrich.Heuristics.heuristic import Heuristic
class BlocksworldGemini3ProHeuristic13(Heuristic):
    """
    A domain-dependent heuristic for Blocksworld that estimates the number of moves
    required to reach the goal state.

    It works by:
    1. Reconstructing the current configuration of blocks (who is on whom) from the bitwise state.
    2. Identifying 'correct' blocks. A block is correct if:
       - It is in a position (on table or on another block) that satisfies the goal.
       - The block it rests on is also 'correct' (recursive foundation).
       - It is NOT obstructing a goal (i.e., it is not sitting on a block X when the goal requires block Y to be on X).
    3. Counting the number of incorrect blocks. The heuristic value is 2 * incorrect_count,
       estimating that each misplaced block requires at least an unstack and a stack operation.
    """

    def __init__(self, blocksworld_gemini3pro_heuristic_10):
        super().__init__(blocksworld_gemini3pro_heuristic_10)
        self.model = blocksworld_gemini3pro_heuristic_10
        
        # --- Pre-computation for Fast State Parsing ---
        # Map global_id -> (type, arg1_idx, arg2_idx)
        # Types: 1=ON, 2=ONTABLE, 3=HOLDING
        self.fact_data = {}
        self.TYPE_ON = 1
        self.TYPE_ONTABLE = 2
        self.TYPE_HOLDING = 3
        
        # Internal block ID mapping
        self.block_to_id = {}
        self.id_to_block = []
        self.num_blocks = 0
        
        # Locations
        self.LOC_TABLE = -1
        self.LOC_HAND = -2
        self.LOC_NONE = -3

        self._initialize_mappings()
        self._parse_goals()

    def _initialize_mappings(self):
        """
        Parses model facts to map string names to integer IDs for efficiency.
        """
        # Sort facts by ID to ensure consistent iteration
        sorted_facts = sorted(self.model.facts, key=lambda f: f.global_id)
        
        # 1. Collect all block names
        block_names = set()
        for fact in sorted_facts:
            name = fact.name.strip()
            # Expecting format: +predicate[arg1,arg2]
            if name.startswith("+") or name.startswith("-"):
                content = name[1:]
                if "[" in content and content.endswith("]"):
                    pred, args_str = content.split("[", 1)
                    args_str = args_str.strip("]")
                    args = [a.strip() for a in args_str.split(",") if a.strip()]
                    
                    if pred in ["on", "ontable", "holding", "clear"]:
                        for arg in args:
                            block_names.add(arg)
        
        self.id_to_block = sorted(list(block_names))
        self.block_to_id = {name: i for i, name in enumerate(self.id_to_block)}
        self.num_blocks = len(self.id_to_block)
        
        # 2. Map global_id to integer tuples
        for fact in sorted_facts:
            name = fact.name.strip()
            if not (name.startswith("+") or name.startswith("-")):
                continue
                
            content = name[1:]
            if "[" not in content:
                continue
            
            pred, args_str = content.split("[", 1)
            args_str = args_str.strip("]")
            args = [a.strip() for a in args_str.split(",") if a.strip()]
            
            f_type = 0
            a1 = -1
            a2 = -1
            
            if pred == "on" and len(args) == 2:
                f_type = self.TYPE_ON
                a1 = self.block_to_id.get(args[0], -1)
                a2 = self.block_to_id.get(args[1], -1)
            elif pred == "ontable" and len(args) == 1:
                f_type = self.TYPE_ONTABLE
                a1 = self.block_to_id.get(args[0], -1)
            elif pred == "holding" and len(args) == 1:
                f_type = self.TYPE_HOLDING
                a1 = self.block_to_id.get(args[0], -1)
            
            if f_type != 0 and a1 != -1:
                self.fact_data[fact.global_id] = (f_type, a1, a2)

    def _parse_goals(self):
        """
        Parses the static bitwise goal state into a lookup array.
        """
        goals = self.model.goals
        
        # goal_pos[block_id] = target_location (block_id or LOC_TABLE or LOC_NONE)
        self.goal_pos = [self.LOC_NONE] * self.num_blocks
        
        # Reverse lookup: goal_support_for[block_id] = block_that_wants_to_be_on_top
        # Used to detect if a block is obstructing a goal position.
        self.goal_support_for = {}

        if goals == 0:
            return

        # Efficient bit iteration
        temp_goals = goals
        while temp_goals:
            lsb = temp_goals & -temp_goals
            gid = lsb.bit_length() - 1
            temp_goals ^= lsb
            
            if gid in self.fact_data:
                f_type, a1, a2 = self.fact_data[gid]
                if f_type == self.TYPE_ON:
                    self.goal_pos[a1] = a2
                    self.goal_support_for[a2] = a1
                elif f_type == self.TYPE_ONTABLE:
                    self.goal_pos[a1] = self.LOC_TABLE

    def h(self, node):
        state = node.state
        
        # 1. Reconstruct current state from bitwise representation
        current_pos = [self.LOC_NONE] * self.num_blocks
        
        temp_state = state
        while temp_state:
            lsb = temp_state & -temp_state
            gid = lsb.bit_length() - 1
            temp_state ^= lsb
            
            if gid in self.fact_data:
                f_type, a1, a2 = self.fact_data[gid]
                if f_type == self.TYPE_ON:
                    current_pos[a1] = a2
                elif f_type == self.TYPE_ONTABLE:
                    current_pos[a1] = self.LOC_TABLE
                elif f_type == self.TYPE_HOLDING:
                    current_pos[a1] = self.LOC_HAND

        # 2. Determine which blocks are in a 'correct' state
        # is_correct[b] is True if b is part of a valid tower foundation up to b.
        is_correct = [False] * self.num_blocks
        
        # We iterate until convergence to propagate correctness from the table up.
        changed = True
        while changed:
            changed = False
            for b in range(self.num_blocks):
                if is_correct[b]:
                    continue
                
                loc = current_pos[b]
                
                # Condition A: Foundation must be correct
                foundation_correct = False
                if loc == self.LOC_TABLE:
                    foundation_correct = True
                elif loc >= 0: # Resting on another block
                    if is_correct[loc]:
                        foundation_correct = True
                # If LOC_HAND or LOC_NONE, foundation is not correct
                
                if not foundation_correct:
                    continue

                # Condition B: Must satisfy goal (or have no goal constraint)
                target = self.goal_pos[b]
                matches_goal = False
                if target == self.LOC_NONE:
                    matches_goal = True
                elif target == loc:
                    matches_goal = True
                
                if not matches_goal:
                    continue

                # Condition C: Must not be obstructing a goal
                # If b is on 'loc', check if any OTHER block 'z' has 'loc' as its target.
                # If so, b is in the way.
                blocking = False
                if loc >= 0:
                    # See if 'loc' is reserved for someone else
                    desired_user = self.goal_support_for.get(loc, -1)
                    if desired_user != -1 and desired_user != b:
                        blocking = True
                
                if not blocking:
                    is_correct[b] = True
                    changed = True
        
        # 3. Calculate heuristic based on incorrect blocks
        incorrect_count = 0
        for b in range(self.num_blocks):
            if not is_correct[b]:
                incorrect_count += 1
        
        # Multiplier 2 accounts for unstack + stack cost roughly
        return incorrect_count * 2
