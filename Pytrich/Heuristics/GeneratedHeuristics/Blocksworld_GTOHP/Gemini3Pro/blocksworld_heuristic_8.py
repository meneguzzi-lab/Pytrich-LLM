import sys

# Attempt to import the base Heuristic class. 
# Adjust the import path according to the specific Pytrich installation if necessary.
from Pytrich.Heuristics.heuristic import Heuristic
class BlocksworldGemini3ProHeuristic8(Heuristic):
    def __init__(self, blocksworld_gemini3pro_heuristic_5):
        super().__init__(blocksworld_gemini3pro_heuristic_5)
        self.model = blocksworld_gemini3pro_heuristic_5
        self.facts_initialized = False

        # Caches for fast state reconstruction
        self.on_facts_indices = []      # List of (global_id, top_block, bottom_block)
        self.ontable_facts_indices = [] # List of (global_id, block)
        self.holding_facts_indices = [] # List of (global_id, block)
        
        # Goal representation: block -> target_block_or_table
        # 'table' is represented by the string "table"
        self.goal_map = {}
        self.all_blocks = set()

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

    def _initialize_facts(self):
        if self.facts_initialized:
            return

        # 1. Parse all facts in the model to build index lists
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue
            
            if predicate == "on" and len(args) == 2:
                self.on_facts_indices.append((fact.global_id, args[0], args[1]))
                self.all_blocks.add(args[0])
                self.all_blocks.add(args[1])
            elif predicate == "ontable" and len(args) == 1:
                self.ontable_facts_indices.append((fact.global_id, args[0]))
                self.all_blocks.add(args[0])
            elif predicate == "holding" and len(args) == 1:
                self.holding_facts_indices.append((fact.global_id, args[0]))
                self.all_blocks.add(args[0])

        # 2. Parse Goals
        # model.goals is a bitwise integer. We convert it to explicit strings to parse.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            predicate, args = self._parse_fact_name(name)
            if predicate == "on" and len(args) == 2:
                self.goal_map[args[0]] = args[1]
            elif predicate == "ontable" and len(args) == 1:
                self.goal_map[args[0]] = "table"
            # We typically ignore 'clear' or 'handempty' in goal for the heuristic 
            # as 'on' relations define the structural cost.

        self.facts_initialized = True

    def infer(self, node):
        """
        Domain-dependent heuristic for Blocksworld.
        Estimates the number of moves required to reach the goal state.
        Uses the 'stable/correct' block logic: a block is stable if it matches 
        its goal position and rests on a stable stack.
        """
        if not self.facts_initialized:
            self._initialize_facts()

        state = node.state
        
        # 1. Reconstruct current topology from state bits
        # parent_map: child -> parent (where parent is block_name or "table")
        parent_map = {}
        holding_block = None

        # Check on(x, y)
        for gid, top, bot in self.on_facts_indices:
            if (state >> gid) & 1:
                parent_map[top] = bot

        # Check ontable(x)
        for gid, block in self.ontable_facts_indices:
            if (state >> gid) & 1:
                parent_map[block] = "table"

        # Check holding(x)
        for gid, block in self.holding_facts_indices:
            if (state >> gid) & 1:
                holding_block = block

        # 2. Calculate Stability
        # A block is stable if:
        # - It is on the correct target (as per goal)
        # - AND the target itself is stable
        # If a block has no goal position, it is stable if it rests on a stable stack
        # (conceptually, it doesn't need to move unless it's blocking something, 
        # but for the 'move count' heuristic, we count cost if it sits on an unstable base).

        memo_stable = {}

        def is_stable(block):
            if block == "table":
                return True
            
            if block in memo_stable:
                return memo_stable[block]
            
            # If block is held, it's not stable on the board
            if block == holding_block:
                memo_stable[block] = False
                return False

            current_parent = parent_map.get(block)
            
            # If block location unknown (e.g. held but logic missed it, or inconsistent), assume unstable
            if current_parent is None:
                memo_stable[block] = False
                return False

            goal_parent = self.goal_map.get(block)

            if goal_parent is not None:
                # Block has a specific goal
                if current_parent != goal_parent:
                    # Wrong place
                    memo_stable[block] = False
                    return False
                else:
                    # Right place, but is the base stable?
                    res = is_stable(current_parent)
                    memo_stable[block] = res
                    return res
            else:
                # Block has no specific goal.
                # It is stable if its base is stable.
                # (If base moves, this block must move)
                res = is_stable(current_parent)
                memo_stable[block] = res
                return res

        # 3. Sum Costs
        # Basic heuristic: Count blocks that are NOT stable.
        # If a block is not stable, it must be moved.
        # - If on table/stack: Cost = 2 (Unstack + Stack)
        # - If held: Cost = 1 (Stack)
        
        cost = 0
        for block in self.all_blocks:
            if not is_stable(block):
                if block == holding_block:
                    cost += 1
                else:
                    cost += 2
        
        return float(cost)
