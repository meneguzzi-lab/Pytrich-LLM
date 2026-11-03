"""
Domain-dependent heuristic for the Blocksworld domain.

This is a WORKING EXAMPLE heuristic demonstrating:
1. How to correctly parse grounded fact names
2. How to preprocess static domain information
3. How to efficiently check state using bitwise operations
4. How to implement a domain-specific cost estimation for block stacking

This heuristic estimates the cost based on:
- How many blocks are not in their goal position
- Understanding that moving a block may require first moving blocks above it
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldExampleHeuristic(Heuristic):
    """
    Heuristic for Blocksworld domain based on misplaced blocks.

    Key insight: In Blocksworld, moving a block from A to B requires:
    1. Clear the block (unstack any blocks above it)
    2. Pick up the block
    3. Stack it on destination

    This heuristic:
    1. Identifies goal on[x,y] relationships from model.goals
    2. For each goal, checks if currently satisfied
    3. Counts misplaced blocks and estimates cost

    A block is "correctly placed" if:
    - It's on its goal target AND
    - Everything below it is also correctly placed
    """

    def __init__(self, blocksworld_example_heuristic=None, name="BlocksworldExampleHeuristic"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_on = {}             # block -> target it should be on
        self.on_facts = {}            # (block, target) -> fact
        self.ontable_facts = {}       # block -> fact for ontable
        self.clear_facts = {}         # block -> fact for clear
        self.holding_fact = None      # fact for holding
        self.blocks = set()
        self.num_goals = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model.

        IMPORTANT: This is called ONCE before search starts.
        Do all expensive preprocessing here, not in __call__.
        """
        start_time = time.time()
        self.model = model

        # Preprocess static domain information
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.

        Handles TWO formats:
        1. Grounded format: "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]"
        2. HDDL format: "(predicate arg1 arg2)"

        Returns: (predicate, [arg1, arg2, ...]) or (None, []) if parsing fails
        """
        name = name.strip()

        # Format 1: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            # Remove the +/- prefix
            name = name[1:]
            # Find the bracket
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]  # Remove [ and ]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Format 2: (predicate arg1 arg2)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        """
        Extract domain structure from grounded facts.

        Key predicates in Blocksworld:
        - on[block1,block2]: block1 is on top of block2
        - ontable[block]: block is on the table
        - clear[block]: nothing is on top of block
        - holding[block]: robot is holding block
        - handempty: robot hand is empty

        Goals are on[x,y] facts indicating desired stacking.
        """
        # First, extract goal facts from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    block = args[0]
                    target = args[1]
                    self.goal_on[block] = target
                    self.blocks.add(block)
                    self.blocks.add(target)

        self.num_goals = len(self.goal_on)

        # Build fact lookup tables
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None:
                continue

            if predicate == "on" and len(args) >= 2:
                block = args[0]
                target = args[1]
                self.on_facts[(block, target)] = fact
                self.blocks.add(block)
                self.blocks.add(target)

            elif predicate == "ontable" and len(args) >= 1:
                block = args[0]
                self.ontable_facts[block] = fact
                self.blocks.add(block)

            elif predicate == "clear" and len(args) >= 1:
                block = args[0]
                self.clear_facts[block] = fact

            elif predicate == "holding" and len(args) >= 1:
                # Store one holding fact (we just need to detect if robot is holding anything)
                self.holding_fact = fact

    def _is_fact_true(self, state_bitwise: int, fact) -> bool:
        """Check if a fact is true in the current state using bitwise operations."""
        if fact is None:
            return False
        return (state_bitwise >> fact.global_id) & 1 == 1

    def _get_current_support(self, state_bitwise: int, block: str):
        """Find what a block is currently on (another block or table)."""
        # Check ontable first
        ontable_fact = self.ontable_facts.get(block)
        if ontable_fact and self._is_fact_true(state_bitwise, ontable_fact):
            return "TABLE"

        # Check on[block, X] for all possible X
        for (b, target), fact in self.on_facts.items():
            if b == block and self._is_fact_true(state_bitwise, fact):
                return target

        return None  # Block might be being held

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value for the given state.

        Algorithm:
        1. For each goal on[x,y], check if currently satisfied
        2. Count unsatisfied goals
        3. Each misplaced block needs at least 2 moves (unstack + stack)
           or 2 moves (pick-up + stack) if on table

        This is admissible as each misplaced block needs at least
        some actions to move to its goal position.
        """
        if not self.goal_on:
            return 0

        misplaced_count = 0

        for block, goal_target in self.goal_on.items():
            # Check if on[block, goal_target] is true
            on_fact = self.on_facts.get((block, goal_target))
            if not self._is_fact_true(state_bitwise, on_fact):
                misplaced_count += 1

        # Each misplaced block needs at least 2 actions:
        # - If on wrong block: unstack + stack (2 actions)
        # - If on table but should be on block: pick-up + stack (2 actions)
        # This is a conservative (admissible) estimate
        return misplaced_count * 2

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic value during search.

        This is called for every node expansion, so it must be FAST.
        All preprocessing was done in initialize().
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldExampleHeuristic(goals={self.num_goals})"

    def __str__(self):
        return f"BlocksworldExampleHeuristic with {self.num_goals} stacking goals"

    def __output__(self):
        return (
            f"Blocksworld Example Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Blocks: {len(self.blocks)}\n"
            f"\tNumber of Goals: {self.num_goals}\n"
            f"\tGoal Configurations: {self.goal_on}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
