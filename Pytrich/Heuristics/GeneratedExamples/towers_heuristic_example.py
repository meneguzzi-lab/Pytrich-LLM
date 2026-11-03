"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

This is a WORKING EXAMPLE heuristic demonstrating:
1. How to correctly parse grounded fact names
2. How to preprocess static domain information
3. How to efficiently check state using bitwise operations
4. How to implement the optimal 2^k-1 heuristic for Towers of Hanoi

This heuristic is ADMISSIBLE and provides tight lower bounds.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersExampleHeuristic(Heuristic):
    """
    Optimal heuristic for Towers of Hanoi based on the classic solution.

    The minimum number of moves to solve a k-disk Towers of Hanoi is 2^k - 1.

    This heuristic:
    1. Identifies which rings are NOT in their goal position
    2. Finds the LARGEST ring that needs to move
    3. Returns 2^k - 1 where k is the number of rings from smallest to that largest ring

    This is admissible because moving the largest unsettled ring requires
    first moving all smaller rings out of the way.
    """

    def __init__(self, towers_example_heuristic=None, name="TowersExampleHeuristic"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring -> target object (from goal_on facts)
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0

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

        Grounded facts can have two formats:
        1. "+predicate[arg1,arg2]" - e.g., "+on[r1,r2]", "+goal_on[r1,t3]"
        2. "(predicate arg1 arg2)" - e.g., "(on r1 r2)", "(goal_on r1 t3)"

        Both formats are handled by _parse_fact_name().

        IMPORTANT: Goal positions are extracted from model.goals (bitwise integer),
        NOT from goal_on predicates. Use model.state_explicit_repr(model.goals)
        to get the list of goal fact names.
        """
        rings = set()
        smaller_than = []  # List of (smaller_ring, larger_obj) pairs

        # First, extract goal positions from model.goals
        # model.goals is a bitwise integer representing the goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                # smallerThan defines size ordering
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                # on is dynamic - we need to track which facts represent each ring's position
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)

        if self.num_rings == 0:
            return

        # Sort rings by size
        # Method 1: Use smallerThan facts if available
        if smaller_than:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than:
                if smaller in rings:
                    smaller_count[smaller] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Method 2: Infer from goal state - if r1 is ON r2, then r1 < r2
            # Build a graph of "is smaller than" relationships
            is_smaller_than = defaultdict(set)  # ring -> set of rings it's smaller than
            for ring, target in self.goal_positions.items():
                if target.startswith("r"):
                    # ring is on target, so ring is smaller than target
                    is_smaller_than[ring].add(target)

            # Transitively expand: if a < b and b < c, then a < c
            changed = True
            while changed:
                changed = False
                for ring in list(is_smaller_than.keys()):
                    for larger in list(is_smaller_than[ring]):
                        for even_larger in is_smaller_than.get(larger, set()):
                            if even_larger not in is_smaller_than[ring]:
                                is_smaller_than[ring].add(even_larger)
                                changed = True

            # Count how many rings each ring is smaller than
            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            # Sort: ring with highest count is smallest (on top of more things)
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Find where a ring currently is by checking which 'on' fact is true.

        Uses bitwise state checking for efficiency:
          is_true = (state_bitwise >> fact.global_id) & 1
        """
        if ring not in self.on_facts_by_ring:
            return None

        for fact, target in self.on_facts_by_ring[ring]:
            # Check if this fact is true in the current state
            if (state_bitwise >> fact.global_id) & 1:
                return target

        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value for the given state.

        Algorithm:
        1. For each ring (largest to smallest), check if it's at goal
        2. Find the LARGEST ring that is NOT at its goal position
        3. Return 2^k - 1 where k is the count of rings from smallest to that ring

        This is admissible because to move the largest misplaced ring,
        we must first move all smaller rings (at least once each).
        """
        if not self.rings_by_size:
            return 0

        # Check rings from largest to smallest
        # rings_by_size is sorted smallest to largest, so we iterate in reverse
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]

            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)

            if current_pos != goal_pos:
                # This ring (and all smaller ones) need to move
                # There are i+1 rings from index 0 to index i (inclusive)
                num_rings_to_move = i + 1

                # Minimum moves for k-disk Towers of Hanoi is 2^k - 1
                return (1 << num_rings_to_move) - 1

        # All rings are at goal positions
        return 0

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
        return f"TowersExampleHeuristic(rings={self.num_rings})"

    def __str__(self):
        return f"TowersExampleHeuristic with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers Example Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
