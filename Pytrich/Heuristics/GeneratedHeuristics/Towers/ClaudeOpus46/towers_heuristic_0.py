"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single constraint that most serializes work is the size ordering:
  to move ring k to its goal, ALL rings 1..k-1 (smaller rings) must first
  be moved out of the way. This creates an exponential cascade where moving
  ring k requires 2^(k-1) moves just for the smaller rings.

Admissible Lower Bounds:
  Term 1 (Largest Displaced Ring): Find the largest ring not at its goal
    position. To move it, all smaller rings must be cleared first, giving
    a lower bound of 2^k - 1 where k = number of rings from smallest to
    that largest displaced ring. Admissible because the classic Tower of
    Hanoi solution is optimal.

  Term 2 (Count of Displaced Rings): Count rings not at their goal position.
    Each displaced ring requires at least 1 move, so this is a trivial
    admissible lower bound. While weaker than Term 1, it can help when
    rings are scattered across towers in non-standard configurations.

  Term 3 (Bottom-up goal chain analysis): Check how many rings from the
    bottom of the goal tower are already correctly stacked. The remaining
    rings that are NOT correctly settled from the bottom up each require
    at least one move. More precisely, if the bottom k rings of the goal
    are settled, then the remaining n-k rings give a lower bound of
    2^(n-k) - 1. This is admissible since settled bottom rings never need
    to move again.

The heuristic returns max(term1, term2, term3).
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic0(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Combines three admissible lower bounds via max() for strong guidance.
    All preprocessing is done in initialize(); __call__ uses only O(1)
    bitwise operations per ring.
    """

    def __init__(self, towers_anthropicopus46_heuristic_0=None, name="TowersAnthropicOpus46Heuristic0"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.on_fact_lookup = {}       # (ring_name, target_name) -> fact.global_id
        self.num_rings = 0
        self.goal_tower = None         # The tower that is the ultimate destination
        self.goal_stack_order = []     # Rings in order from bottom to top on goal tower

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        rings = set()
        smaller_than_pairs = []  # (smaller, larger) pairs where both are rings
        smaller_count = defaultdict(int)  # ring -> count of things it's smaller than

        # Extract goal positions from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # Parse all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_count[smaller] += 1
                    if larger.startswith("r"):
                        rings.add(larger)
                        smaller_than_pairs.append((smaller, larger))

            elif predicate == "on":
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))
                    self.on_fact_lookup[(ring, target)] = fact.global_id

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size: most "smaller than" relationships = smallest ring
        if smaller_count:
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            self.rings_by_size = sorted(list(rings))

        # Determine the goal tower and goal stack order
        # The largest ring's goal position should be a tower
        if self.rings_by_size:
            largest_ring = self.rings_by_size[-1]
            goal_target = self.goal_positions.get(largest_ring)
            if goal_target and goal_target.startswith("t"):
                self.goal_tower = goal_target

            # Build goal stack order: from bottom (largest) to top (smallest)
            # by following the goal_on chain
            if self.goal_tower:
                # Build reverse mapping: target -> ring that should be on it
                target_to_ring = {}
                for ring, target in self.goal_positions.items():
                    target_to_ring[target] = ring

                # Start from the goal tower and walk up
                self.goal_stack_order = []
                current = self.goal_tower
                while current in target_to_ring:
                    ring = target_to_ring[current]
                    self.goal_stack_order.append(ring)
                    current = ring
                # goal_stack_order is now bottom-to-top (largest to smallest)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _term1_largest_displaced(self, state_bitwise: int) -> int:
        """
        Term 1: Find the largest ring not at its goal position.
        Return 2^k - 1 where k is the number of rings from smallest
        to that largest displaced ring (inclusive).
        
        Admissible: this is the exact minimum for standard Tower of Hanoi.
        """
        if not self.rings_by_size:
            return 0

        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                continue

            # Check if ring is at goal using fast lookup
            fact_key = (ring, goal_pos)
            if fact_key in self.on_fact_lookup:
                gid = self.on_fact_lookup[fact_key]
                if (state_bitwise >> gid) & 1:
                    continue  # Ring is at goal

            # Ring i is displaced; all rings 0..i need to move
            num_rings_to_move = i + 1
            return (1 << num_rings_to_move) - 1

        return 0

    def _term2_displaced_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count of rings not at their goal position.
        Each displaced ring needs at least 1 move.
        
        Admissible: trivially, each ring not at goal needs >= 1 action.
        """
        count = 0
        for ring in self.rings_by_size:
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                continue
            fact_key = (ring, goal_pos)
            if fact_key in self.on_fact_lookup:
                gid = self.on_fact_lookup[fact_key]
                if not ((state_bitwise >> gid) & 1):
                    count += 1
            else:
                count += 1
        return count

    def _term3_settled_bottom_up(self, state_bitwise: int) -> int:
        """
        Term 3: Bottom-up goal chain analysis.
        Count how many rings from the bottom of the goal stack are correctly
        settled (in correct position AND on the correct thing beneath them).
        The remaining unsettled rings give a lower bound of 2^(unsettled) - 1.
        
        Admissible: settled bottom rings never need to move; the unsettled
        rings above them form a sub-Tower-of-Hanoi problem.
        """
        if not self.goal_stack_order:
            return 0

        settled = 0
        for ring in self.goal_stack_order:  # bottom to top
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                break
            fact_key = (ring, goal_pos)
            if fact_key in self.on_fact_lookup:
                gid = self.on_fact_lookup[fact_key]
                if (state_bitwise >> gid) & 1:
                    settled += 1
                else:
                    break
            else:
                break

        unsettled = self.num_rings - settled
        if unsettled <= 0:
            return 0
        return (1 << unsettled) - 1

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute heuristic as max of three admissible terms.
        """
        if not self.rings_by_size:
            return 0

        t1 = self._term1_largest_displaced(state_bitwise)
        t2 = self._term2_displaced_count(state_bitwise)
        t3 = self._term3_settled_bottom_up(state_bitwise)

        # Also factor in task network: if there are no tasks left, h=0
        if len(task_network) == 0:
            return 0

        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic0(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic0 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 0 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tGoal Tower: {self.goal_tower}\n"
            f"\tGoal Stack Order (bottom to top): {self.goal_stack_order}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
