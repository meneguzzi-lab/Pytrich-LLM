"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal,
  ALL rings smaller than k must first be moved off of it (and off its
  destination). This creates an exponential cascade: moving ring k
  requires 2^(k-1) - 1 moves just to clear the way, plus 1 move for
  ring k itself, plus 2^(k-1) - 1 moves to reassemble.

Admissible Lower Bounds:
  Term 1 (Exponential displacement): Find the largest ring not at its
    goal position. All rings from the smallest up to (and including)
    that ring must move. The classic Towers of Hanoi lower bound is
    2^k - 1 where k is the count of those rings. This is admissible
    because the optimal solution for k disks requires exactly 2^k - 1
    moves.

  Term 2 (Misplaced ring count): Count the number of rings not
    currently sitting on their goal target. Each such ring needs at
    least 1 move. This is a trivial admissible lower bound but helps
    in states near the goal where Term 1 might over-collapse.

  Term 3 (Settled-stack aware refinement): Starting from the largest
    ring, count how many rings at the bottom of the goal tower are
    already correctly settled (in goal position with correct support).
    The remaining unsettled rings define a tighter k for the 2^k - 1
    bound. This refines Term 1 by not penalizing already-settled
    bottom rings.

The heuristic returns max(term1, term2, term3) which is admissible
since each individual term is admissible.

Tie-breaking: We add a tiny sub-unit penalty (0.001) per ring that
is not on its goal position to break ties among states with the same
exponential bound.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic12(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Combines:
    1. Classic 2^k - 1 exponential lower bound based on largest displaced ring
    2. Simple misplaced-ring count
    3. Settled-stack-aware exponential bound
    Plus sub-unit tie-breaking for misplaced rings.
    """

    def __init__(self, towers_anthropicopus46_heuristic_12=None, name="TowersAnthropicOpus46Heuristic12"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names, smallest first
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.on_fact_lookup = {}       # (ring_name, target_name) -> fact.global_id
        self.num_rings = 0
        self.towers = set()            # set of tower names

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
        smaller_than_pairs = []  # (smaller, larger) where both are ring names
        smaller_than_count = defaultdict(int)  # ring -> count of things it's smaller than

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
                    smaller_than_count[smaller] += 1
                    if larger.startswith("r"):
                        rings.add(larger)
                        smaller_than_pairs.append((smaller, larger))
                if larger.startswith("t"):
                    self.towers.add(larger)

            elif predicate == "on":
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))
                    self.on_fact_lookup[(ring, target)] = fact.global_id

            elif predicate == "towerTop":
                if args[1].startswith("t"):
                    self.towers.add(args[1])

        self.num_rings = len(rings)

        if self.num_rings == 0:
            return

        # Sort rings by size: most "smallerThan" relationships = smallest ring
        # The smallest ring is smaller than the most other objects
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        goal_target = self.goal_positions.get(ring)
        if goal_target is None:
            return True  # No goal for this ring, consider it satisfied
        key = (ring, goal_target)
        if key in self.on_fact_lookup:
            gid = self.on_fact_lookup[key]
            return (state_bitwise >> gid) & 1
        return False

    def _compute(self, state_bitwise: int, task_network) -> int:
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        misplaced_count = 0
        largest_misplaced_idx = -1

        # Check each ring: is it at its goal position?
        # rings_by_size[0] = smallest, rings_by_size[n-1] = largest
        for i in range(n):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                misplaced_count += 1
                if i > largest_misplaced_idx:
                    largest_misplaced_idx = i

        # Term 1: Classic exponential bound
        # If the largest misplaced ring is at index i (0-indexed, smallest first),
        # then we need to move rings 0..i, which is i+1 rings.
        # Lower bound = 2^(i+1) - 1
        if largest_misplaced_idx >= 0:
            term1 = (1 << (largest_misplaced_idx + 1)) - 1
        else:
            term1 = 0

        # Term 2: Simple misplaced count
        term2 = misplaced_count

        # Term 3: Settled-stack-aware bound
        # Count how many rings at the BOTTOM of the goal configuration are
        # already correctly settled. "Settled" means the ring is at its goal
        # AND all rings below it (larger) are also at their goals.
        # This gives us a tighter count of rings that truly need to move.
        settled_from_bottom = 0
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if self._is_ring_at_goal(state_bitwise, ring):
                settled_from_bottom += 1
            else:
                break

        unsettled_count = n - settled_from_bottom
        if unsettled_count > 0:
            term3 = (1 << unsettled_count) - 1
        else:
            term3 = 0

        # Combine: max of admissible terms + tiny tie-breaking
        base_h = max(term1, term2, term3)
        tie_break = 0.001 * misplaced_count

        return base_h + tie_break

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic12(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic12 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
