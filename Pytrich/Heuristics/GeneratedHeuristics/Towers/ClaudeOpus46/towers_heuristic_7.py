"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single constraint that most serializes work is the size ordering:
  to move ring k to its goal, ALL rings 1..k-1 (smaller) must first be
  moved off of ring k, requiring at least 2^(k-1) - 1 moves, plus the
  move of ring k itself. This cascading dependency is the fundamental
  serializing constraint.

Admissible Lower Bounds (combined via max):

  Term 1 — Classic exponential bound:
    Find the largest ring not at its goal position. To move that ring,
    all smaller rings must be cleared first. If ring at index k (0-indexed
    from smallest) is the largest misplaced ring, the minimum moves is
    2^(k+1) - 1. This is the classic Towers of Hanoi lower bound and is
    admissible because no solution can do fewer moves.

  Term 2 — Additive misplaced-ring count:
    Count the number of rings not on their goal target. Each such ring
    requires at least one move, so this count is a (weak) admissible
    lower bound. While dominated by Term 1 in most cases, it can help
    in degenerate edge cases and serves as a sanity floor.

  Term 3 — Settled-stack aware refinement:
    Starting from the largest ring, count how many consecutive rings
    (from bottom up) are already correctly stacked on their goal
    positions. The first ring (from the bottom) that is NOT settled
    defines the "break point". All rings above (and including) this
    break point must move. If there are m such rings, the lower bound
    is 2^m - 1. This is tighter than Term 1 in cases where some bottom
    rings are already settled. Admissible because settled bottom rings
    never need to move, and the remaining m rings form a sub-problem
    requiring at least 2^m - 1 moves.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic7(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines three admissible lower bounds:
    1. Classic 2^k - 1 bound based on largest misplaced ring
    2. Simple misplaced ring count
    3. Settled-stack aware bound that accounts for rings already
       correctly placed at the bottom of the goal tower
    """

    def __init__(self, towers_anthropicopus46_heuristic_7=None, name="TowersAnthropicOpus46Heuristic7"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names, smallest first
        self.on_facts_by_ring = {}     # ring_name -> list of (Fact, target_obj_name)
        self.goal_on_fact_ids = {}     # ring_name -> global_id of the goal 'on' fact
        self.num_rings = 0

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
        """
        Extract domain structure from grounded facts.
        Build fast-lookup structures for state evaluation.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller_ring, larger_obj)
        
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

        # Process all facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_pairs.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))
                    # Check if this is the goal on-fact for this ring
                    if ring in self.goal_positions and target == self.goal_positions[ring]:
                        self.goal_on_fact_ids[ring] = fact.global_id

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size using smallerThan count
        # A ring that is smaller than more objects is a smaller ring
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than_pairs:
                if smaller in rings:
                    smaller_count[smaller] += 1
            # More "smallerThan" relationships = smaller ring = earlier in sorted list
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by ring number
            def ring_num(r):
                try:
                    return int(r[1:])
                except ValueError:
                    return 0
            self.rings_by_size = sorted(list(rings), key=ring_num)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """Find where a ring currently is using fast bitwise checks."""
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """Fast check if ring is at its goal position using precomputed fact id."""
        if ring in self.goal_on_fact_ids:
            return (state_bitwise >> self.goal_on_fact_ids[ring]) & 1
        # Fallback: ring has no goal, consider it settled
        return True

    def _term1_classic_exponential(self, state_bitwise: int) -> int:
        """
        Term 1: Classic 2^k - 1 bound.
        Find the largest misplaced ring (scanning from largest to smallest).
        All rings from smallest up to (and including) that ring must move.
        """
        if not self.rings_by_size:
            return 0
        # rings_by_size[0] = smallest, rings_by_size[-1] = largest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                return (1 << (i + 1)) - 1
        return 0

    def _term2_misplaced_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count of rings not at goal position.
        Each misplaced ring needs at least 1 move. Admissible.
        """
        count = 0
        for ring in self.rings_by_size:
            if not self._is_ring_at_goal(state_bitwise, ring):
                count += 1
        return count

    def _term3_settled_stack(self, state_bitwise: int) -> int:
        """
        Term 3: Settled-stack aware bound.
        
        Count how many rings from the BOTTOM of the goal configuration are
        already correctly placed AND correctly stacked on each other.
        The remaining unsettled rings form a sub-problem of size m,
        requiring at least 2^m - 1 moves.
        
        A ring is "settled" if:
          1. It is on its goal target, AND
          2. ALL rings below it (larger rings) are also settled
        
        This is tighter than Term 1 when some bottom rings are already in place.
        """
        if not self.rings_by_size:
            return 0
        
        # Count settled rings from the bottom (largest first)
        settled_count = 0
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if self._is_ring_at_goal(state_bitwise, ring):
                settled_count += 1
            else:
                break  # Once we find an unsettled ring, all above are unsettled
        
        unsettled_count = self.num_rings - settled_count
        if unsettled_count == 0:
            return 0
        return (1 << unsettled_count) - 1

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute heuristic as max of three admissible terms.
        """
        if not self.rings_by_size:
            return 0
        
        # If task network is empty and we still have goals unmet, return large value
        # If task network is empty and goals met, return 0
        
        t1 = self._term1_classic_exponential(state_bitwise)
        t2 = self._term2_misplaced_count(state_bitwise)
        t3 = self._term3_settled_stack(state_bitwise)
        
        h = max(t1, t2, t3)
        
        # Small tie-breaking: prefer states where more tasks are resolved
        # This helps guide search when h-values are equal
        if task_network is not None:
            tn_len = len(task_network)
            # Add a tiny penalty proportional to task network size
            # This breaks ties in favor of nodes closer to completion
            h = h + 0.001 * tn_len
        
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic7(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic7 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 7 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
