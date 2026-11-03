"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings 1..k-1 (smaller rings) must first be moved out of the way. This creates
  an exponential lower bound: moving n rings requires at least 2^n - 1 moves.

Admissible Lower Bounds:
  Term 1 (Exponential structural bound): Find the largest ring not at its goal.
    All rings from smallest up to that ring must move at least once each in a
    specific pattern. The minimum is 2^k - 1 where k = number of rings from
    smallest to the largest misplaced ring (inclusive). This is admissible because
    the classic Tower of Hanoi solution is optimal.

  Term 2 (Misplaced ring count): Count the number of rings not currently on their
    goal target. Each such ring needs at least 1 move. This is trivially admissible.
    While dominated by Term 1 in most cases, it provides a tighter bound when rings
    are scattered but the largest misplaced ring is small.

  Term 3 (Goal-stability / chain disruption): A ring may currently be at its goal
    position but will need to move because a ring below it is NOT at its goal.
    We count these "falsely settled" rings. Each such ring needs at least 2 moves
    (move away, then move back). This catches cases where Term 1 underestimates
    because it only looks at the largest misplaced ring but ignores disruption costs
    for rings that appear settled but aren't.

The final heuristic returns max(term1, term2, term3).
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic9(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential structural bound (2^k - 1)
    2. Misplaced ring count
    3. Goal-stability disruption penalty
    """

    def __init__(self, towers_anthropicopus46_heuristic_9=None, name="TowersAnthropicOpus46Heuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For goal-stability analysis: which ring is ON which ring in the goal
        # goal_support[ring] = the object ring should be ON in the goal
        # goal_supported_by[obj] = list of rings that should be ON obj in the goal
        self.goal_support = {}         # ring -> obj it should be on
        
        # Ring index for quick lookup
        self.ring_index = {}           # ring_name -> index in rings_by_size (0=smallest)
        
        # Precomputed powers of 2
        self._pow2 = []

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node.state)
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
        smaller_than_pairs = []  # (smaller, larger) where both are rings
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
                        self.goal_support[ring] = target

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

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size: most smaller_count = smallest ring
        if smaller_count:
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by ring number
            def ring_num(r):
                try:
                    return int(r[1:])
                except:
                    return 0
            self.rings_by_size = sorted(list(rings), key=ring_num)

        # Build ring index
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

        # Precompute powers of 2
        self._pow2 = [1 << i for i in range(self.num_rings + 2)]

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        
        # Compute current positions for all rings
        current_pos = {}
        is_at_goal = {}
        for ring in self.rings_by_size:
            cp = self._get_current_position(state_bitwise, ring)
            current_pos[ring] = cp
            goal_pos = self.goal_positions.get(ring)
            is_at_goal[ring] = (cp == goal_pos)

        # --- Term 1: Exponential structural bound ---
        # Find the largest ring not at its goal position
        term1 = 0
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not is_at_goal[ring]:
                # i+1 rings (indices 0..i) must all move
                term1 = self._pow2[i + 1] - 1
                break

        # --- Term 2: Simple misplaced count ---
        term2 = sum(1 for ring in self.rings_by_size if not is_at_goal[ring])

        # --- Term 3: Goal-stability / chain disruption ---
        # A ring appears "settled" (at goal) but sits on a chain where
        # some ring below it is NOT at its goal. Such a ring must move
        # at least twice (away and back).
        # 
        # We check: for each ring at its goal, trace downward through the
        # goal chain. If any ring below is not at its goal, this ring is
        # "falsely settled" and needs 2 extra moves.
        #
        # Actually, for the standard Towers of Hanoi where all rings start
        # on one peg and go to another, this is subsumed by Term 1.
        # But for partial-progress states it can help.
        
        # Build: for each ring at goal, check if the ring it's supposed to
        # be on is also at goal (recursively).
        falsely_settled = 0
        for ring in self.rings_by_size:
            if is_at_goal[ring]:
                # Check what this ring should be on
                support = self.goal_support.get(ring)
                if support and support.startswith("r"):
                    # support is a ring - check if it's at its goal
                    if not is_at_goal.get(support, True):
                        falsely_settled += 1
                # Also check: is the ring currently ON the right object,
                # but that object itself is in the wrong place?
                # For towers, if ring is on ring_below, and ring_below
                # is not at its goal, then ring will need to move.
        
        # Each falsely settled ring needs at least 2 moves
        term3 = term2 + 2 * falsely_settled

        # --- Enhanced Term 1: Account for settled rings above misplaced ones ---
        # If there's a settled ring at index j < i (smaller than the largest
        # misplaced ring at index i), and this settled ring is NOT on its
        # goal support (which is also misplaced), then the exponential bound
        # already accounts for it. But we can add the falsely_settled count
        # to see if it exceeds 2^k - 1.
        
        # For the deepest misplaced ring analysis:
        # Find the deepest ring in the GOAL configuration that is not at goal
        # In the goal, rings are stacked on the target tower.
        # The deepest misplaced ring determines the minimum work.
        # This is what term1 already computes.
        
        # Additional refinement: if multiple rings are scattered across
        # different non-goal towers, we might need extra moves.
        # Count rings on each tower in current state to detect scatter.
        
        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic9(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic9 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 9 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
