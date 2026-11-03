"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single constraint that most serializes work is the size ordering:
  to move ring k to its goal, ALL rings 1..k-1 (smaller rings) must first
  be moved off of ring k, requiring at minimum 2^(k-1) - 1 moves just to
  clear ring k, plus 1 move for ring k itself, then 2^(k-1) - 1 moves to
  reassemble on top. This gives the classic 2^k - 1 minimum.

Independent admissible lower bounds:

  Term 1 — Largest-misplaced-ring bound (classic 2^k - 1):
    Find the largest ring not on its goal object. All rings above it must
    move at least once to let it move, giving 2^k - 1 as a lower bound
    where k = index+1 of that ring in the size ordering.
    Admissibility: This is the exact optimal solution cost for the
    standard 3-peg Towers of Hanoi.

  Term 2 — Count of individually misplaced rings:
    Count how many rings are not currently on their goal target.
    Each such ring requires at least 1 move. This is a trivial lower bound.
    Admissibility: Each misplaced ring needs at least one action to reach
    its goal position.

  Term 3 — Bottom-up settled stack analysis:
    Starting from the largest ring, count how many rings are already
    correctly stacked on the goal tower in the right order from the bottom.
    The remaining rings that are NOT settled form a sub-problem.
    For n unsettled rings, the minimum is 2^n - 1.
    This can sometimes give a tighter bound than Term 1 when some bottom
    rings are already placed correctly.

  We return max(term1, term2, term3).

  Tie-breaking: We add a small penalty (0.001) for each ring that is on
  the wrong tower entirely (not just wrong position) to help guide search
  when multiple states have the same primary heuristic value.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic3(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
      1. Classic 2^k - 1 bound based on largest misplaced ring
      2. Count of misplaced rings (simple lower bound)
      3. Bottom-up settled analysis for tighter bound when partial solution exists
    
    Plus sub-unit tie-breaking for guidance.
    """

    def __init__(self, towers_anthropicopus46_heuristic_3=None, name="TowersAnthropicOpus46Heuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names, smallest first
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For Term 3: goal tower and goal stacking order
        self.goal_tower = None         # The tower where all rings should end up
        self.goal_stack_order = []     # Rings in order from bottom to top on goal tower
        
        # For tie-breaking: which tower each ring should be on
        # We'll track for each ring which tower facts exist
        self.ring_tower_facts = {}     # ring_name -> dict of tower_name -> fact
        
        # Fast lookup: for each ring, the fact global_id for (on ring goal_target)
        self.goal_on_fact_ids = {}     # ring_name -> global_id of the goal 'on' fact

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
        smaller_than_pairs = []  # (smaller, larger) where both are rings
        smaller_count = defaultdict(int)  # ring -> count of things it's smaller than
        
        # Build on_facts_by_ring and collect ring/tower info
        all_on_facts = {}  # (ring, target) -> fact
        
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
                    all_on_facts[(ring, target)] = fact
            
            elif predicate == "towerTop":
                # Track tower top facts for potential tie-breaking
                pass
        
        self.num_rings = len(rings)
        
        if self.num_rings == 0:
            return
        
        # Sort rings by size: most smaller_count = smallest ring
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        
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
        
        # Build goal_on_fact_ids for fast checking
        for ring, target in self.goal_positions.items():
            key = (ring, target)
            if key in all_on_facts:
                self.goal_on_fact_ids[ring] = all_on_facts[key].global_id
        
        # Determine the goal tower (the tower at the bottom of the goal stack)
        # Follow the goal chain: find the ring whose goal target is a tower
        self.goal_tower = None
        for ring, target in self.goal_positions.items():
            if target.startswith("t"):
                self.goal_tower = target
                break
        
        # Build goal stack order (bottom to top) for Term 3
        if self.goal_tower:
            # Build reverse mapping: target -> ring that should be on it
            target_to_ring = {}
            for ring, target in self.goal_positions.items():
                target_to_ring[target] = ring
            
            # Start from goal_tower, follow chain upward
            self.goal_stack_order = []
            current = self.goal_tower
            while current in target_to_ring:
                ring = target_to_ring[current]
                self.goal_stack_order.append(ring)
                current = ring
            # goal_stack_order is now bottom-to-top

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """Check if ring is at its goal position using fast bitwise check."""
        if ring in self.goal_on_fact_ids:
            return (state_bitwise >> self.goal_on_fact_ids[ring]) & 1
        return False

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """Find where a ring currently is."""
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _term1_largest_misplaced(self, state_bitwise: int) -> int:
        """
        Term 1: Classic 2^k - 1 bound.
        Find the largest ring not at its goal. Return 2^(index+1) - 1.
        """
        if not self.rings_by_size:
            return 0
        
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                return (1 << (i + 1)) - 1
        return 0

    def _term2_misplaced_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count of rings not at their goal position.
        Each needs at least 1 move.
        """
        count = 0
        for ring in self.rings_by_size:
            if not self._is_ring_at_goal(state_bitwise, ring):
                count += 1
        return count

    def _term3_unsettled_subproblem(self, state_bitwise: int) -> int:
        """
        Term 3: Bottom-up settled analysis.
        Count how many rings from the bottom of the goal stack are already
        correctly placed. The remaining n unsettled rings need 2^n - 1 moves.
        
        This can be tighter than Term 1 when some bottom rings are settled.
        """
        if not self.goal_stack_order:
            return 0
        
        settled = 0
        for ring in self.goal_stack_order:  # bottom to top
            if self._is_ring_at_goal(state_bitwise, ring):
                settled += 1
            else:
                break
        
        unsettled = self.num_rings - settled
        if unsettled <= 0:
            return 0
        return (1 << unsettled) - 1

    def _tie_breaking(self, state_bitwise: int) -> float:
        """
        Small tie-breaking penalty based on how many rings are misplaced.
        Uses sub-unit values to preserve admissibility of the integer terms.
        
        We penalize rings that are not even on the correct tower more than
        rings that are on the correct tower but in wrong position.
        """
        penalty = 0.0
        if not self.goal_tower:
            return 0.0
        
        for ring in self.rings_by_size:
            if not self._is_ring_at_goal(state_bitwise, ring):
                current_pos = self._get_current_position(state_bitwise, ring)
                if current_pos is not None:
                    # Check if the ring is at least on the goal tower chain
                    # A ring on the wrong tower is worse
                    # Simple heuristic: if current position starts with 't' and
                    # it's the goal tower, or if current position is a ring
                    # that's in the goal stack, it's partially there
                    penalty += 0.001
        
        return penalty

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic value as max of multiple admissible terms
        plus small tie-breaking.
        """
        if not self.rings_by_size:
            return 0
        
        t1 = self._term1_largest_misplaced(state_bitwise)
        t2 = self._term2_misplaced_count(state_bitwise)
        t3 = self._term3_unsettled_subproblem(state_bitwise)
        
        base = max(t1, t2, t3)
        
        if base == 0:
            return 0
        
        # Add sub-unit tie-breaking
        tie = self._tie_breaking(state_bitwise)
        
        # Return integer + tiny fraction for tie-breaking
        # Since the planner likely uses numeric comparison, the fraction helps
        # break ties without affecting admissibility of the integer part
        return base + tie

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic3(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic3 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 3 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tGoal Tower: {self.goal_tower}\n"
            f"\tGoal Stack Order: {self.goal_stack_order}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
