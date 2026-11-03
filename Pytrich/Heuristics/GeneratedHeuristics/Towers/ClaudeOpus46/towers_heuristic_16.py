"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal,
  ALL rings 1..k-1 (smaller) must first be moved off ring k, requiring
  at minimum 2^(k-1) - 1 moves just for the clearing, plus 1 move for
  ring k itself. This recurses, giving the classic 2^k - 1 formula.

Admissible Lower Bounds:
  Term 1 (Largest-misplaced exponential): Find the largest ring not at
    its goal position. If it is ring index k (0-indexed from smallest),
    then at least 2^k - 1 moves are needed. This is admissible because
    the classic Towers of Hanoi optimal solution requires exactly this.
    
  Term 2 (Count of misplaced rings): Simply count rings not on their
    goal target. Each misplaced ring needs at least 1 move. This is a
    trivial admissible bound but helps when many small rings are displaced.

  Term 3 (Settled-from-bottom analysis): Starting from the largest ring,
    count how many consecutive rings (bottom-up) are already settled in
    their goal positions. If s rings are settled, only n-s need moving,
    giving a tighter 2^(n-s) - 1 bound. This refines Term 1 by not
    over-counting already-placed bottom rings.

The heuristic returns max(term1, term2, term3).

Implementation uses O(1) bitwise state checks per ring in __call__.
All parsing and preprocessing is done in initialize().
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic16(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Combines:
    1. Largest-misplaced-ring exponential bound
    2. Count of misplaced rings (linear bound)
    3. Settled-from-bottom exponential bound
    
    Returns max of all three for a tight admissible estimate.
    """

    def __init__(self, towers_anthropicopus46_heuristic_16=None, name="TowersAnthropicOpus46Heuristic16"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring_name -> target_obj_name
        self.rings_by_size = []       # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}    # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For fast goal checking: ring_name -> (goal_fact_global_id or None)
        # If we can find the specific on[ring, goal_target] fact, we store its global_id
        self.goal_on_fact_id = {}     # ring_name -> global_id of the on[ring, goal_target] fact
        
        # For settled-from-bottom analysis
        # rings_by_size[-1] is the largest ring, rings_by_size[0] is smallest

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
        smaller_than_pairs = []  # (smaller, larger) where both are ring names
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

        # Build fact index maps
        # on_facts_by_ring: for finding current position of each ring
        # Also build a direct lookup: for each ring, the global_id of on[ring, goal_target]
        on_fact_map = {}  # (ring, target) -> fact
        
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
                    on_fact_map[(ring, target)] = fact

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size: smallest first (most things it's smaller than)
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

        # Build goal_on_fact_id for O(1) goal checking per ring
        for ring, goal_target in self.goal_positions.items():
            key = (ring, goal_target)
            if key in on_fact_map:
                self.goal_on_fact_id[ring] = on_fact_map[key].global_id

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """O(1) check if ring is at its goal position using bitwise test."""
        if ring not in self.goal_on_fact_id:
            # No goal for this ring, or goal fact not found - treat as satisfied
            return True
        fact_id = self.goal_on_fact_id[ring]
        return (state_bitwise >> fact_id) & 1

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        
        # --- Term 1: Largest misplaced ring exponential bound ---
        # Find largest ring (by index) not at goal
        term1 = 0
        largest_misplaced_idx = -1
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                largest_misplaced_idx = i
                break
        
        if largest_misplaced_idx >= 0:
            # All rings from 0 to largest_misplaced_idx need to move
            term1 = (1 << (largest_misplaced_idx + 1)) - 1
        
        # --- Term 2: Count of misplaced rings ---
        term2 = 0
        for i in range(n):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                term2 += 1
        
        # --- Term 3: Settled-from-bottom analysis ---
        # Count consecutive settled rings from bottom (largest) up
        settled = 0
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if self._is_ring_at_goal(state_bitwise, ring):
                settled += 1
            else:
                break
        
        unsettled = n - settled
        term3 = (1 << unsettled) - 1 if unsettled > 0 else 0
        
        # Return max of all terms
        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic16(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic16 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 16 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
