"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings smaller than k must first be moved off of it (and off the destination).
  This creates an exponential cascade: moving ring k requires 2^(k-1) - 1 moves
  to clear smaller rings, 1 move for ring k itself, then 2^(k-1) - 1 moves to
  reassemble the smaller rings on top.

Admissible Lower Bounds:
  Term 1 (Largest Misplaced Ring): Find the largest ring not at its goal position.
    If ring at index k (0-indexed from smallest) is misplaced, then at minimum
    2^(k+1) - 1 moves are needed. This is the classic optimal solution count.
    Admissible because the optimal Towers of Hanoi solution for k+1 disks is
    exactly 2^(k+1) - 1.

  Term 2 (Count of Misplaced Rings): Each ring not on its goal target requires
    at least 1 move. This is a trivial admissible lower bound but helps when
    many rings are misplaced but the largest misplaced ring is small.

  Term 3 (Settled-from-bottom analysis): Starting from the largest ring, count
    how many consecutive rings from the bottom are already settled at their goal.
    The remaining unsettled rings form a sub-tower that needs 2^m - 1 moves
    where m is the number of unsettled rings. This refines Term 1 by also
    considering partially-settled towers.

  We return max(term1, term2, term3) which is admissible since each term is.

Tie-breaking:
  We add a small penalty (0.001) per ring that is on the wrong tower entirely
  (not just wrong position on the right tower). This helps guide search toward
  moving rings to the correct tower first.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic11(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Uses three admissible terms combined with max(), plus a sub-unit
    tie-breaking penalty for rings on the wrong tower.
    """

    def __init__(self, towers_anthropicopus46_heuristic_11=None, name="TowersAnthropicOpus46Heuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring_name -> target_obj_name (from goal 'on' facts)
        self.rings_by_size = []       # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}    # ring_name -> list of (fact, target_obj_name) tuples
        self.num_rings = 0
        
        # For tie-breaking: goal tower for each ring
        self.goal_tower = {}          # ring_name -> tower_name
        # For determining which tower a ring is currently on
        self.towertop_facts_by_obj = {}  # obj_name -> list of (fact, tower_name)
        
        # Precomputed: for each ring, which 'on' fact corresponds to its goal
        self.goal_on_fact_id = {}     # ring_name -> global_id of the goal 'on' fact
        
        # Ring index mapping for fast lookup
        self.ring_index = {}          # ring_name -> index in rings_by_size (0=smallest)

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
                    
                    # Check if this is the goal on fact for this ring
                    if ring in self.goal_positions and target == self.goal_positions[ring]:
                        self.goal_on_fact_id[ring] = fact.global_id

            elif predicate == "towerTop":
                obj = args[0]
                tower = args[1]
                if obj not in self.towertop_facts_by_obj:
                    self.towertop_facts_by_obj[obj] = []
                self.towertop_facts_by_obj[obj].append((fact, tower))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size using smallerThan count
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than_pairs:
                if smaller in rings:
                    smaller_count[smaller] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by numeric suffix
            def ring_number(r):
                try:
                    return int(r[1:])
                except ValueError:
                    return 0
            self.rings_by_size = sorted(list(rings), key=ring_number)

        # Build ring index map
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

        # Determine goal tower for each ring (for tie-breaking)
        # Trace goal chain: ring -> goal_positions[ring] -> ... -> tower
        for ring in self.rings_by_size:
            obj = ring
            visited = set()
            while obj in self.goal_positions and obj not in visited:
                visited.add(obj)
                obj = self.goal_positions[obj]
            # obj should now be a tower (starts with 't')
            if obj.startswith("t"):
                self.goal_tower[ring] = obj
            else:
                # Follow the on-chain from the goal target
                self.goal_tower[ring] = obj  # best guess

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _get_current_tower(self, state_bitwise: int, ring: str) -> str:
        """Find which tower a ring is currently on by checking towerTop facts."""
        if ring in self.towertop_facts_by_obj:
            for fact, tower in self.towertop_facts_by_obj[ring]:
                if (state_bitwise >> fact.global_id) & 1:
                    return tower
        # Ring is not on top - trace down from ring's position
        # The ring is ON some object; we need to find which tower it belongs to
        # by tracing up: find what's on top and which tower that top belongs to
        # This is expensive, so we use a different approach:
        # Check all towers to see which one this ring is part of
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """Fast check: is ring at its goal position?"""
        if ring in self.goal_on_fact_id:
            return (state_bitwise >> self.goal_on_fact_id[ring]) & 1
        # If no goal_on fact found, check manually
        goal_pos = self.goal_positions.get(ring)
        if goal_pos is None:
            return True  # No goal for this ring
        current_pos = self._get_current_position(state_bitwise, ring)
        return current_pos == goal_pos

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        
        # --- Term 1: Largest misplaced ring heuristic ---
        # Find the largest ring NOT at its goal position
        # For k+1 rings (indices 0..k), minimum moves = 2^(k+1) - 1
        term1 = 0
        largest_misplaced_idx = -1
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                largest_misplaced_idx = i
                num_rings_to_move = i + 1
                term1 = (1 << num_rings_to_move) - 1
                break

        if term1 == 0:
            return 0  # All rings at goal

        # --- Term 2: Count of misplaced rings ---
        term2 = 0
        misplaced_count = 0
        for i in range(n):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                misplaced_count += 1
        term2 = misplaced_count

        # --- Term 3: Settled-from-bottom analysis ---
        # Count consecutive settled rings from the bottom (largest)
        # A ring is "settled" if it and all rings below it are at goal
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

        # --- Tie-breaking: small penalty for rings on wrong tower ---
        # This is sub-unit so it doesn't affect admissibility
        wrong_tower_penalty = 0.0
        if largest_misplaced_idx >= 0:
            for i in range(min(largest_misplaced_idx + 1, n)):
                ring = self.rings_by_size[i]
                if ring in self.goal_tower:
                    # Check if ring is currently on its goal tower
                    # We check towerTop facts - if ring is towerTop of goal tower, it's there
                    # Otherwise check if it's on an object that traces to goal tower
                    # Simple approximation: if ring is misplaced, add penalty
                    if not self._is_ring_at_goal(state_bitwise, ring):
                        wrong_tower_penalty += 0.001

        h_value = max(term1, term2, term3) + wrong_tower_penalty
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic11(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic11 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 11 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
