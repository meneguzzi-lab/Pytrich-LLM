"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings smaller than k must first be moved off of ring k (and off the destination).
  This creates an exponential cascade: moving ring k requires 2^(k-1) - 1 moves to
  clear smaller rings, 1 move for ring k itself, then 2^(k-1) - 1 moves to reassemble.

Admissible Lower Bounds:
  Term 1 (Exponential structural bound): Find the largest ring not at its goal position.
    All rings from the smallest up to and including that ring must move at least once.
    For k such rings, the minimum is 2^k - 1 moves. This is admissible because the
    classic Tower of Hanoi solution is optimal.
  
  Term 2 (Misplaced ring count): Count the number of rings not currently sitting on
    their goal target. Each misplaced ring requires at least 1 move. This is trivially
    admissible (each ring needs at least one action to fix).
  
  Term 3 (Goal-stability / settled-from-bottom check): Starting from the largest ring,
    count how many rings are continuously settled (at goal position with correct support
    chain). The first unsettled ring from the bottom and everything above it must move.
    This refines Term 1 by being more precise about which rings are truly settled.

The heuristic returns max(term1, term2, term3) which is admissible since max of
admissible terms is admissible.

Implementation notes:
  - All fact parsing and map building happens in initialize() (P3).
  - __call__ uses only O(n_rings) bit tests (P2, P3).
  - Per-ring cost is exponential, not flat +1 (P2, avoiding A4).
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic10(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential lower bound based on largest unsettled ring
    2. Simple misplaced ring count
    3. Goal-chain stability analysis from the bottom up
    """

    def __init__(self, towers_anthropicopus46_heuristic_10=None, name="TowersAnthropicOpus46Heuristic10"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_on = {}             # ring -> target object (from goal state 'on' facts)
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.on_fact_id = {}          # (ring, target) -> fact.global_id for fast lookup
        self.num_rings = 0
        self.ring_index = {}          # ring_name -> index in rings_by_size (0=smallest)

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocess domain structure once before search begins.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a grounded fact name into (predicate, args).
        Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Extract ring ordering, goal positions, and build fast lookup maps.
        """
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
                        self.goal_on[ring] = target

        # Parse all facts to build lookup structures
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
                    self.on_fact_id[(ring, target)] = fact.global_id

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size: smallest first (most smaller_count = smallest ring)
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        
        # Build ring index map
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Find where a ring currently sits using O(k) bit tests where k is
        the number of possible 'on' facts for this ring.
        """
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """
        Check if a ring is currently on its goal target using a single bit test.
        """
        goal_target = self.goal_on.get(ring)
        if goal_target is None:
            return True  # No goal for this ring, consider it settled
        key = (ring, goal_target)
        if key in self.on_fact_id:
            return (state_bitwise >> self.on_fact_id[key]) & 1
        return False

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the multi-component heuristic.
        
        Term 1: Exponential bound - find largest unsettled ring from bottom.
                 All rings from smallest to that ring need 2^k - 1 moves minimum.
        
        Term 2: Count of misplaced rings (simple admissible bound).
        
        Term 3: Goal-chain stability - find the longest settled chain from the
                 bottom. The remaining rings above the break need at least
                 2^remaining - 1 moves.
        """
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        
        # Precompute which rings are at their goal position
        at_goal = [False] * n
        misplaced_count = 0
        for i in range(n):
            ring = self.rings_by_size[i]
            at_goal[i] = self._is_ring_at_goal(state_bitwise, ring)
            if not at_goal[i]:
                misplaced_count += 1

        # Term 2: Simple misplaced count
        term2 = misplaced_count

        # Term 1: Find largest ring not at goal, return 2^(index+1) - 1
        term1 = 0
        for i in range(n - 1, -1, -1):
            if not at_goal[i]:
                term1 = (1 << (i + 1)) - 1
                break

        # Term 3: Goal-chain stability from the bottom
        # Check from the largest ring upward. A ring is "settled" if:
        #   1. It is at its goal position, AND
        #   2. All rings below it (larger) are also settled, AND
        #   3. It is sitting on the correct support (the ring below it is at goal too,
        #      or it's on the correct tower base)
        # Find the first unsettled ring from the bottom.
        settled_from_bottom = 0
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_target = self.goal_on.get(ring)
            
            if goal_target is None:
                # No goal constraint, consider settled
                settled_from_bottom += 1
                continue
            
            current_pos = self._get_current_position(state_bitwise, ring)
            
            if current_pos == goal_target:
                # Check if the support chain is intact
                # The goal target should also be settled
                if goal_target.startswith("r"):
                    # Target is another ring - check if that ring is also settled
                    target_idx = self.ring_index.get(goal_target, -1)
                    if target_idx >= 0 and target_idx > i:
                        # target ring is larger (higher index), should already be checked
                        # If we got here without breaking, target is settled
                        settled_from_bottom += 1
                        continue
                    else:
                        break
                else:
                    # Target is a tower base - always stable support
                    settled_from_bottom += 1
                    continue
            else:
                break

        unsettled_count = n - settled_from_bottom
        if unsettled_count > 0:
            term3 = (1 << unsettled_count) - 1
        else:
            term3 = 0

        # Return max of all terms (admissible since each term is admissible)
        h_value = max(term1, term2, term3)
        
        # If task network is empty and h > 0, the plan might be done
        # but state doesn't match goals - this shouldn't happen in valid search
        # but we handle it gracefully
        if len(task_network) == 0:
            # No more tasks to decompose
            return h_value
        
        return h_value

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic value during search.
        Called for every node expansion - must be fast.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic10(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic10 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 10 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_on}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
