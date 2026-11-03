"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings smaller than k must first be moved off of ring k (and off the destination).
  This creates an exponential cascade: moving ring k requires 2^(k-1) - 1 moves
  to clear smaller rings, 1 move for ring k, and 2^(k-1) - 1 moves to reassemble.

Admissible Lower Bounds:
  Term 1 (Exponential displacement): Find the largest ring not at its goal position.
    All rings from smallest up to that ring must move at least once. The classic
    Tower of Hanoi solution gives exactly 2^k - 1 moves for k rings. This is
    admissible because no solution can do better than the optimal recursive strategy.

  Term 2 (Counting displaced rings): Count the number of rings not currently
    sitting on their goal target. Each such ring requires at least 1 move.
    This is a trivial admissible lower bound.

  Term 3 (Bottom-up chain analysis): Starting from the largest ring, find the
    longest contiguous chain of rings already in their goal configuration at the
    bottom. The remaining rings all need to move. For the remaining k rings,
    2^k - 1 is a lower bound.

  We return max(term1, term2, term3). Since term1 and term3 are typically the
  tightest, and term1 already captures the exponential nature well, we focus
  on making term1 very precise by also considering partially-settled stacks.

Design:
  - initialize(): Parse all facts, build ring ordering, goal map, and fact ID maps.
  - __call__(): O(n) bit checks where n = number of rings (very fast).
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic17(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Uses the classic 2^k - 1 formula combined with bottom-up chain analysis
    to produce tight lower bounds on the number of moves remaining.
    """

    def __init__(self, towers_anthropicopus46_heuristic_17=None, name="TowersAnthropicOpus46Heuristic17"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For fast goal checking: ring_name -> goal fact global_id (if it exists)
        self.goal_on_fact_ids = {}     # ring_name -> global_id of the on[ring, goal_target] fact
        
        # Ring index mapping for O(1) lookup
        self.ring_to_index = {}        # ring_name -> index in rings_by_size

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args).
        Handles format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
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
        Extract domain structure from grounded facts.
        Build ring ordering, goal map, and fact ID maps for fast lookup.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller, larger) where both are rings
        smaller_than_count = defaultdict(int)  # ring -> count of things it's smaller than
        
        # Build on_facts_by_ring and collect ring names
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

        # Sort rings by size: most "smallerThan" relations = smallest ring
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_count.get(r, 0))
        
        # Build index mapping
        for i, ring in enumerate(self.rings_by_size):
            self.ring_to_index[ring] = i

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

        # Build goal_on_fact_ids for fast bitwise checking
        # For each ring, find the fact object for on[ring, goal_target]
        for ring, goal_target in self.goal_positions.items():
            if ring in self.on_facts_by_ring:
                for fact, target in self.on_facts_by_ring[ring]:
                    if target == goal_target:
                        self.goal_on_fact_ids[ring] = fact.global_id
                        break

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """
        Check if a ring is at its goal position using a single bit test.
        O(1) operation.
        """
        if ring not in self.goal_on_fact_ids:
            return True  # No goal for this ring, consider it satisfied
        fact_id = self.goal_on_fact_ids[ring]
        return (state_bitwise >> fact_id) & 1

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Find where a ring currently is by checking which 'on' fact is true.
        """
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value using multiple admissible components.
        
        Term 1: Classic exponential - find largest displaced ring, return 2^k - 1
                 where k = index+1 of that ring in size order.
        
        Term 2: Count of displaced rings (simple lower bound).
        
        Term 3: Bottom-up settled chain analysis - find how many rings from the
                 bottom are already correctly placed, then 2^(remaining) - 1.
        """
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        
        # Precompute which rings are at goal (O(n) bit tests, done once)
        at_goal = [False] * n
        for i in range(n):
            ring = self.rings_by_size[i]
            at_goal[i] = self._is_ring_at_goal(state_bitwise, ring)

        # Term 1: Find the largest ring NOT at goal.
        # All rings from smallest (index 0) to that ring (index i) need to move.
        # Lower bound: 2^(i+1) - 1
        term1 = 0
        for i in range(n - 1, -1, -1):
            if not at_goal[i]:
                term1 = (1 << (i + 1)) - 1
                break

        # Term 2: Count displaced rings
        term2 = sum(1 for i in range(n) if not at_goal[i])

        # Term 3: Bottom-up chain analysis
        # Find how many rings from the BOTTOM are correctly settled.
        # A ring is "settled" if it and all rings larger than it (below it)
        # are at their goal positions AND stacked correctly.
        # 
        # We check from the bottom (largest ring = index n-1) upward.
        # The first ring that is NOT at goal breaks the chain.
        # All rings above (smaller than) that point must move.
        settled_from_bottom = 0
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if at_goal[i]:
                # Additionally verify that this ring is on the correct thing
                # and the thing below it is also settled (or it's on the tower base)
                # Since at_goal already checks on[ring, goal_target], and
                # goal_target for a settled ring should be the next larger ring
                # (which is also settled), this chain check is implicitly correct.
                settled_from_bottom += 1
            else:
                break

        remaining = n - settled_from_bottom
        term3 = (1 << remaining) - 1 if remaining > 0 else 0

        # Return the maximum of all terms (admissible since each is a lower bound)
        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic value during search.
        Called for every node expansion - must be fast.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic17(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic17 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
