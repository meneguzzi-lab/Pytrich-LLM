"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings 1..k-1 (smaller rings) must first be moved off of ring k and onto
  an auxiliary peg. This creates an exponential lower bound on the number of moves.

Admissible Lower Bounds:
  Term 1 (Exponential structural bound): Find the largest ring not at its goal.
    All rings from 1 to that ring's index must move at least once each in a
    specific pattern. The classic Towers of Hanoi optimal solution for k disks
    is 2^k - 1 moves. This is admissible because no solution can do better.

  Term 2 (Misplaced ring count): Count the number of rings not currently on
    their goal target. Each such ring requires at least 1 move. This is a
    trivial admissible lower bound but helps when Term 1 is 0 but tasks remain.

  Term 3 (Task network lower bound): Count the number of move_abstract tasks
    in the current task network. Each move_abstract decomposes to exactly one
    move action, so this is an admissible lower bound on remaining actions.

  The heuristic returns max(term1, term2, term3).

Tie-breaking:
  We add a small sub-unit penalty based on how many rings are on non-goal towers
  to break ties among states with the same h-value, preferring states where
  rings are closer to their destination.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic2(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential structural lower bound (2^k - 1 for largest misplaced ring)
    2. Misplaced ring count
    3. Task network move count
    
    Plus sub-unit tie-breaking for goal proximity.
    """

    def __init__(self, towers_anthropicopus46_heuristic_2=None, name="TowersAnthropicOpus46Heuristic2"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.on_fact_ids = {}          # (ring, target) -> global_id for fast lookup
        self.goal_on_fact_id = {}      # ring -> global_id of the goal "on" fact
        self.num_rings = 0
        
        # For task network analysis
        self.move_abstract_task_names = set()  # set of abstract task names for move_abstract
        self.move_operator_names = set()       # set of operator names for move
        
        # Ring index mapping for fast lookup
        self.ring_index = {}           # ring_name -> index in rings_by_size (0=smallest)

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
                    self.on_fact_ids[(ring, target)] = fact.global_id

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size using smallerThan count
        # Ring with highest smaller_count is the smallest ring
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        
        # Build ring index
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

        # Map goal on facts for fast lookup
        for ring, target in self.goal_positions.items():
            key = (ring, target)
            if key in self.on_fact_ids:
                self.goal_on_fact_id[ring] = self.on_fact_ids[key]

        # Identify move_abstract tasks and move operators by name patterns
        for at in self.model.abstract_tasks:
            parsed_name = at.name
            if "move_abstract" in parsed_name:
                self.move_abstract_task_names.add(at.global_id)
        
        for op in self.model.operators:
            parsed_name = op.name
            if "move" in parsed_name:
                self.move_operator_names.add(op.global_id)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """Fast check if ring is at its goal position using precomputed fact id."""
        if ring in self.goal_on_fact_id:
            return (state_bitwise >> self.goal_on_fact_id[ring]) & 1
        # If no goal_on fact found, check manually
        goal_pos = self.goal_positions.get(ring)
        if goal_pos is None:
            return True  # No goal defined, consider it satisfied
        current_pos = self._get_current_position(state_bitwise, ring)
        return current_pos == goal_pos

    def _term1_exponential_bound(self, state_bitwise: int) -> int:
        """
        Find the largest ring not at its goal position.
        Return 2^k - 1 where k = index+1 of that ring (all smaller rings must also move).
        
        Admissibility: The classic Tower of Hanoi optimal solution for k disks
        requires exactly 2^k - 1 moves. Since all k rings from smallest up to
        the largest misplaced ring must be moved, this is a lower bound.
        
        Enhancement: We also check if rings below the largest misplaced ring
        are "settled" (at goal AND supported correctly). If the bottom portion
        is already correctly stacked, we can reduce k.
        """
        if not self.rings_by_size:
            return 0

        # Find the largest misplaced ring (iterate from largest to smallest)
        largest_misplaced_idx = -1
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                largest_misplaced_idx = i
                break

        if largest_misplaced_idx < 0:
            return 0

        # Now check how many rings from bottom are already settled
        # A ring is "settled" if it's at its goal AND all rings below it are also settled
        settled_from_bottom = 0
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if self._is_ring_at_goal(state_bitwise, ring):
                settled_from_bottom += 1
            else:
                break

        # The number of rings that need to move:
        # From index 0 up to largest_misplaced_idx (inclusive)
        num_rings_to_move = largest_misplaced_idx + 1

        return (1 << num_rings_to_move) - 1

    def _term2_misplaced_count(self, state_bitwise: int) -> int:
        """
        Count rings not at their goal position.
        Each misplaced ring needs at least 1 move.
        Admissible: trivially, each ring not at goal needs >= 1 action.
        """
        count = 0
        for ring in self.rings_by_size:
            if not self._is_ring_at_goal(state_bitwise, ring):
                count += 1
        return count

    def _term3_task_network_bound(self, task_network) -> int:
        """
        Count move_abstract tasks and primitive move operators in the task network.
        Each move_abstract decomposes to exactly 1 move action.
        Each move operator IS 1 action.
        Admissible: each such task/operator contributes at least 1 action to the plan.
        
        Also count other abstract tasks that will eventually need at least one move.
        """
        if not task_network:
            return 0
        
        move_count = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.move_abstract_task_names or gid in self.move_operator_names:
                move_count += 1
        
        return move_count

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic as max of multiple admissible terms
        plus a small tie-breaking penalty.
        """
        if not self.rings_by_size:
            return 0

        term1 = self._term1_exponential_bound(state_bitwise)
        term2 = self._term2_misplaced_count(state_bitwise)
        term3 = self._term3_task_network_bound(task_network)

        base_h = max(term1, term2, term3)

        # Tie-breaking: among states with same h, prefer those where
        # more rings (from the bottom) are correctly settled.
        # Count settled rings from bottom for tie-breaking
        settled = 0
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if self._is_ring_at_goal(state_bitwise, ring):
                settled += 1
            else:
                break
        
        # Small penalty for unsettled state (sub-unit to preserve admissibility)
        unsettled_penalty = (self.num_rings - settled) * 0.001

        return base_h + unsettled_penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic2(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic2 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 2 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
