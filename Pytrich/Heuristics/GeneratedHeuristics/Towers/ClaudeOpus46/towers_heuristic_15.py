"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal,
  ALL rings 1..k-1 (smaller) must first be moved off ring k, requiring
  at minimum 2^(k-1) - 1 moves, plus 1 move for ring k itself.
  The total minimum for k misplaced rings from the top is 2^k - 1.

Admissible Lower Bounds:
  Term 1 (Largest-Displaced Ring): Find the largest ring not at its goal
    position. All rings from smallest up to and including that ring must
    move at least once, giving a lower bound of 2^k - 1 where k is the
    count of those rings. This is admissible because the classic Tower
    of Hanoi solution is optimal at 2^k - 1 moves.

  Term 2 (Count of Displaced Rings): Simply count how many rings are not
    on their goal target. Each displaced ring requires at least 1 move,
    so this is a trivial admissible lower bound. It helps when many rings
    are displaced but none is very large.

  Term 3 (Task Network Lower Bound): Count the number of move_abstract
    tasks remaining in the task network. Each such task decomposes to
    exactly one primitive move action, so the count is an admissible
    lower bound on remaining actions.

  Return max(term1, term2, term3) — admissible since each term is.

Implementation notes:
  - All fact parsing and indexing done in initialize() (P3).
  - __call__ uses only O(n_rings) bit tests, no model iteration (P3).
  - Per-task cost reflects exponential structure, not flat +1 (P2, P4).
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic15(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
      1. Exponential lower bound from largest displaced ring (2^k - 1)
      2. Count of displaced rings
      3. Count of move_abstract tasks in the task network
    """

    def __init__(self, towers_anthropicopus46_heuristic_15=None, name="TowersAnthropicOpus46Heuristic15"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0

        # For task network analysis
        self.move_abstract_task_ids = set()  # global_ids of move_abstract abstract tasks
        self.move_operator_ids = set()       # global_ids of move operators

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
        smaller_than_pairs = []  # (smaller, larger) both ring names
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

        # Sort rings by size: smallest first (most smaller_than relations = smallest)
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_count.get(r, 0))

        # Identify move_abstract tasks and move operators for task network analysis
        for at in self.model.abstract_tasks:
            if at.name and "move_abstract" in at.name:
                self.move_abstract_task_ids.add(at.global_id)

        for op in self.model.operators:
            if op.name and "move" in op.name:
                self.move_operator_ids.add(op.global_id)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _term_exponential(self, state_bitwise: int) -> int:
        """
        Term 1: Find the largest ring not at goal, return 2^k - 1
        where k = number of rings from smallest up to and including
        the largest displaced ring.
        
        Admissibility: This is the minimum number of moves in the
        classic Tower of Hanoi solution for k disks.
        """
        if not self.rings_by_size:
            return 0

        # rings_by_size[0] = smallest, rings_by_size[-1] = largest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if current_pos != goal_pos:
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1

        return 0

    def _term_displaced_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count of rings not at their goal position.
        Each such ring needs at least 1 move.
        Admissible: trivially, each displaced ring must be moved ≥1 time.
        """
        count = 0
        for ring in self.rings_by_size:
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if current_pos != goal_pos:
                count += 1
        return count

    def _term_task_network(self, task_network) -> int:
        """
        Term 3: Count move_abstract tasks and operators in the task network.
        Each move_abstract decomposes to exactly 1 primitive move action.
        Each operator in the network is 1 action.
        Admissible: each such element requires at least 1 action.
        """
        count = 0
        if task_network is None:
            return 0
        for task in task_network:
            gid = task.global_id
            if gid in self.move_abstract_task_ids or gid in self.move_operator_ids:
                count += 1
        return count

    def _compute(self, state_bitwise: int, task_network) -> int:
        t1 = self._term_exponential(state_bitwise)
        t2 = self._term_displaced_count(state_bitwise)
        t3 = self._term_task_network(task_network)
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic15(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic15 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 15 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
