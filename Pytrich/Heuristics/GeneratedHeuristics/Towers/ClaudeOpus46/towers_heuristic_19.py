"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings smaller than k must first be moved off of ring k (and off the destination).
  This creates an exponential lower bound: moving n misplaced rings requires at least
  2^n - 1 moves in the classic 3-peg Tower of Hanoi.

Admissible Lower Bounds:
  Term 1 (Exponential displacement): Find the largest ring not at its goal position.
    All rings from the smallest up to (and including) that ring must move at least once.
    The classic Towers of Hanoi optimal solution for k disks is 2^k - 1 moves.
    This is admissible because no solution can do better than the classic optimum.

  Term 2 (Simple displacement count): Count the number of rings not currently
    sitting on their goal target. Each such ring requires at least 1 move.
    This is admissible (trivially: each misplaced ring needs >= 1 action).

  Term 3 (Task network lower bound): Count the number of move_abstract / primitive
    tasks remaining in the task network. Each such task corresponds to at least 1 action.
    Admissible because each primitive task decomposes to exactly 1 operator.

  We return max(term1, term2, term3) which is admissible since max of admissible terms
  is admissible.

Additional refinement:
  For Term 1, we also check "settled" status: a ring is only considered settled if
  it is on its goal target AND all rings below it (larger rings) are also settled.
  This catches cases where a ring appears to be at its goal but will need to move
  because a ring beneath it is misplaced.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic19(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential displacement bound (2^k - 1)
    2. Simple misplaced ring count
    3. Task network primitive count lower bound
    """

    def __init__(self, towers_anthropicopus46_heuristic_19=None, name="TowersAnthropicOpus46Heuristic19"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For fast operator/abstract task identification in task network
        self.move_abstract_task_ids = set()  # global_ids of move_abstract abstract tasks
        self.operator_global_ids = set()     # global_ids of all operators (primitive tasks)

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

        # Sort rings smallest to largest using smallerThan count
        # The smallest ring is smaller than the most things
        if smaller_count:
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: use ring number
            self.rings_by_size = sorted(list(rings), key=lambda r: int(r[1:]))

        # Identify move_abstract tasks and operators for task network counting
        for op in self.model.operators:
            self.operator_global_ids.add(op.global_id)
        
        for at in self.model.abstract_tasks:
            if at.name and "move_abstract" in at.name:
                self.move_abstract_task_ids.add(at.global_id)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _term_exponential_displacement(self, state_bitwise: int) -> int:
        """
        Term 1: Find the largest ring that is NOT settled at its goal.
        A ring is "settled" only if it AND all larger rings below it are at goal.
        Return 2^k - 1 where k = number of rings from smallest to that largest unsettled ring.
        """
        if not self.rings_by_size:
            return 0

        # Check from largest to smallest ring for settled status
        # rings_by_size: index 0 = smallest, index n-1 = largest
        settled = [False] * len(self.rings_by_size)
        
        # Start from the largest ring
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            
            if current_pos == goal_pos:
                if i == len(self.rings_by_size) - 1:
                    # Largest ring: settled if at goal
                    settled[i] = True
                else:
                    # Settled if at goal AND ring below (larger ring) is also settled
                    settled[i] = settled[i + 1]
            else:
                settled[i] = False

        # Find the largest unsettled ring
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            if not settled[i]:
                # Rings 0..i all need to move
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1

        return 0

    def _term_displacement_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count the number of rings not at their goal position.
        Each needs at least 1 move. Admissible.
        """
        count = 0
        for ring in self.rings_by_size:
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if current_pos != goal_pos:
                count += 1
        return count

    def _term_task_network_lb(self, task_network) -> int:
        """
        Term 3: Count primitive tasks (operators) and move_abstract tasks in the
        task network. Each move_abstract decomposes to exactly 1 operator.
        Each operator is 1 action. So this is a lower bound on remaining actions.
        """
        count = 0
        if task_network is None:
            return 0
        for task in task_network:
            gid = task.global_id
            if gid in self.operator_global_ids:
                count += 1
            elif gid in self.move_abstract_task_ids:
                count += 1
        return count

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic as max of three admissible terms.
        """
        t1 = self._term_exponential_displacement(state_bitwise)
        t2 = self._term_displacement_count(state_bitwise)
        t3 = self._term_task_network_lb(task_network)
        
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic19(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic19 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 19 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
