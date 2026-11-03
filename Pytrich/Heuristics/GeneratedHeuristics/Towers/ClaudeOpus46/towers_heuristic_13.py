"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single serializing constraint is that to move ring k to its goal position,
  ALL rings smaller than k must first be moved off of ring k (and off its destination).
  This creates an exponential lower bound: moving n misplaced rings requires at least
  2^n - 1 moves in the classic Tower of Hanoi.

Admissible Lower Bounds:
  Term 1 (Exponential structural bound): Find the largest ring not at its goal position.
    All rings from the smallest up to (and including) that ring must move at least once
    each in a specific order. The minimum number of moves is 2^k - 1 where k is the
    count of rings from smallest to that largest misplaced ring. This is admissible
    because it equals the optimal solution for the classical Towers of Hanoi subproblem.

  Term 2 (Misplaced ring count): Count the number of rings not currently on their goal
    target. Each misplaced ring requires at least 1 move. This is trivially admissible.
    This helps when all large rings are placed but small rings are scattered.

  Term 3 (Task network lower bound): Count the number of primitive tasks (move_abstract)
    and compound tasks that must decompose into at least one move. Each move_abstract
    task requires exactly one action. This is admissible as each such task needs >= 1 action.

  Final heuristic: max(term1, term2, term3)

  Additional refinement: We also check for "settled" rings from the bottom up.
  A ring is settled if it's at its goal AND all rings below it are also settled.
  Once we find the first unsettled ring from the bottom, all rings above it
  (including smaller ones that happen to be at goal) are considered unsettled
  because they'll need to move to allow the unsettled ring below to reach its goal.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic13(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential structural bound based on largest unsettled ring
    2. Count of misplaced rings
    3. Task network primitive action lower bound
    """

    def __init__(self, towers_anthropicopus46_heuristic_13=None, name="TowersAnthropicOpus46Heuristic13"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For fast goal checking
        self.goal_on_fact_ids = {}     # ring_name -> global_id of the goal 'on' fact
        
        # For task network analysis
        self.move_abstract_task_names = set()  # names of move_abstract abstract tasks
        self.operator_names = set()            # names of primitive move operators
        
        # Ring name to index mapping for fast settled-ring computation
        self.ring_to_idx = {}

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
        Build fast-lookup structures for heuristic computation.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller, larger) where both are rings
        smaller_than_count = defaultdict(int)  # ring -> how many things it's smaller than

        # Step 1: Extract goal positions from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # Step 2: Scan all facts to build lookup structures
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
                    
                    # Check if this is a goal fact
                    if ring in self.goal_positions and self.goal_positions[ring] == target:
                        self.goal_on_fact_ids[ring] = fact.global_id

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Step 3: Sort rings by size (smallest first)
        # Ring with highest smaller_than_count is smallest (it's smaller than the most things)
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_count.get(r, 0))
        
        # Build index mapping
        for idx, ring in enumerate(self.rings_by_size):
            self.ring_to_idx[ring] = idx

        # Step 4: Identify move_abstract and operator tasks for task network analysis
        for at in self.model.abstract_tasks:
            pred, args = self._parse_fact_name(at.name)
            if pred is None:
                # Try simpler parsing for task names like "move_abstract[t1,t2]"
                name = at.name.strip()
                if "move_abstract" in name:
                    self.move_abstract_task_names.add(at.name)
            else:
                if pred == "move_abstract":
                    self.move_abstract_task_names.add(at.name)
        
        for op in self.model.operators:
            self.operator_names.add(op.name)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """Find where a ring currently is by checking which 'on' fact is true."""
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """Fast check if ring is at its goal position using precomputed fact id."""
        if ring not in self.goal_on_fact_ids:
            # No goal for this ring, or goal fact not found - check manually
            if ring not in self.goal_positions:
                return True  # No goal means trivially satisfied
            return self._get_current_position(state_bitwise, ring) == self.goal_positions[ring]
        return (state_bitwise >> self.goal_on_fact_ids[ring]) & 1

    def _exponential_bound(self, state_bitwise: int) -> int:
        """
        Term 1: Exponential structural bound.
        
        Check rings from largest to smallest. Find the largest ring that is NOT
        "settled" (at goal with all rings below it also settled).
        
        For the settled analysis: a ring is settled if:
        - It is at its goal position, AND
        - All rings larger than it (below it in the goal configuration) are also settled
        
        We scan from largest to smallest. Once we find an unsettled ring,
        all rings from smallest to that ring need to move.
        Return 2^k - 1 where k = number of such rings.
        """
        if not self.rings_by_size:
            return 0

        # Scan from largest (bottom) to smallest (top)
        # rings_by_size is sorted smallest to largest, so reverse
        largest_unsettled_idx = -1
        settled = True
        
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if ring not in self.goal_positions:
                # Ring has no goal - skip but don't break settled chain
                continue
            
            if settled and self._is_ring_at_goal(state_bitwise, ring):
                # This ring is settled (at goal and all below it are settled)
                continue
            else:
                # This ring is not settled
                settled = False
                largest_unsettled_idx = i
                break
        
        if largest_unsettled_idx < 0:
            # Check remaining rings above the break point
            # All rings from bottom are settled, but maybe some above aren't
            # Actually if we broke out, we need to check if there are unsettled ones above
            # Let me redo: scan from bottom, first unsettled marks everything above as needing to move
            pass
        
        # Better approach: scan from largest to smallest
        # Track whether we've seen an unsettled ring
        first_unsettled_from_bottom = -1
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if ring not in self.goal_positions:
                continue
            if not self._is_ring_at_goal(state_bitwise, ring):
                first_unsettled_from_bottom = i
                break
            # Check if the ring below it is settled too
            # Actually, if ring i is at goal but ring i+1 below it isn't settled,
            # ring i will need to move too. But we're scanning from bottom up,
            # so if we haven't found unsettled yet, all below are settled.
        
        if first_unsettled_from_bottom < 0:
            # All rings are at goal
            return 0
        
        # Now we know the largest unsettled ring is at index first_unsettled_from_bottom
        # But we need to also check: are there rings BELOW it that are at goal
        # but will need to move because this ring needs to move through them?
        # Actually in Towers of Hanoi, rings below the unsettled one might already be settled.
        # The key insight: if ring at index i is unsettled, rings 0..i all need to move.
        # (rings 0..i are the smallest i+1 rings)
        
        # But wait - we also need to account for rings that ARE at goal but sit above
        # an unsettled ring. Those are "falsely settled."
        # Example: if ring 3 (from bottom) is unsettled, rings 1 and 2 (smaller) 
        # must move even if they're currently at goal, because ring 3 can't move
        # until rings 1 and 2 are out of the way.
        
        # Actually, let me reconsider the settled concept:
        # Scan from LARGEST to smallest.
        # A ring is "settled" iff it's at goal AND all LARGER rings are settled.
        # Once a ring is unsettled, all smaller rings are also unsettled
        # (they'll need to move to let the unsettled ring move).
        
        highest_unsettled_idx = -1
        all_settled_below = True
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if ring in self.goal_positions:
                if all_settled_below and self._is_ring_at_goal(state_bitwise, ring):
                    continue  # settled
                else:
                    all_settled_below = False
                    highest_unsettled_idx = i
                    # Don't break - we want the highest (largest index going down)
                    # Actually we want the first unsettled from bottom
                    # which gives us the largest unsettled ring index
                    break
        
        if highest_unsettled_idx < 0:
            return 0
        
        # All rings from index 0 to highest_unsettled_idx must move
        num_rings_to_move = highest_unsettled_idx + 1
        return (1 << num_rings_to_move) - 1

    def _misplaced_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count of misplaced rings.
        Each misplaced ring needs at least 1 move.
        """
        count = 0
        for ring in self.rings_by_size:
            if ring in self.goal_positions:
                if not self._is_ring_at_goal(state_bitwise, ring):
                    count += 1
        return count

    def _task_network_bound(self, task_network) -> int:
        """
        Term 3: Lower bound from the task network.
        
        Count the minimum number of primitive actions that must be executed.
        - Each Operator in the task network = 1 action
        - Each move_abstract abstract task = at least 1 action
        - Each other compound task contributes based on its minimum decomposition depth
          (conservatively, we use 0 for non-move tasks to stay admissible)
        """
        if not task_network:
            return 0
        
        count = 0
        for task in task_network:
            # Check if it's an operator (primitive)
            if hasattr(task, 'add_effects'):
                # It's an Operator
                count += 1
            elif hasattr(task, 'decompositions'):
                # It's an AbstractTask
                # move_abstract decomposes to exactly 1 move action
                if task.name in self.move_abstract_task_names:
                    count += 1
                elif "move_abstract" in task.name:
                    count += 1
                # For rotateTower, exchange, shiftTower, selectDirection:
                # These decompose into subtasks that eventually produce moves.
                # Being conservative (admissible), we don't add extra here
                # because the exponential bound already captures this.
        
        return count

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic as max of multiple admissible terms.
        """
        if not self.rings_by_size:
            return 0
        
        # Check if goal is reached
        if self.model.goal_reached(state_bitwise, task_network):
            return 0

        term1 = self._exponential_bound(state_bitwise)
        term2 = self._misplaced_count(state_bitwise)
        term3 = self._task_network_bound(task_network)

        return max(term1, term2, term3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic13(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic13 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 13 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
