"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single constraint that most serializes work is that only the top ring
  of a tower can be moved, and larger rings cannot be placed on smaller ones.
  To move ring k to its goal, ALL k-1 smaller rings must first be moved out
  of the way. This creates exponential work: 2^k - 1 moves minimum for k rings.

Independent Admissible Lower Bounds:

  Term 1 — Largest-misplaced-ring bound (2^k - 1):
    Find the largest ring not at its goal position. All rings from the smallest
    up to (and including) that ring must move at least once. The classic Tower
    of Hanoi solution shows 2^k - 1 is the minimum number of moves for k rings.
    Admissible because it equals the optimal solution for the subproblem of
    settling that ring and all smaller rings.

  Term 2 — Count of unsettled rings:
    Count the number of rings not currently on their goal target. Each such ring
    requires at least 1 move. This is a trivial admissible lower bound.
    While dominated by Term 1 in most cases, it provides a floor when the
    largest-misplaced analysis might miss edge cases.

  Term 3 — Bottom-up settled chain penalty:
    Starting from the largest ring, count how many consecutive rings from the
    bottom are already settled in their goal positions. The remaining unsettled
    rings form a sub-tower of size m that needs 2^m - 1 moves. This is tighter
    than Term 1 when some large rings are already settled but smaller ones are
    scattered. Admissible because it's the exact minimum for the remaining
    sub-tower problem.

  The heuristic returns max(term1, term2, term3).

  Additionally, we add a task-network-aware component: if there are pending
  abstract tasks, we estimate a minimum decomposition cost based on the number
  of remaining compound tasks, providing additional guidance.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic4(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Uses three independent lower bounds and takes the max.
    Preprocessing builds all needed lookup structures so __call__
    uses only O(n) bit tests where n = number of rings (typically small).
    """

    def __init__(self, towers_anthropicopus46_heuristic_4=None, name="TowersAnthropicOpus46Heuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # rings sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For fast goal checking: ring_name -> goal fact global_id (if exists)
        self.goal_on_fact_ids = {}     # ring_name -> global_id of the goal on[ring, goal_target] fact
        
        # Ring index for fast lookup
        self.ring_index = {}           # ring_name -> index in rings_by_size (0=smallest)

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node.state, initial_node.task_network)
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
        
        # Build on_facts_by_ring and collect ring/size info
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

        # Sort rings by size: most smallerThan relationships = smallest ring
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        
        # Build ring index
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

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

        # Precompute goal fact global_ids for fast bitwise checking
        # For each ring, find the on[ring, goal_target] fact's global_id
        for ring, goal_target in self.goal_positions.items():
            if ring in self.on_facts_by_ring:
                for fact, target in self.on_facts_by_ring[ring]:
                    if target == goal_target:
                        self.goal_on_fact_ids[ring] = fact.global_id
                        break

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """Check if ring is at its goal position using a single bit test."""
        if ring not in self.goal_on_fact_ids:
            return True  # No goal for this ring, consider it satisfied
        gid = self.goal_on_fact_ids[ring]
        return (state_bitwise >> gid) & 1

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
        Term 1: Find largest misplaced ring, return 2^k - 1.
        
        Scan from largest ring to smallest. The first (largest) ring not at
        its goal position determines k = index + 1 (number of rings from
        smallest up to and including it). Return 2^k - 1.
        """
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1
        return 0

    def _term2_unsettled_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count rings not at their goal position.
        Each needs at least 1 move.
        """
        count = 0
        for ring in self.rings_by_size:
            if not self._is_ring_at_goal(state_bitwise, ring):
                count += 1
        return count

    def _term3_bottom_up_settled(self, state_bitwise: int) -> int:
        """
        Term 3: Bottom-up settled chain analysis.
        
        Starting from the largest ring, count how many consecutive rings
        from the bottom are settled at their goal AND their goal chain is
        intact (i.e., ring k is on ring k+1 which is on ring k+2 ... and
        the bottom ring is on the goal tower).
        
        The remaining unsettled rings form a sub-problem needing 2^m - 1 moves.
        
        This is tighter than Term 1 when e.g. the bottom 5 rings are settled
        but the top 3 are scattered (gives 2^3 - 1 = 7 instead of 2^8 - 1 = 255).
        
        Actually, Term 1 already handles this correctly since it scans from
        largest to smallest. So Term 3 adds value when we check the goal chain
        more carefully: a ring might be "on" its goal target but the target
        itself might not be settled yet.
        """
        # Check from bottom (largest) upward for a settled chain
        settled_from_bottom = 0
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if ring not in self.goal_positions:
                break
            
            # Check if this ring is at its goal position
            if not self._is_ring_at_goal(state_bitwise, ring):
                break
            
            # Additionally check: is the ring's goal target also settled?
            # For the largest ring, its goal target is a tower (always stable).
            # For other rings, goal target should be the ring below it,
            # and that ring should also be at its goal.
            goal_target = self.goal_positions[ring]
            if goal_target.startswith("r"):
                # The target is another ring; check if that ring is also at goal
                if not self._is_ring_at_goal(state_bitwise, goal_target):
                    break
            
            settled_from_bottom += 1
        
        unsettled = self.num_rings - settled_from_bottom
        if unsettled <= 0:
            return 0
        return (1 << unsettled) - 1

    def _term4_task_network_estimate(self, task_network) -> int:
        """
        Term 4: Task network lower bound.
        
        Count the minimum number of primitive actions that must result from
        decomposing the remaining task network. Each move_abstract task yields
        exactly 1 action. Each rotateTower yields at least 1 action (its own
        move_abstract). Each exchange can yield 0 (exchangeClear) or more.
        Each shiftTower/selectDirection eventually leads to rotateTower(s).
        
        We count the number of Operator instances (already primitive) and
        estimate abstract tasks conservatively.
        """
        if not task_network:
            return 0
        
        count = 0
        for task in task_network:
            # Check if it's an operator (primitive action)
            if hasattr(task, 'add_effects'):
                count += 1
            else:
                # Abstract task - estimate minimum decomposition cost
                # move_abstract -> exactly 1 move
                # rotateTower -> 1 move_abstract + 1 exchange (>= 0 moves) = >= 1
                # exchange -> 0 (exchangeClear) or >= 1
                # shiftTower -> >= 1
                # selectDirection -> >= 1
                task_name = task.name if hasattr(task, 'name') else ''
                if 'move_abstract' in task_name:
                    count += 1
                elif 'rotateTower' in task_name:
                    count += 1  # At least 1 move from the move_abstract subtask
                elif 'shiftTower' in task_name:
                    count += 1  # At least 1 move
                elif 'selectDirection' in task_name:
                    count += 1  # Will eventually reach rotateTower
                elif 'exchange' in task_name:
                    # exchangeClear produces 0 moves, so we can't guarantee any
                    count += 0
                else:
                    count += 1  # Generic: at least 1 action per abstract task
        return count

    def _calculate_heuristic(self, state_bitwise: int, task_network=None) -> int:
        if not self.rings_by_size:
            return 0

        t1 = self._term1_largest_misplaced(state_bitwise)
        t2 = self._term2_unsettled_count(state_bitwise)
        t3 = self._term3_bottom_up_settled(state_bitwise)
        
        # Task network estimate
        t4 = 0
        if task_network is not None:
            t4 = self._term4_task_network_estimate(task_network)

        return max(t1, t2, t3, t4)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic4(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic4 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
