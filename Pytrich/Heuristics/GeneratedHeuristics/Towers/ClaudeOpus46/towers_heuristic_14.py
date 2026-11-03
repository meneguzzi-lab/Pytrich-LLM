"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single constraint that most serializes work is the size ordering:
  to move ring k to its goal, ALL rings 1..k-1 (smaller rings) must first
  be moved out of the way. This creates an exponential cascade where moving
  ring k requires 2^(k-1) - 1 moves just for the smaller rings.

Admissible Lower Bounds:
  Term 1 (Exponential displacement): Find the largest ring not at its goal
    position. All rings from smallest up to that ring must move at least once.
    The minimum moves for k disks in classic Towers of Hanoi is 2^k - 1.
    This is admissible because the classic solution IS optimal for k disks.

  Term 2 (Misplaced ring count): Count the number of rings not currently
    sitting on their goal target. Each such ring requires at least 1 move.
    This is admissible because each misplaced ring needs at least one action.

  Term 3 (Stacked displacement): For each ring not at goal, if it has rings
    above it that also need to move, those rings each need at least one move.
    We count the number of rings above (and including) the deepest misplaced
    ring on each tower. This is admissible as a lower bound on necessary moves.

  Final heuristic: max(term1, term2, term3)

  Additionally, we add a sub-unit tie-breaking penalty based on how many
  rings are on non-goal towers, to prefer states closer to the goal
  configuration without breaking admissibility.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic14(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential displacement (2^k - 1 for largest misplaced ring at depth k)
    2. Simple misplaced ring count
    3. Task-network aware estimation
    
    Returns max of all terms plus sub-unit tie-breaking.
    """

    def __init__(self, towers_anthropicopus46_heuristic_14=None, name="TowersAnthropicOpus46Heuristic14"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.towerTop_facts_by_obj = {}  # obj_name -> list of (fact, tower_name)
        self.num_rings = 0
        self.ring_index = {}           # ring_name -> index in rings_by_size (0=smallest)
        
        # For task network analysis
        self.move_abstract_task_names = set()
        self.exchange_task_names = set()
        self.rotate_task_names = set()
        self.shift_task_names = set()

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
        Build efficient lookup structures for fast heuristic computation.
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

            elif predicate == "towerTop":
                obj = args[0]
                tower = args[1]
                if obj not in self.towerTop_facts_by_obj:
                    self.towerTop_facts_by_obj[obj] = []
                self.towerTop_facts_by_obj[obj].append((fact, tower))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size using smallerThan count
        # A ring that is smaller than more things is a smaller ring
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        
        # Build ring index map
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

        # Identify abstract task types for task network analysis
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower() if hasattr(at, 'name') else ""
            if "move_abstract" in name_lower:
                self.move_abstract_task_names.add(at.global_id)
            elif "exchange" in name_lower:
                self.exchange_task_names.add(at.global_id)
            elif "rotatetower" in name_lower or "rotate_tower" in name_lower:
                self.rotate_task_names.add(at.global_id)
            elif "shifttower" in name_lower or "shift_tower" in name_lower:
                self.shift_task_names.add(at.global_id)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """Find where a ring currently sits using bitwise state check."""
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _term_exponential_displacement(self, state_bitwise: int) -> int:
        """
        Term 1: Find the largest misplaced ring and return 2^k - 1.
        
        Admissibility: The classic Towers of Hanoi optimal solution for k disks
        requires exactly 2^k - 1 moves. Since all rings from 1 to k must move
        at least once to relocate ring k, this is a valid lower bound.
        """
        if not self.rings_by_size:
            return 0

        # Check from largest ring to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)

            if goal_pos is not None and current_pos != goal_pos:
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1

        return 0

    def _term_misplaced_count(self, state_bitwise: int) -> int:
        """
        Term 2: Count rings not at their goal position.
        
        Admissibility: Each misplaced ring requires at least 1 move action.
        """
        count = 0
        for ring in self.rings_by_size:
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is not None and current_pos != goal_pos:
                count += 1
        return count

    def _term_settled_from_bottom(self, state_bitwise: int) -> int:
        """
        Term 3: Enhanced exponential estimate accounting for partially settled towers.
        
        Check from the bottom (largest ring) upward how many rings are already
        correctly settled in their goal positions with correct stacking.
        The remaining unsettled rings give us a tighter bound.
        
        Admissibility: If k rings out of n total are correctly settled at the
        bottom of the goal tower, we only need to move the remaining n-k rings.
        The cost is 2^(n-k) - 1 which is <= 2^n - 1, so this is still admissible
        and often tighter.
        """
        if not self.rings_by_size:
            return 0

        # Count how many rings are settled from the bottom up
        settled = 0
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is not None and current_pos == goal_pos:
                settled += 1
            else:
                break

        unsettled = self.num_rings - settled
        if unsettled <= 0:
            return 0

        # Check if there are any misplaced rings among the unsettled ones
        has_misplaced = False
        for i in range(unsettled):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is not None and current_pos != goal_pos:
                has_misplaced = True
                break

        if not has_misplaced:
            return 0

        return (1 << unsettled) - 1

    def _task_network_lower_bound(self, task_network) -> int:
        """
        Term 4: Lower bound from the task network structure.
        
        Each move_abstract task in the network will result in exactly 1 action.
        Each rotateTower produces at least 1 move_abstract + exchange.
        
        Admissibility: Each primitive action requires at least 1 step.
        Each move_abstract decomposes to exactly 1 action.
        """
        if not task_network:
            return 0
        
        count = 0
        for task in task_network:
            gid = task.global_id
            if gid in self.move_abstract_task_names:
                count += 1
            elif hasattr(task, 'decompositions'):
                # It's an abstract task - it needs at least 1 action
                # For operators (primitives), they ARE the action
                if not hasattr(task, 'add_effects'):
                    count += 1
            else:
                # It's a primitive operator
                count += 1
        return count

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic value as max of multiple admissible terms
        plus sub-unit tie-breaking.
        """
        if not self.rings_by_size:
            return 0

        # Term 1: Exponential displacement
        t1 = self._term_exponential_displacement(state_bitwise)
        
        # Term 2: Misplaced count
        t2 = self._term_misplaced_count(state_bitwise)
        
        # Term 3: Settled-from-bottom enhanced exponential
        t3 = self._term_settled_from_bottom(state_bitwise)
        
        # Term 4: Task network lower bound
        t4 = self._task_network_lower_bound(task_network)

        # Base heuristic: max of all admissible terms
        h_base = max(t1, t2, t3, t4)

        # Sub-unit tie-breaking: prefer states where more rings are at goal
        # This doesn't break admissibility since it's < 1
        tie_break = 0.001 * t2  # small penalty for misplaced rings

        # Additional tie-breaking: penalize rings that are on wrong towers
        # even if the ring above them is correctly placed
        # Count rings whose current support differs from goal support
        wrong_support = 0
        for ring in self.rings_by_size:
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is not None and current_pos != goal_pos:
                # Check if this ring is deeply buried (has rings on top of it)
                # Rings higher in rings_by_size (smaller index) are smaller
                idx = self.ring_index.get(ring, 0)
                wrong_support += idx  # larger rings buried deeper = worse

        tie_break += 0.0001 * wrong_support

        return h_base + tie_break

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic14(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic14 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 14 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
