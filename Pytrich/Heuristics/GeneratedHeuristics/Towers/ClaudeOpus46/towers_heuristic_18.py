"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The single constraint that most serializes work is the size ordering:
  to move ring k to its goal, ALL rings 1..k-1 (smaller rings) must first
  be moved out of the way. This creates an exponential lower bound.

Admissible Lower Bounds:
  Term 1 (Exponential displacement): Find the largest ring not at its goal
  position. To move it, all smaller rings must be displaced first.
  Minimum moves = 2^k - 1 where k = index+1 of largest displaced ring.
  Admissible because the classic Tower of Hanoi optimal solution is 2^k-1.

  Term 2 (Count of displaced rings): Simply count how many rings are not
  on their goal target. Each such ring needs at least 1 move.
  Admissible because each displaced ring requires at least one action.

  Term 3 (Task network lower bound): Count the number of move_abstract
  tasks in the current task network. Each such task decomposes to exactly
  one move action, so this is a perfect count of remaining moves from
  the HTN perspective for those tasks already committed.
  Admissible because each move_abstract requires exactly 1 action.

  Final heuristic = max(term1, term2, term3)
  Since each term is admissible, max is also admissible.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic18(Heuristic):
    """
    Multi-component max heuristic for Towers of Hanoi.
    
    Combines:
    1. Exponential displacement bound (2^k - 1)
    2. Simple displaced ring count
    3. Task network move_abstract count
    
    Also includes a refinement: checks whether rings that ARE at their
    goal position are "stable" (i.e., all rings below them are also at
    goal). Unstable goal-positioned rings will need to move again.
    """

    def __init__(self, towers_anthropicopus46_heuristic_18=None, name="TowersAnthropicOpus46Heuristic18"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge
        self.goal_positions = {}       # ring_name -> target_obj_name
        self.rings_by_size = []        # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}     # ring_name -> list of (fact, target_obj_name)
        self.num_rings = 0
        
        # For task network analysis
        self.move_abstract_task_names = set()  # set of abstract task name substrings for move_abstract
        self.rotate_tower_task_names = set()
        self.exchange_task_names = set()
        self.shift_tower_task_names = set()
        
        # Precomputed: for each ring, its index in rings_by_size (0=smallest)
        self.ring_index = {}  # ring_name -> index in rings_by_size
        
        # Goal on facts (from goal_on predicate in initial state)
        self.goal_on_from_init = {}  # ring -> target from goal_on predicates

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        self._preprocess()
        
        # Identify move_abstract task names for task network counting
        self._identify_task_types()

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
        smaller_than_all = []    # all smallerThan pairs

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

        # Also extract goal_on from initial facts for additional info
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "goal_on":
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    self.goal_on_from_init[ring] = target

            if predicate == "smallerThan":
                smaller = args[0]
                larger = args[1]
                smaller_than_all.append((smaller, larger))
                if smaller.startswith("r"):
                    rings.add(smaller)
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

        # Sort rings by size using smallerThan count
        # A ring that is smaller than more things is smaller overall
        if smaller_than_all:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than_all:
                if smaller in rings:
                    smaller_count[smaller] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by ring number
            def ring_num(r):
                try:
                    return int(r[1:])
                except:
                    return 0
            self.rings_by_size = sorted(list(rings), key=ring_num)

        # Build ring index map
        for i, ring in enumerate(self.rings_by_size):
            self.ring_index[ring] = i

        # If goal_positions is empty, try using goal_on_from_init
        if not self.goal_positions and self.goal_on_from_init:
            self.goal_positions = dict(self.goal_on_from_init)

    def _identify_task_types(self):
        """Identify abstract task types for task network analysis."""
        for at in self.model.abstract_tasks:
            name_lower = at.name.lower()
            if "move_abstract" in name_lower:
                self.move_abstract_task_names.add(at.global_id)
            elif "rotatetower" in name_lower:
                self.rotate_tower_task_names.add(at.global_id)
            elif "exchange" in name_lower:
                self.exchange_task_names.add(at.global_id)
            elif "shifttower" in name_lower:
                self.shift_tower_task_names.add(at.global_id)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _term1_exponential_displacement(self, state_bitwise: int) -> int:
        """
        Find the largest ring not at its goal position.
        All rings from smallest up to (and including) that ring must move.
        Returns 2^k - 1 where k is the count of those rings.
        
        Enhanced: also checks for "false stability" - a ring at its goal
        position but sitting on a ring that is NOT at its goal position
        is itself unstable and will need to move.
        """
        if not self.rings_by_size:
            return 0

        # Determine which rings are truly settled (at goal AND all below are at goal)
        # Process from largest to smallest
        settled = [False] * len(self.rings_by_size)
        
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            
            if current_pos is None or goal_pos is None:
                settled[i] = False
                continue
                
            if current_pos != goal_pos:
                settled[i] = False
            else:
                # Ring is at goal. But is it stable?
                # It's stable only if it's the largest ring OR
                # the ring below it (i+1) is also settled
                if i == len(self.rings_by_size) - 1:
                    # Largest ring, if at goal it's settled
                    settled[i] = True
                else:
                    # Check if the ring it's sitting on is also settled
                    # current_pos == goal_pos, and goal_pos should be the next larger ring
                    # which is at index i+1
                    settled[i] = settled[i + 1]

        # Find largest unsettled ring
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            if not settled[i]:
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1

        return 0

    def _term2_displaced_count(self, state_bitwise: int) -> int:
        """
        Count rings not at their goal position.
        Each needs at least 1 move. Admissible.
        """
        count = 0
        for ring in self.rings_by_size:
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if current_pos is not None and goal_pos is not None and current_pos != goal_pos:
                count += 1
        return count

    def _term3_task_network_bound(self, task_network) -> int:
        """
        Count move_abstract tasks in the task network.
        Each decomposes to exactly 1 move action.
        Also count operators directly (they are 1 action each).
        
        For higher-level tasks, estimate minimum moves:
        - rotateTower decomposes to 1 move_abstract + 1 exchange
        - exchange can be empty (0 moves) or involves moves
        - shiftTower involves recursive structure
        
        We conservatively count only direct move_abstract tasks and operators.
        """
        if task_network is None:
            return 0
        
        count = 0
        for task in task_network:
            gid = task.global_id
            # Check if it's an operator (direct action)
            if gid >= self.model.iop_init and gid <= self.model.iop_end:
                count += 1
            # Check if it's a move_abstract task (decomposes to exactly 1 action)
            elif gid in self.move_abstract_task_names:
                count += 1
            # rotateTower = move_abstract + exchange >= 1 action
            elif gid in self.rotate_tower_task_names:
                count += 1
        
        return count

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic as max of multiple admissible terms.
        """
        t1 = self._term1_exponential_displacement(state_bitwise)
        t2 = self._term2_displaced_count(state_bitwise)
        t3 = self._term3_task_network_bound(task_network)
        
        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic18(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic18 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 18 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
