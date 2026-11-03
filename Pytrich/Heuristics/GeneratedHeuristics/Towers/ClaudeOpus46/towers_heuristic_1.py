"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

Bottleneck Analysis:
  The serializing constraint is that only the top ring of any tower can be moved,
  and a larger ring can never be placed on a smaller ring. To move ring k to its
  goal, ALL rings 1..k-1 (smaller rings) must first be cleared off it, requiring
  at minimum 2^(k-1) - 1 moves, then ring k itself must move (1 move), then the
  smaller rings must be reassembled (potentially 2^(k-1) - 1 more moves).

Admissible Lower Bounds:
  Term 1 (Largest-Misplaced-Ring Bound): Find the largest ring not at its goal
    position. All rings from smallest up to and including that ring must participate
    in moves. The classic optimal solution for k disks is 2^k - 1 moves. This is
    admissible because it equals the optimal solution cost for the standard problem
    and is a lower bound for any configuration.

  Term 2 (Count-Misplaced Bound): Count the number of rings not currently on their
    goal target. Each such ring needs at least 1 move. This is trivially admissible.

  Term 3 (Bottom-Up Settled Analysis): Starting from the largest ring, count how many
    rings are already correctly settled (in goal position with all rings below them
    also settled). The first unsettled ring from the bottom defines a subtower of size
    k that must be solved, giving 2^k - 1. This refines Term 1 by not counting rings
    that are already correctly placed at the bottom of the goal tower.

  We return max(term1, term2, term3) which is admissible since max of admissible
  terms is admissible.

Implementation Notes:
  - All fact parsing and map building happens in initialize() (O(|facts|) once).
  - __call__ uses only O(num_rings) bitwise checks, no iteration over model collections.
  - Rings are sorted by size using smallerThan relation counts.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersAnthropicOpus46Heuristic1(Heuristic):
    """
    Multi-component admissible heuristic for Towers of Hanoi.
    
    Combines three admissible lower bounds:
    1. Largest-misplaced-ring exponential bound
    2. Count of misplaced rings
    3. Bottom-up settled analysis (refined exponential bound)
    """

    def __init__(self, towers_anthropicopus46_heuristic_1=None, name="TowersAnthropicOpus46Heuristic1"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring_name -> target_obj_name (from goal state)
        self.rings_by_size = []       # List of ring names sorted smallest to largest
        self.on_facts_by_ring = {}    # ring_name -> list of (fact, target_obj_name)
        self.on_fact_lookup = {}      # (ring_name, target_name) -> fact.global_id
        self.num_rings = 0
        
        # For fast settled analysis
        self.goal_on_global_ids = {}  # ring_name -> global_id of the on[ring, goal_target] fact

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocess the model: extract ring ordering, goal positions, and fact maps.
        Called once before search starts.
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
        Extract domain structure from grounded facts.
        Build efficient lookup structures for the heuristic computation.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller, larger) pairs
        
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

        # Step 2: Parse all facts to build lookup structures
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
                    self.on_fact_lookup[(ring, target)] = fact.global_id

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Step 3: Sort rings by size using smallerThan count
        # A ring that is smaller than more things is a smaller ring
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than_pairs:
                if smaller in rings:
                    smaller_count[smaller] += 1
            # More count = smaller ring, so sort descending by count = smallest first
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by ring number
            self.rings_by_size = sorted(list(rings), key=lambda r: int(r[1:]))

        # Step 4: Precompute goal on fact global_ids for fast settled checking
        for ring, target in self.goal_positions.items():
            key = (ring, target)
            if key in self.on_fact_lookup:
                self.goal_on_global_ids[ring] = self.on_fact_lookup[key]

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Find where a ring currently is by checking which 'on' fact is true.
        Uses O(k) bitwise checks where k is the number of possible positions for the ring.
        """
        if ring not in self.on_facts_by_ring:
            return None
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring_at_goal(self, state_bitwise: int, ring: str) -> bool:
        """
        Fast check: is this ring currently at its goal position?
        Single bitwise operation.
        """
        if ring not in self.goal_on_global_ids:
            return True  # No goal for this ring, consider it settled
        gid = self.goal_on_global_ids[ring]
        return (state_bitwise >> gid) & 1

    def _term1_largest_misplaced(self, state_bitwise: int) -> int:
        """
        Term 1: Find the largest ring not at its goal position.
        Return 2^k - 1 where k = index+1 of that ring in size order.
        
        Admissible because moving the k-th largest ring requires moving
        all k-1 smaller rings first (classic Tower of Hanoi lower bound).
        """
        if not self.rings_by_size:
            return 0

        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            if not self._is_ring_at_goal(state_bitwise, ring):
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1
        return 0

    def _term2_count_misplaced(self, state_bitwise: int) -> int:
        """
        Term 2: Count rings not at their goal position.
        Each misplaced ring needs at least 1 move.
        Trivially admissible.
        """
        count = 0
        for ring in self.rings_by_size:
            if not self._is_ring_at_goal(state_bitwise, ring):
                count += 1
        return count

    def _term3_bottom_up_settled(self, state_bitwise: int) -> int:
        """
        Term 3: Bottom-up settled analysis.
        
        Starting from the largest ring, count how many consecutive rings
        (from largest going up) are already at their goal positions.
        These rings don't need to move. The remaining unsettled rings
        form a subtower of size k, requiring at least 2^k - 1 moves.
        
        This refines Term 1 by recognizing that rings already correctly
        placed at the bottom of the goal tower don't contribute to cost.
        
        Admissible: if the bottom m rings are settled, we have a k-ring
        subproblem (k = total - m) that requires >= 2^k - 1 moves.
        """
        if not self.rings_by_size:
            return 0

        n = len(self.rings_by_size)
        settled_from_bottom = 0
        
        # Check from largest (bottom) ring upward
        for i in range(n - 1, -1, -1):
            ring = self.rings_by_size[i]
            if self._is_ring_at_goal(state_bitwise, ring):
                settled_from_bottom += 1
            else:
                break

        unsettled = n - settled_from_bottom
        if unsettled == 0:
            return 0
        return (1 << unsettled) - 1

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Compute the heuristic value as max of three admissible terms.
        """
        if not self.rings_by_size:
            return 0

        # If task network is empty and goals met, we're done
        if len(task_network) == 0:
            return 0

        t1 = self._term1_largest_misplaced(state_bitwise)
        t2 = self._term2_count_misplaced(state_bitwise)
        t3 = self._term3_bottom_up_settled(state_bitwise)

        return max(t1, t2, t3)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic value during search.
        Called for every node expansion - kept fast with bitwise operations only.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersAnthropicOpus46Heuristic1(rings={self.num_rings})"

    def __str__(self):
        return f"TowersAnthropicOpus46Heuristic1 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers AnthropicOpus46 Heuristic 1 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
