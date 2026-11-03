import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - In the Towers domain, every legal plan that moves the largest misplaced ring
      must first relocate all smaller rings out of its way, then put them back on top
      afterward. This induces the classic exponential lower bound of 2^k - 1 moves
      for k rings (smallest..that largest misplaced ring).
    - The hierarchy (shiftTower -> selectDirection -> rotateTower -> move_abstract -> move)
      always decomposes down to the primitive 'move' operator, so a lower bound on the
      number of remaining 'move' actions is also a lower bound for the HTN problem.
    - We combine two admissible bounds and take their maximum:
        1) Hanoi bound (largest misplaced ring): 2^(k) - 1
        2) Unachieved-goal-count bound: count of rings not currently at their goal support.
       The latter accounts for the fact that each ring requires at least one final move
       to achieve its goal 'on' fact; it can dominate in very late stages, though the
       Hanoi bound typically dominates earlier.

    Efficiency considerations:
    - All expensive preprocessing (fact parsing, goal extraction, size ordering) is done
      once in initialize().
    - During search, only fast bitwise checks are used to test current 'on' facts.

    Admissibility and consistency:
    - The heuristic returns a lower bound on the number of remaining primitive 'move'
      actions. It is admissible (never overestimates). The largest-misplaced-ring bound
      is a standard admissible lower bound for Towers of Hanoi. The goal-count bound is
      also admissible because each remaining ring needs at least one final placement move.
    """

    def __init__(self, towers_gpt5_heuristic_3=None, name="TowersGPT5Heuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}       # ring -> target object (from goal facts "on")
        self.rings_by_size = []        # list of rings sorted smallest -> largest
        self.on_facts_by_ring = {}     # ring -> list of (Fact, target_obj)
        self.num_rings = 0
        self.rings_set = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # Preprocess static domain information
        self._preprocess()
        self.preprocessing_time = time.time() - start_time

        # Initial heuristic value
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a grounded fact string in the form "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]".
        Returns (predicate, [arg1, arg2, ...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # strip +/-
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _numeric_ring_key(self, r: str):
        # Robust numeric key for rings named like r1, r2, ..., fallback to string
        try:
            if r.startswith("r"):
                return int(r[1:])
        except Exception:
            pass
        return float('inf')

    def _preprocess(self):
        rings = set()
        smaller_edges = []  # (smaller_ring, larger_ring)

        # 1) Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)
                        if target.startswith("r"):
                            rings.add(target)

        # 2) Scan all facts to build ring set, on-facts index per ring, and smaller-than edges
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if target.startswith("r"):
                        rings.add(target)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

            elif predicate == "smallerThan":
                small, large = args[0], args[1]
                if small.startswith("r"):
                    rings.add(small)
                    if large.startswith("r"):
                        rings.add(large)
                        smaller_edges.append((small, large))

        self.rings_set = rings
        self.num_rings = len(rings)

        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine size order (smallest -> largest)
        # Primary: use smallerThan ring->ring edges if available
        order_from_edges = []
        if smaller_edges:
            larger_count = defaultdict(int)
            for s, l in smaller_edges:
                if s in rings:
                    larger_count[s] += 1
            # In case some rings have no outgoing edges (e.g., the largest), ensure key exists
            for r in rings:
                _ = larger_count[r]
            order_from_edges = sorted(list(rings), key=lambda r: (-larger_count[r], self._numeric_ring_key(r), r))

        # Fallback: derive partial order from goal 'on' facts (ring on ring => smaller < larger)
        if not order_from_edges and self.goal_positions:
            tmp_edges = []
            for r, tgt in self.goal_positions.items():
                if r in rings and tgt in rings:
                    tmp_edges.append((r, tgt))
            if tmp_edges:
                larger_count = defaultdict(int)
                for s, l in tmp_edges:
                    larger_count[s] += 1
                for r in rings:
                    _ = larger_count[r]
                order_from_edges = sorted(list(rings), key=lambda r: (-larger_count[r], self._numeric_ring_key(r), r))

        # Last fallback: numeric by name (r1 < r2 < ...)
        if not order_from_edges:
            order_from_edges = sorted(list(rings), key=lambda r: (self._numeric_ring_key(r), r))

        self.rings_by_size = order_from_edges

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the current support object of 'ring' by scanning its indexed 'on' facts.
        Uses fast bitwise checks on fact.global_id.
        """
        facts_list = self.on_facts_by_ring.get(ring)
        if not facts_list:
            return None
        for fact, target in facts_list:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _largest_misplaced_hanoi_lb(self, state_bitwise: int) -> int:
        """
        Hanoi lower bound: find the largest ring that is NOT at its goal support.
        If that ring is at index i (0-based from smallest), then at least 2^(i+1) - 1 moves
        remain (must move i smaller rings off/on plus the ring itself).
        """
        if not self.rings_by_size:
            return 0

        # Consider only rings that actually have goal positions
        goal_rings = [r for r in self.rings_by_size if r in self.goal_positions]
        if not goal_rings:
            return 0

        for i in range(len(goal_rings) - 1, -1, -1):  # largest -> smallest
            r = goal_rings[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                k = i + 1  # number of rings from smallest up to this ring
                return (1 << k) - 1
        return 0

    def _unachieved_goal_count_lb(self, state_bitwise: int) -> int:
        """
        Lower bound by counting how many rings are not yet at their final goal support.
        Each such ring needs at least one final move to be placed correctly.
        """
        cnt = 0
        for r, goal in self.goal_positions.items():
            cur = self._get_current_position(state_bitwise, r)
            if cur != goal:
                cnt += 1
        return cnt

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0
        h_hanoi = self._largest_misplaced_hanoi_lb(state_bitwise)
        h_goals = self._unachieved_goal_count_lb(state_bitwise)
        return max(h_hanoi, h_goals)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT5Heuristic3(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic3 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers GPT-5 Heuristic 3 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
