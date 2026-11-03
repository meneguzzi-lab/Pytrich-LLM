import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic13(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Rationale and domain insights:
    - The domain encodes the classic Towers of Hanoi using three towers (t1, t2, t3) and a set of rings r1..rk.
    - The primitive action is 'move' with legality governed by 'towerTop' and 'smallerThan'.
    - The HTN structure (shiftTower -> selectDirection -> rotateTower -> exchange/move_abstract -> move) implements the optimal recursive strategy.

    Heuristic design (admissible, consistent, and efficient):
    - The minimum number of moves to place the largest misplaced ring correctly (and thus eventually solve the problem) is 2^k - 1 where k is the number of rings from the smallest up to that largest misplaced ring.
    - We therefore:
      1) Identify all rings and sort them from smallest to largest using smallerThan relations (ring-to-ring) if present, and/or using the goal "on" chain as a fallback.
      2) Extract the goal positions for each ring from model.goals (bitwise goal state).
      3) For the current state, find the largest ring that is not at its goal position.
      4) Return (1 << (index+1)) - 1 where index is the zero-based index of that ring in the size ordering.

    Why this works well for HTN planning in this domain:
    - The HTN encodes the optimal strategy; the above lower bound is tight in Towers of Hanoi and independent of the particular decomposition path chosen.
    - It uses only bitwise checks for facts, avoiding expensive conversions per node.
    - It scales smoothly from the smallest to the largest provided instances.

    Implementation details:
    - We strictly parse grounded fact names in the "+predicate[arg1,arg2]" format (and also support the parenthesized HDDL form for robustness).
    - We obtain goals via model.state_explicit_repr(model.goals), as goals are stored bitwise in Pytrich.
    - We preprocess once (in initialize), building:
        * goal_positions: ring -> goal support object
        * rings_by_size: list of rings sorted smallest -> largest (topological order over smallerThan ring->ring)
        * on_facts_by_ring: ring -> list of (Fact, support_object) for fast bitwise membership checks

    Admissibility & Consistency:
    - The heuristic never overestimates the true remaining cost, and each move reduces the estimate by at most the move cost, ensuring consistency in this domain.
    """

    def __init__(self, towers_gpt5_heuristic_13=None, name="TowersGPT5Heuristic13"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Filled at preprocess
        self.goal_positions = {}       # ring -> goal support object (ring or tower)
        self.rings_by_size = []        # [smallest, ..., largest]
        self.on_facts_by_ring = {}     # ring -> list of (Fact, target_obj)
        self.num_rings = 0

        # Internal caches
        self._ring_set = set()

    # ---------- Framework hooks ----------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    # ---------- Core heuristic ----------
    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest; return bound for first misplaced ring
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if current_pos != goal_pos:
                # Number of rings from smallest up to this ring = i+1
                k = i + 1
                return (1 << k) - 1  # 2^k - 1
        return 0

    # ---------- Preprocessing ----------
    def _preprocess(self):
        # 1) Extract goal positions from the bitwise goal state (only 'on' facts)
        self._extract_goal_positions_from_bitwise_goals()

        # 2) Collect ring set and build mapping from rings to their possible 'on' facts
        ring_edges = self._collect_facts_and_edges()

        # 3) Build a total order for rings (smallest -> largest) using topological sort over ring_edges
        self.rings_by_size = self._topological_sort_rings(self._ring_set, ring_edges)
        self.num_rings = len(self.rings_by_size)

    def _extract_goal_positions_from_bitwise_goals(self):
        self.goal_positions.clear()
        if hasattr(self.model, 'goals') and self.model.goals:
            for fact_name in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(fact_name)
                if pred == 'on' and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if self._is_ring(ring):
                        self.goal_positions[ring] = target

    def _collect_facts_and_edges(self):
        # edges: directed edges ring_a -> ring_b if ring_a smallerThan ring_b
        edges = defaultdict(set)
        self._ring_set = set()

        # First pass: identify rings and ring->ring smallerThan edges
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue

            if pred == 'smallerThan':
                a, b = args[0], args[1]
                if self._is_ring(a):
                    self._ring_set.add(a)
                    # Only ring->ring edges matter for ordering
                    if self._is_ring(b):
                        self._ring_set.add(b)
                        edges[a].add(b)

        # Fallback edges from goal chain (on r_i r_{i+1} implies r_i < r_{i+1})
        if not edges and self.goal_positions:
            for r, tgt in self.goal_positions.items():
                if self._is_ring(r) and self._is_ring(tgt):
                    edges[r].add(tgt)
                    self._ring_set.add(r)
                    self._ring_set.add(tgt)

        # Second pass: collect 'on' facts per ring for quick bitwise checks
        self.on_facts_by_ring.clear()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on' and len(args) >= 2:
                r, obj = args[0], args[1]
                if self._is_ring(r):
                    self._ring_set.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, obj))

        return edges

    def _topological_sort_rings(self, rings: set, edges: dict):
        if not rings:
            return []

        # Build indegree map
        indeg = {r: 0 for r in rings}
        for a in edges:
            for b in edges[a]:
                if b in indeg:
                    indeg[b] += 1

        # Kahn's algorithm: rings with indegree 0 are the SMALLEST
        q = deque([r for r, d in indeg.items() if d == 0])
        order = []
        # Stable order by name for determinism
        q = deque(sorted(q))

        local_edges = {a: set(bs) for a, bs in edges.items()}

        while q:
            u = q.popleft()
            order.append(u)
            for v in list(local_edges.get(u, [])):
                indeg[v] -= 1
                local_edges[u].remove(v)
                if indeg[v] == 0:
                    # insert maintaining lexicographic order for stability
                    inserted = False
                    for i in range(len(q)):
                        if v < q[i]:
                            q.insert(i, v)
                            inserted = True
                            break
                    if not inserted:
                        q.append(v)

        # If order incomplete (no ring->ring edges), fall back to deterministic name sort
        if len(order) != len(rings):
            return sorted(list(rings))
        return order

    # ---------- Utilities ----------
    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        # Find which 'on[ring,obj]' fact is true in the bitwise state
        for fact, target in self.on_facts_by_ring.get(ring, []):
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _is_ring(self, name: str) -> bool:
        # Rings start with 'r' in provided instances
        return isinstance(name, str) and name.startswith('r')

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Grounded: +predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            b = name2.find("[")
            if b > 0 and name2.endswith("]"):
                pred = name2[:b]
                args_str = name2[b + 1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        # HDDL-style backup: (predicate arg1 arg2)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    # ---------- Introspection ----------
    def __repr__(self):
        return f"TowersGPT5Heuristic13(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic13 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic13 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
