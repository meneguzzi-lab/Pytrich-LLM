import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - The domain models the classic 3-peg Towers of Hanoi with RING objects stacked via 'on' relations and
      goal positions captured by goal facts in the bitwise goal state (model.goals), not via goal_on() facts.
    - The unique primitive action is 'move', while methods structure the hierarchical decomposition. Regardless
      of hierarchy, the total cost is the number of primitive 'move' actions, which matches the number of legal ring moves.
    - A strong and admissible lower bound in this domain is obtained by identifying the LARGEST ring that is not yet
      at its goal position. To position that ring correctly (eventually, w.r.t. its goal), one must at least move all
      smaller rings some number of times. A classic admissible bound returns 2^k - 1, where k is the number of rings
      from the smallest up to and including that largest misplaced ring. This bound is fast to compute and scales well.

    Implementation notes:
    - We preprocess the grounded model once to:
        * Extract goal ring positions from model.goals (bitwise integer -> explicit names via model.state_explicit_repr)
        * Build a size order over rings using smallerThan facts (ring-to-ring edges), falling back gracefully if needed
        * Index all 'on' facts per ring for O(1)-ish bitwise state checks
    - During search, we check (bitwise) the current position of each ring and compare to its goal. We scan from the
      largest ring down to the smallest to find the largest misplaced one and return (1 << k) - 1 for k rings
      up to that ring. If all rings are at goal, we return 0.

    Efficiency:
    - All expensive parsing and indexing is done in initialize().
    - State queries use bitwise checks via fact.global_id, never string scanning of state.

    This heuristic is admissible and typically provides tight guidance in Towers of Hanoi tasks.
    """

    def __init__(self, towers_gpt5_heuristic_5=None, name="TowersGPT5Heuristic5"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}       # ring -> target object (from goal facts)
        self.rings_by_size = []        # list of rings ordered from smallest to largest
        self.on_facts_by_ring = {}     # ring -> list of (fact, target_obj) tuples for all grounded on(ring, X)
        self.num_rings = 0

        # Keep a fast lookup ring->bool for ring membership
        self._is_ring = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # Preprocess static domain information once
        self._preprocess()
        self.preprocessing_time = time.time() - start_time

        # Initial heuristic value (for root)
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a grounded fact name into (predicate, [args]).
        Supported formats (CRITICAL):
        - +predicate[arg1,arg2] or -predicate[arg1,arg2]
        - (predicate arg1 arg2)
        """
        name = name.strip()

        # Grounded, bitwise-encoded explicit format
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Fallback HDDL textual format if ever encountered
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        rings = set()
        ring_to_ring_edges = defaultdict(set)  # smaller -> set(larger)

        # Extract goals (CRITICAL: from model.goals bitwise integer)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    # Goals only care for rings as first argument
                    if r.startswith("r"):
                        self.goal_positions[r] = tgt

        # Scan all grounded facts for ring set, size ordering and on-fact indexing
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    rings.add(a)
                if b.startswith("r"):
                    rings.add(b)
                # We only build ring-to-ring size edges
                if a.startswith("r") and b.startswith("r"):
                    ring_to_ring_edges[a].add(b)

            elif pred == "on":
                r, tgt = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    if r not in self.on_facts_by_ring:
                        self.on_facts_by_ring[r] = []
                    self.on_facts_by_ring[r].append((fact, tgt))

        self._is_ring = rings
        self.num_rings = len(rings)

        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # Establish ring order (smallest -> largest) via topological sort on smallerThan ring->ring edges.
        # Edges: smaller -> larger
        in_degree = defaultdict(int)
        nodes = set(rings)
        for u in nodes:
            for v in ring_to_ring_edges.get(u, set()):
                in_degree[v] += 1
                if u not in in_degree:
                    in_degree[u] = in_degree[u]  # ensure key exists
        # Queue all nodes with in_degree 0 (smallest rings)
        q = deque([u for u in nodes if in_degree[u] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in ring_to_ring_edges.get(u, set()):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    q.append(v)

        # If topological sort covered all rings, use it; otherwise fall back to count-based ordering
        if len(topo) == len(nodes):
            self.rings_by_size = topo
        else:
            # Fallback: sort by how many objects a ring is smaller than (descending) -> approx smallest to largest
            smaller_count = defaultdict(int)
            for u in nodes:
                for v in ring_to_ring_edges.get(u, set()):
                    smaller_count[u] += 1
            self.rings_by_size = sorted(list(nodes), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the current support object (OBJ) that 'ring' is on by scanning its indexed 'on' facts
        and checking which is true via bitwise membership: (state >> fact.global_id) & 1.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, tgt in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Compute an admissible lower bound using the largest-misplaced-ring rule:
        - Iterate rings from largest to smallest (reverse of smallest->largest ordering)
        - Find the first ring whose current position != goal position
        - If that ring is at index i (0-based from smallest), then at least (2^(i+1) - 1) moves are still required
        - If all rings match goals, return 0
        """
        if not self.rings_by_size:
            return 0

        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # If the ring has no explicit goal, we cannot demand moves for it
                continue
            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                # i is 0-based idx among rings_by_size (smallest at 0)
                k = i + 1
                return (1 << k) - 1
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT5Heuristic5(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic5 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic5 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
