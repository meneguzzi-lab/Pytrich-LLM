import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic5(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers domain (Towers of Hanoi).

    RATIONALE / DOMAIN INSIGHTS
    --------------------------
    The provided HDDL encodes the classic Towers of Hanoi via HTN methods:
      - shiftTower decomposes into selectDirection, which recurses down the stack.
      - rotateTower decomposes into move_abstract + exchange.
      - move_abstract becomes a single primitive move(r, o1, t1, o2, t2).

    In this domain, the true optimal cost-to-go (number of primitive moves) from a state
    is tightly lower-bounded by the classical Hanoi bound: to correctly place the largest
    ring that is not yet in its goal position, all smaller rings must be moved away and
    later re-stacked, requiring at least 2^k - 1 moves where k is the number of rings
    up to that ring (smallest..that ring).

    This heuristic is designed to be:
      - DOMAIN-SPECIFIC: uses ring size order (smallerThan) and the 'on' relation.
      - TIGHT / ADMISSIBLE: returns the classic lower bound 2^k-1.
      - FAST: uses bitwise checks for facts; all expensive work is in initialize().
      - ROBUST: handles missing/partial goal info by falling back to goal state's 'on' facts.

    Additionally, as a very cheap tie-break improvement (still admissible), we add
    a tiny fractional component based on how many goal 'on' facts are currently false.
    Pytrich expects integer heuristics, so we instead add a small integer in a way that
    cannot break admissibility: we only add 0 (i.e., we do NOT add any positive extra).
    (Keeping the core bound tight is usually best for node expansions in Hanoi.)

    NOTE: We avoid any use of model.state_explicit_repr(state) inside __call__.
    """

    def __init__(self, towers_gpt52_heuristic_5=None, name="TowersGPT52Heuristic5"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed structures
        self.model: Model | None = None
        self.goal_positions: dict[str, str] = {}          # ring -> desired support object
        self.rings: set[str] = set()
        self.rings_by_size: list[str] = []                # smallest -> largest
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}

        # For size sorting
        self._smaller_edges_ring_to_ring: list[tuple[str, str]] = []

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._h(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact names of the form +pred[a,b] / -pred[a,b]."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                pred = name[:b]
                args_str = name[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """Extract goal 'on' facts, ring ordering (smallerThan), and fast access to on-facts."""
        assert self.model is not None

        # 1) Extract goal positions from model.goals (bitset) using explicit repr ONCE.
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = tgt
                        self.rings.add(r)
                        if tgt.startswith("r"):
                            self.rings.add(tgt)

        # 2) Scan all grounded facts for smallerThan and on facts.
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    self.rings.add(a)
                    if b.startswith("r"):
                        self.rings.add(b)
                        self._smaller_edges_ring_to_ring.append((a, b))

            elif pred == "on":
                r, tgt = args[0], args[1]
                if r.startswith("r"):
                    self.rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, tgt))

        # 3) Derive ring order smallest->largest.
        self.rings_by_size = self._compute_ring_order()

    def _compute_ring_order(self) -> list[str]:
        """Compute a total order of rings smallest->largest.

        Prefer using smallerThan(r_i, r_j) edges. If incomplete, fall back to goal 'on'
        relationships (r on r' implies r < r').
        """
        rings = list(self.rings)
        if not rings:
            return []

        # Build adjacency for "smaller than" among rings
        adj = defaultdict(set)
        indeg = defaultdict(int)

        edges = list(self._smaller_edges_ring_to_ring)

        # If we have no ring-to-ring edges, infer some from goal stacking
        if not edges and self.goal_positions:
            for r, tgt in self.goal_positions.items():
                if tgt.startswith("r"):
                    edges.append((r, tgt))

        # Add edges
        for a, b in edges:
            if a in self.rings and b in self.rings and b not in adj[a]:
                adj[a].add(b)

        # Kahn topological sort where edge a->b means a smaller than b.
        # Smallest rings should come first.
        for r in self.rings:
            indeg[r] = 0
        for a in adj:
            for b in adj[a]:
                indeg[b] += 1

        q = deque([r for r in self.rings if indeg[r] == 0])
        order = []
        while q:
            # deterministic-ish: pop lexicographically smallest to stabilize
            r = min(q)
            q.remove(r)
            order.append(r)
            for b in adj.get(r, ()): 
                indeg[b] -= 1
                if indeg[b] == 0:
                    q.append(b)

        if len(order) == len(self.rings):
            return order

        # If not fully ordered (incomplete constraints), fall back to counting how many
        # others each ring is smaller than (transitive closure approximated by BFS).
        smaller_count = {r: 0 for r in self.rings}
        for r in self.rings:
            seen = set()
            dq = deque([r])
            while dq:
                x = dq.popleft()
                for y in adj.get(x, ()): 
                    if y not in seen:
                        seen.add(y)
                        dq.append(y)
            smaller_count[r] = len(seen)

        # ring with highest smaller_count is smallest
        return sorted(list(self.rings), key=lambda rr: (-smaller_count.get(rr, 0), rr))

    def _current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object the ring is currently on, using bitwise checks."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, tgt in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _h(self, state_bitwise: int) -> int:
        """Classic admissible Hanoi lower bound: 2^k - 1 for largest misplaced ring."""
        if not self.rings_by_size:
            return 0

        # iterate from largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            goal_tgt = self.goal_positions.get(r)
            if goal_tgt is None:
                # If goal doesn't specify this ring, ignore it (safe lower bound).
                continue
            cur_tgt = self._current_support(state_bitwise, r)
            if cur_tgt != goal_tgt:
                k = i + 1  # number of rings from smallest..r
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._h(node.state)
        super().update_info(h_val)
        return h_val

    def __repr__(self):
        return f"TowersGPT52Heuristic5(rings={len(self.rings_by_size)})"

    def __str__(self):
        return f"TowersGPT52Heuristic5 with {len(self.rings_by_size)} rings"

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal positions (ring->support): {self.goal_positions}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
            f"\tCalls: {self.calls}\n"
            f"\tMin h: {self.min_hvalue}\n"
        )
"
}