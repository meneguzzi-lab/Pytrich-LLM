import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic6(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers (Towers of Hanoi) domain.

    Rationale / domain insights
    ---------------------------
    The domain encodes Towers of Hanoi with a single primitive operator `move` and
    hierarchical tasks that recursively implement the optimal strategy.

    Key observation (classic Hanoi lower bound):
      If the largest ring that is not yet in its goal position has index i (counting
      from smallest=0), then at least 2^(i+1)-1 moves remain.

    Why this works well here:
    - In every valid state, to move a ring r_k (k-th smallest) you must clear all
      smaller rings above it; that inherently costs at least 2^k - 1 moves in total
      before r_k can be placed correctly.
    - This provides a tight admissible lower bound and scales excellently up to 20 rings.

    Implementation details for Pytrich:
    - Goals must be obtained from `model.goals` (bitset) via `state_explicit_repr`.
    - State checks must be bitwise using fact.global_id.
    - We preprocess:
        * ring size order from static `smallerThan` facts
        * current position lookup tables from all grounded `on[r, x]` facts
        * goal position per ring from goal `+on[r, x]` facts

    The heuristic returned is:
        h(state) = 0                          if all rings are at goal positions
                 = 2^(k) - 1                 where k = (#rings up to largest misplaced)

    Notes:
    - Admissible and consistent for unit-cost moves.
    - Very low per-node overhead (O(#rings * avg_on_facts_per_ring)).
    """

    def __init__(self, towers_gpt52_heuristic_6=None, name="TowersGPT52Heuristic6"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        self.model: Model = None

        # Preprocessed structures
        self.goal_positions = {}            # ring -> target obj
        self.rings_by_size = []             # smallest -> largest
        self.on_facts_by_ring = {}          # ring -> list[(Fact, target_obj)]
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_h(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """Parse a grounded fact name.

        Expected grounded format:
          +predicate[arg1,arg2]  or -predicate[arg1,arg2]

        Also tolerates HDDL-like format (predicate arg1 arg2) in case of different printers.
        """
        if not name:
            return None, []
        name = name.strip()

        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                pred = name[:b]
                args_str = name[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
            return None, []

        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if not parts:
                return None, []
            return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        # 1) Extract goal positions from model.goals (bitset)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = target

        rings = set(self.goal_positions.keys())
        smaller_pairs = []

        # 2) Scan all grounded facts once: gather ring universe, ordering, and on-fact lookup tables
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "smallerThan" and len(args) >= 2:
                a, b = args[0], args[1]
                # Only rings participate in size ordering that matters
                if a.startswith("r"):
                    rings.add(a)
                    smaller_pairs.append((a, b))
                    if b.startswith("r"):
                        rings.add(b)

            elif pred == "on" and len(args) >= 2:
                r, target = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Derive size order (smallest -> largest).
        # We compute, for each ring, how many other rings/objects it is smaller than.
        # In the training instances, smallerThan is dense (ri < rj for i<j), so counts work well.
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for a, b in smaller_pairs:
                if a in rings:
                    smaller_count[a] += 1
            # higher count => smaller (smaller than more things)
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by numeric suffix if present: r1 < r2 < ...
            def _ring_key(r):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9
            self.rings_by_size = sorted(list(rings), key=_ring_key)

    def _current_position(self, state_bitwise: int, ring: str):
        """Return current support object of ring by testing which +on[ring, x] is true."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_h(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Find largest ring (iterate from largest to smallest) that is not at its goal position.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal = self.goal_positions.get(ring)
            if goal is None:
                # If goal doesn't mention a ring, ignore it (robustness for partial goals)
                continue
            cur = self._current_position(state_bitwise, ring)
            if cur != goal:
                k = i + 1  # number of rings from smallest..this ring inclusive
                return (1 << k) - 1
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_h(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic6(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic6 with {self.num_rings} rings"

    def __output__(self):
        return (
            "TowersGPT52Heuristic6 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions (from +on in model.goals): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
