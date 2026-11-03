import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic15(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers domain (Towers of Hanoi).

    Rationale (domain insights):
    - The domain encodes Towers of Hanoi using a single primitive operator `move`.
    - All higher-level tasks (shiftTower/selectDirection/rotateTower/exchange/move_abstract)
      deterministically reduce to sequences of `move` operations.
    - For standard Hanoi, the optimal number of moves to relocate a stack of k smallest disks
      (rings) is 2^k - 1.

    Heuristic design:
    - Extract the goal configuration from `model.goals` (bitset) by reading the goal `on` facts.
    - Preprocess ring ordering (smallest -> largest) from static `smallerThan` facts.
    - For a given state, find the largest ring that is not in its goal position.
      If that ring has index i (0-based in smallest->largest), then at least 2^(i+1)-1 moves
      are required (because all smaller rings must be moved out of the way to place that ring).

    Properties:
    - Admissible lower bound for the classic Hanoi structure present in this domain.
    - Very fast during search: only bitwise checks on a small set of pre-indexed facts.
    - Scales well up to 20 rings (and beyond), avoiding expensive explicit state decoding.

    Notes:
    - The heuristic ignores the remaining abstract task network explicitly; in this domain the
      HTN essentially enforces the Hanoi procedure, and the cost is dominated by required moves.
      Using state-based Hanoi lower bound still provides strong guidance.
    """

    def __init__(self, towers_gpt52_heuristic_15=None, name="TowersGPT52Heuristic15"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Filled in preprocessing
        self.model: Model = None
        self.goal_positions = {}          # ring -> goal support (ring or tower)
        self.on_facts_by_ring = {}        # ring -> list[(Fact, support)]
        self.rings_by_size = []           # smallest -> largest
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._hanoi_lower_bound(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact names: '+pred[a,b]' / '-pred[a,b]'. Also tolerates '(pred a b)'."""
        name = name.strip()
        if not name:
            return None, []

        if name[0] in "+-":
            raw = name[1:]
            b = raw.find("[")
            if b > 0 and raw.endswith("]"):
                pred = raw[:b]
                args_str = raw[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
            return None, []

        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        """One-time preprocessing.

        - Build ring set and ordering from smallerThan.
        - Index all dynamic on(r, x) facts per ring for fast position lookup.
        - Extract goal positions from model.goals (bitset) using state_explicit_repr.
        """
        rings = set()
        smaller_pairs = []

        # Goal positions from model.goals (bitwise integer)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, sup = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = sup

        # Scan all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    rings.add(a)
                    smaller_pairs.append((a, b))
                    if b.startswith("r"):
                        rings.add(b)

            elif pred == "on":
                r, sup = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, sup))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # Order rings smallest -> largest.
        # Use: count how many things each ring is smaller than (more => smaller).
        smaller_count = defaultdict(int)
        for a, b in smaller_pairs:
            if a in rings:
                smaller_count[a] += 1

        # If smallerThan info is missing/weak, fall back to lexical order r1,r2,...
        if any(smaller_count.values()):
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            def ring_key(x: str):
                # r12 -> 12, else large
                if x.startswith("r") and x[1:].isdigit():
                    return int(x[1:])
                return 10**9
            self.rings_by_size = sorted(list(rings), key=ring_key)

    def _current_support(self, state_bitwise: int, ring: str):
        """Return the current support object for a ring by checking which on-fact is true."""
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, sup in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return sup
        return None

    def _hanoi_lower_bound(self, state_bitwise: int) -> int:
        """Classic Hanoi admissible lower bound: 2^k - 1 where k is number of smallest rings
        up to (and including) the largest ring not at its goal support.
        """
        if not self.rings_by_size:
            return 0

        # Iterate largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            goal_sup = self.goal_positions.get(r)
            if goal_sup is None:
                # If goal doesn't constrain this ring, ignore it (keeps heuristic safe/admissible)
                continue
            cur_sup = self._current_support(state_bitwise, r)
            if cur_sup != goal_sup:
                k = i + 1
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._hanoi_lower_bound(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic15(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic15 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal positions extracted: {len(self.goal_positions)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.6f}s\n"
            f"\tCalls: {self.calls}\n"
            f"\tMin h: {self.min_hvalue}\n"
        )
"
}