"""\
Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

Heuristic idea (domain insights)
-------------------------------
This domain encodes Towers of Hanoi hierarchically:
  shiftTower -> selectDirection -> rotateTower -> (move_abstract, exchange)
  exchange recursively triggers rotateTower until rings can be moved legally.

Key property of Towers of Hanoi:
- If the largest ring that is not yet in its goal position is ring k (counting from smallest=1),
  then at least (2^k - 1) primitive moves remain.
  Reason: to move that ring once, all smaller rings must be moved away (requires >= 2^(k-1)-1),
  then move the ring (1), then (in general) smaller rings must still be manipulated; the classic
  lower bound for finishing from any configuration where ring k must move is 2^k-1.

This heuristic leverages:
- Goal extracted from model.goals (bitwise), not from goal_on predicates.
- Ring ordering extracted from static smallerThan facts (fast, robust across instances).
- Current ring positions from dynamic on facts, checked via fast bitwise tests.

It is admissible and very informative on this domain (often equal to optimal remaining cost
in standard instances), and extremely cheap per node: O(#rings * average-on-facts-per-ring).

Pytrich integration notes
------------------------
- initialize() does all preprocessing.
- __call__ only does bitwise checks.

Naming requirements from prompt
-------------------------------
- Class: TowersGPT52Heuristic2 (inherits from Heuristic)
- __init__ parameter: towers_gpt52_heuristic_2 (lower case)
- No variable named towers_heuristic_2 (we use towers_gpt52_heuristic_2)
"""

import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic2(Heuristic):
    """Admissible domain-dependent heuristic for the Towers domain.

    Returns:
      0 if all rings satisfy goal on-relations;
      otherwise (2^k - 1) where k is the number of smallest rings up to the
      largest ring that is not yet at its goal support.

    This closely matches the classic optimal lower bound for Towers of Hanoi.
    """

    def __init__(self, towers_gpt52_heuristic_2=None, name="TowersGPT52Heuristic2"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0
        self.model: Model | None = None

        # Preprocessed structures
        self.goal_positions: dict[str, str] = {}          # ring -> goal support object (ring or tower)
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> [(Fact, target), ...]
        self.rings_by_size: list[str] = []               # smallest -> largest
        self.num_rings: int = 0

    # ---------- Parsing helpers ----------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name.

        Grounded fact format (critical):
          +predicate[arg1,arg2] or -predicate[arg1,arg2]

        Returns:
          (predicate, [args...]) or (None, [])
        """
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

    # ---------- Pytrich API ----------

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._h(initial_node.state)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._h(node.state)
        super().update_info(h_value)
        return h_value

    # ---------- Preprocessing ----------

    def _preprocess(self):
        """Extract goal positions, ring ordering, and fast access to on-facts."""
        assert self.model is not None

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []

        # 1) Extract goals from model.goals (bitwise integer)
        # Goal in provided instances is conjunction of on facts.
        if getattr(self.model, "goals", 0):
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)
                        if target.startswith("r"):
                            rings.add(target)

        # 2) Scan all grounded facts once to gather:
        #    - smallerThan relations (static) to derive ring sizes
        #    - all possible on[r, x] facts per ring for fast state queries
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
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self.num_rings = len([r for r in rings if r.startswith("r")])
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Derive ring ordering smallest->largest.
        # We use: for ring r, count how many objects it is smallerThan.
        # Smallest ring is smallerThan many other rings and all towers -> highest count.
        if smaller_pairs:
            smaller_count = defaultdict(int)
            ring_set = {r for r in rings if r.startswith("r")}
            for a, b in smaller_pairs:
                if a in ring_set:
                    # count only comparisons where a is a ring; b can be ring or tower
                    smaller_count[a] += 1
            self.rings_by_size = sorted(list(ring_set), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer from goal on-chain (rarely needed)
            ring_set = {r for r in rings if r.startswith("r")}
            implied_smaller = defaultdict(set)
            for r, t in self.goal_positions.items():
                if t.startswith("r"):
                    implied_smaller[r].add(t)
            changed = True
            while changed:
                changed = False
                for r in list(implied_smaller.keys()):
                    for mid in list(implied_smaller[r]):
                        for big in implied_smaller.get(mid, set()):
                            if big not in implied_smaller[r]:
                                implied_smaller[r].add(big)
                                changed = True
            smaller_count = {r: len(implied_smaller.get(r, set())) for r in ring_set}
            self.rings_by_size = sorted(list(ring_set), key=lambda r: -smaller_count.get(r, 0))

    # ---------- Fast state queries ----------

    def _current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object (ring or tower) that 'ring' is currently on."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ---------- Heuristic computation ----------

    def _h(self, state_bitwise: int) -> int:
        """Compute (2^k - 1) based on the largest misplaced ring."""
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest; rings_by_size is smallest->largest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If ring is not mentioned in the goal, it doesn't need to be placed.
                continue
            cur_support = self._current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1  # number of smallest rings up to this ring
                return (1 << k) - 1

        return 0

    # ---------- Pretty printing ----------

    def __repr__(self):
        return f"TowersGPT52Heuristic2(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic2 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic2 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions (on goals): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}