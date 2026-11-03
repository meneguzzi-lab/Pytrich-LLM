"""\
Domain-dependent heuristic for the HDDL Towers (Towers of Hanoi) domain.

Key domain insight
----------------
The provided HDDL domain encodes the classic Towers of Hanoi problem.
Primitive cost is entirely in the single action `move`, while all abstract
methods ultimately refine into a sequence of `move` actions.

For Towers of Hanoi, the optimal number of moves to place k smallest disks
correctly (when the largest misplaced disk has index k) is exactly:

    h* = 2^k - 1

A tight admissible lower bound is therefore:
1) Identify the largest ring that is NOT currently on its goal support.
2) Let k be the number of rings from smallest up to that ring.
3) Return 2^k - 1.

This is admissible because to move that largest misplaced ring at least once,
all smaller rings must be moved out of the way (and in Hanoi this implies the
2^k-1 structure). This heuristic is also consistent/monotone in this domain.

Implementation notes for Pytrich
-------------------------------
- Facts are grounded and represented as strings like "+on[r1,r2]".
- Goal facts are stored as a bitset integer `model.goals`.
- State membership checks must be done with bit-operations using each Fact's
  `global_id`.

The heuristic also lightly considers the hierarchical task structure by being
an action-count lower bound for the remaining refinement cost: every abstract
refinement ultimately produces a sequence of `move` actions, so estimating
minimum remaining moves is a strong guide for HTN search.
"""

import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic0(Heuristic):
    """Optimal, domain-dependent Hanoi lower bound (2^k-1) for the Towers domain."""

    def __init__(self, towers_gpt52_heuristic_0=None, name="TowersGPT52Heuristic0"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0
        self.model: Model | None = None

        # Preprocessed domain knowledge
        self.goal_positions: dict[str, str] = {}     # ring -> goal support object (ring or tower)
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> [(Fact, target), ...]
        self.rings_by_size: list[str] = []           # smallest -> largest
        self.num_rings: int = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name into (predicate, args).

        Grounded Pytrich format:
          +predicate[arg1,arg2]  or  -predicate[arg1,arg2]
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

    def _preprocess(self):
        """Extract goal supports, ring set, ring ordering, and fast 'on' lookup tables."""
        assert self.model is not None

        rings: set[str] = set()
        smaller_than_pairs: list[tuple[str, str]] = []

        # 1) Read goal supports from model.goals (bitset)
        if getattr(self.model, "goals", 0):
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, sup = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = sup
                        rings.add(r)

        # 2) Scan all grounded facts to get:
        #    - all rings (from smallerThan/on)
        #    - ordering info from smallerThan
        #    - all possible on-facts for each ring (for fast current-position query)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    rings.add(a)
                    smaller_than_pairs.append((a, b))
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

        # 3) Determine ring order smallest->largest.
        # Prefer smallerThan: smallest has the MOST "smallerThan" relations.
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for small, _large in smaller_than_pairs:
                if small in rings:
                    smaller_count[small] += 1
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer from goal stacking (if on[r_i, r_j] in goal then r_i smaller than r_j)
            is_smaller = defaultdict(set)
            for r, sup in self.goal_positions.items():
                if sup.startswith("r"):
                    is_smaller[r].add(sup)

            changed = True
            while changed:
                changed = False
                for r in list(is_smaller.keys()):
                    for mid in list(is_smaller[r]):
                        for big in is_smaller.get(mid, set()):
                            if big not in is_smaller[r]:
                                is_smaller[r].add(big)
                                changed = True

            count = {r: len(is_smaller.get(r, set())) for r in rings}
            self.rings_by_size = sorted(rings, key=lambda r: -count.get(r, 0))

    def _get_current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object (ring or tower) that `ring` is currently on."""
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, sup in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return sup
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """Compute 2^k-1 for k = index of largest misplaced ring + 1."""
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_sup = self.goal_positions.get(ring)
            if goal_sup is None:
                # If no explicit goal for this ring, it does not constrain the problem.
                continue
            cur_sup = self._get_current_support(state_bitwise, ring)
            if cur_sup != goal_sup:
                k = i + 1  # smallest..ring inclusive
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic0(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic0 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic0 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}