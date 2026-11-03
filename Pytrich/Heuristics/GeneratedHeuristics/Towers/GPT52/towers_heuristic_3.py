"""\
Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

Heuristic rationale (domain insights)
----------------------------------
The given HDDL domain encodes Towers of Hanoi through an HTN that ultimately
reduces to the single primitive operator `move`.

Key structural facts:
1) The *only* primitive cost comes from `move`. Every abstract task decomposition
   ultimately produces a sequence of `move` actions.
2) To move a larger ring, all smaller rings above it must be moved away first
   (enforced by `towerTop` and `smallerThan`).
3) The optimal number of moves needed to correctly place the top k smallest rings
   relative to a target configuration is 2^k - 1 (classic Towers of Hanoi).

Thus, a very strong admissible lower bound is:
- Find the largest ring that is not already in its goal position.
- Let k be the number of rings from the smallest up to that ring.
- Return 2^k - 1.

This is admissible and consistent in this domain because:
- Any plan that changes the position of that largest misplaced ring must first
  relocate all smaller rings at least once in a legal sequence.
- The minimum number of `move`s required to do that is exactly 2^k - 1.

Implementation notes (Pytrich-specific)
--------------------------------------
- Goal facts must be extracted from `model.goals` via `model.state_explicit_repr`.
- State membership tests must use bitwise operations on fact.global_id.
- We preprocess static mappings:
  * goal_positions: ring -> goal support object (from goal state's +on facts)
  * on_facts_by_ring: ring -> list of (Fact, target_obj)
  * rings_by_size: smallest -> largest (from smallerThan graph/counts)

This keeps __call__ extremely fast and scales to 20 rings.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic3(Heuristic):
    """Admissible, near-optimal heuristic for the Towers domain (Hanoi)."""

    def __init__(self, towers_gpt52_heuristic_3=None, name="TowersGPT52Heuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Filled in initialize/_preprocess
        self.model: Model | None = None

        self.goal_positions: dict[str, str] = {}      # ring -> target object in goal
        self.on_facts_by_ring: dict[str, list] = {}   # ring -> list[(Fact, target_obj)]
        self.rings_by_size: list[str] = []            # smallest -> largest
        self.num_rings: int = 0

    # ---------- Parsing helpers ----------
    def _parse_fact_name(self, name: str):
        """Parse grounded fact name into (predicate, args).

        Handles grounded format used by Pytrich after grounding:
          +predicate[arg1,arg2] / -predicate[arg1,arg2]
        Also tolerates HDDL-like format:
          (predicate arg1 arg2)
        """
        name = name.strip()

        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                predicate = name[:b]
                args_str = name[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    # ---------- Preprocessing ----------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        assert self.model is not None

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []

        # 1) Extract goal positions from the *actual goal bitset* (critical in Pytrich)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2) Index all relevant grounded facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                # In this domain, rings are named r1..rn; towers are t1..t3
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

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Derive a size order (smallest -> largest)
        # Use smallerThan relationships: in the training set, r_i smallerThan r_j for i<j.
        if smaller_pairs:
            # Count how many objects each ring is smaller than.
            # The smallest ring is smaller than the most others.
            count_smaller_than = defaultdict(int)
            for a, b in smaller_pairs:
                if a in rings:
                    count_smaller_than[a] += 1
            self.rings_by_size = sorted(rings, key=lambda r: -count_smaller_than.get(r, 0))
        else:
            # Fallback: sort by numeric suffix if available
            def _num(x: str):
                try:
                    return int(x[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(rings, key=_num)

    # ---------- Fast state queries ----------
    def _get_current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object the ring is currently on by checking true on-facts."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ---------- Heuristic computation ----------
    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest to find the largest ring not at its goal support.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If ring has no goal constraint, ignore it (robustness).
                continue

            cur_support = self._get_current_support(state_bitwise, ring)
            if cur_support != goal_support:
                # Need to (at least) move this ring; implies moving all smaller rings too.
                k = i + 1  # number of smallest rings involved
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT52Heuristic3(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic3 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic3 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}