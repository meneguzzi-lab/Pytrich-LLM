"""\
Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

Heuristic design rationale (domain insights)
-------------------------------------------
The provided HDDL encodes Towers of Hanoi in a hierarchical way:
- The only primitive action is `move(r, o1, t1, o2, t2)`.
- All abstract tasks (`shiftTower`, `selectDirection`, `rotateTower`, `exchange`, `move_abstract`)
  eventually refine to sequences of `move` actions that implement the classic optimal Hanoi algorithm.

Key structural challenge:
- To place a larger ring correctly, all smaller rings above it must be moved away first.
  This creates an exponential lower bound in the number of rings that still need rearrangement.

Heuristic used here:
- We compute the *largest* ring (by size order) that is not currently on its goal support.
- If that ring is the i-th ring from the top (counting smallest as index 0), then at least
  2^(i+1) - 1 moves are required. This is the classic optimal lower bound and is admissible.

Why this works well with HTN search:
- Although the search space is hierarchical, the true cost is determined by the number of
  primitive `move` actions needed. This heuristic estimates that remaining primitive cost
  directly from the current world state (bitwise) and the goal on-relations.

Efficiency considerations:
- All expensive parsing and indexing is done once in `initialize`.
- During search, evaluating a node uses only O(#rings) bitwise checks, without calling
  `state_explicit_repr` on the current state.

Implementation notes for Pytrich:
- Grounded fact names use the format: +predicate[arg1,arg2] (or -predicate[...]).
- Goal facts are encoded in `model.goals` as a bitwise integer; we decode them once.
- State membership is checked by: (state >> fact.global_id) & 1

"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic13(Heuristic):
    """Admissible, domain-dependent heuristic for the Towers domain.

    Returns the classic Hanoi lower bound (2^k - 1) where k is the number of smallest rings
    up to and including the largest ring that is not yet in its goal position.
    """

    def __init__(self, towers_gpt52_heuristic_13=None, name="TowersGPT52Heuristic13"):
        super().__init__(name=name)

        # Timing
        self.preprocessing_time = 0.0

        # Preprocessed data
        self.model: Model | None = None
        self.goal_positions: dict[str, str] = {}           # ring -> goal support object
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> [(Fact, target), ...]
        self.rings_by_size: list[str] = []                # smallest -> largest
        self.num_rings: int = 0

    # ------------------------- Parsing helpers -------------------------
    def _parse_fact_name(self, name: str):
        """Parse grounded fact names.

        Expected grounded format:
          +predicate[arg1,arg2]  or  -predicate[arg1,arg2]

        Returns:
          (predicate, [args...]) or (None, []) if parsing fails
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    # ------------------------- Preprocessing -------------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        """Extract rings, size order, goal supports, and fast indices for `on` facts."""
        assert self.model is not None

        rings: set[str] = set()
        smaller_than_pairs: list[tuple[str, str]] = []

        # 1) Extract goal supports from model.goals bitmask.
        # Goal in training set is conjunction of (on ...), so we read those.
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2) Index all `on` facts by ring, and collect `smallerThan` information for ordering.
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

            elif pred == "smallerThan":
                smaller, larger = args[0], args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_pairs.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine ring order (smallest -> largest).
        # Prefer explicit smallerThan graph if present.
        if smaller_than_pairs:
            # Count how many objects each ring is smaller than (in this domain instances,
            # smaller rings have more outgoing smallerThan facts).
            smaller_count = defaultdict(int)
            for s, _l in smaller_than_pairs:
                if s in rings:
                    smaller_count[s] += 1
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer order from goal chain (on r1 r2, on r2 r3, ...)
            # Build "is smaller than" closure from goal on-relations.
            is_smaller = defaultdict(set)
            for ring, target in self.goal_positions.items():
                if target.startswith("r"):
                    is_smaller[ring].add(target)

            changed = True
            while changed:
                changed = False
                for a in list(is_smaller.keys()):
                    newset = set(is_smaller[a])
                    for b in list(is_smaller[a]):
                        newset |= is_smaller.get(b, set())
                    if len(newset) != len(is_smaller[a]):
                        is_smaller[a] = newset
                        changed = True

            smaller_count = {r: len(is_smaller.get(r, set())) for r in rings}
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))

    # ------------------------- Fast state queries -------------------------
    def _get_current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object the ring is currently on, using bitwise checks."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ------------------------- Heuristic computation -------------------------
    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """Compute 2^k-1 for the largest misplaced ring."""
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If the ring has no explicit goal support, ignore it (conservative).
                continue
            cur_support = self._get_current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1  # number of rings that must be moved at least once in classic Hanoi
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic13(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic13 with {self.num_rings} rings"

    def __output__(self):
        return (
            "TowersGPT52Heuristic13 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions (from goal on-facts): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}