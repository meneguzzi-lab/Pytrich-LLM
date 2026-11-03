'''
Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

Heuristic design rationale (domain insights)
------------------------------------------
This domain encodes Towers of Hanoi with:
  - Primitive action: move(r, o1, t1, o2, t2)
  - Abstract tasks: shiftTower/selectDirection/rotateTower/exchange/move_abstract

Key structural challenge:
  Any progress toward the goal ultimately requires executing primitive 'move' actions,
  and to move a larger ring, all smaller rings above it must be moved away first.
  This creates the classic exponential lower bound: to correctly place the largest
  misplaced ring, at least 2^k - 1 moves are required, where k is the count of rings
  that must be relocated (all rings smaller than or equal to that ring).

Domain-dependent heuristic used:
  1) Extract ring size order using the static predicate smallerThan.
  2) Extract each ring's goal support from the *actual goal facts* (model.goals),
     i.e., +on[ri,x].
  3) In a given state, find the largest ring that is not at its goal support.
  4) Return 2^k - 1 where k is the number of rings up to that ring (smallest..that ring).

Why this works well in Pytrich:
  - Uses bitwise checks to query the current state (fast).
  - All parsing/preprocessing is done once in initialize().
  - Gives a tight admissible lower bound for Towers instances, scaling well to 20 rings.

Notes:
  - This heuristic focuses on remaining primitive cost. In this domain, abstract task
    refinement overhead is not part of plan cost (operators have the cost); thus an
    action-count lower bound is an effective guide even for HTN search.
  - Robust to partial states: if a ring's current 'on' position can't be identified,
    it is treated as misplaced, which preserves admissibility.
'''

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic18(Heuristic):
    """Admissible, domain-dependent heuristic for the Towers domain."""

    def __init__(self, towers_gpt52_heuristic_18=None, name="TowersGPT52Heuristic18"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Filled during preprocessing
        self.model: Model | None = None
        self.goal_positions: dict[str, str] = {}        # ring -> target object in goal (+on[ring,target])
        self.rings_by_size: list[str] = []              # smallest -> largest
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> [(Fact, target), ...]
        self.num_rings: int = 0

    # ------------------------- Parsing helpers -------------------------

    def _parse_fact_name(self, name: str):
        """Parse a grounded fact name: '+pred[a,b]' / '-pred[a,b]' (also supports '(pred a b)')."""
        name = name.strip()

        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    # ------------------------- Initialization -------------------------

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time

        h0 = self._calculate(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        """Extract ring ordering, goal positions, and fast access structures."""
        assert self.model is not None

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []

        # Goals are stored as a bitwise int in model.goals; extract +on[ring,target]
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # Scan all grounded facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                # a can be a ring; b can be ring or tower
                if a.startswith("r"):
                    rings.add(a)
                    smaller_pairs.append((a, b))
                    if isinstance(b, str) and b.startswith("r"):
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

        # Determine ring order (smallest -> largest) using smallerThan
        # In standard Hanoi encoding, for ring ri: smallerThan(ri, rj) for all larger rj.
        # So the smallest ring has the highest count of 'is smaller than' relations.
        smaller_count = defaultdict(int)
        if smaller_pairs:
            for a, b in smaller_pairs:
                if a in rings:
                    smaller_count[a] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort lexicographically (keeps heuristic defined even if ordering missing)
            self.rings_by_size = sorted(list(rings))

    # ------------------------- Fast state queries -------------------------

    def _current_support(self, state_bitwise: int, ring: str):
        """Return the object a ring is currently 'on', using bitwise checks."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ------------------------- Heuristic computation -------------------------

    def _calculate(self, state_bitwise: int) -> int:
        """Classic admissible Hanoi lower bound: 2^k - 1 for largest misplaced ring."""
        if not self.rings_by_size:
            return 0

        # Iterate largest -> smallest to find the largest ring not at goal
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal = self.goal_positions.get(ring)
            cur = self._current_support(state_bitwise, ring)

            # If we cannot identify its position, conservatively treat as misplaced (still admissible)
            if cur is None or goal is None or cur != goal:
                k = i + 1  # rings[0..i]
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._calculate(node.state)
        super().update_info(h_val)
        return h_val

    def __repr__(self):
        return f"TowersGPT52Heuristic18(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic18 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal positions: {self.goal_positions}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.6f}s\n"
            f"\tCalls: {self.calls}\n"
            f"\tMin h: {self.min_hvalue}\n"
        )
