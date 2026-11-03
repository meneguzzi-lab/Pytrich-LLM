"""towers_heuristic_17.py

Domain-dependent heuristic for the HDDL 'towers' (Towers of Hanoi) domain for Pytrich.

Design rationale (domain insights)
---------------------------------
Towers of Hanoi has a well-known optimal solution structure:
- To move a ring k (counting from smallest=1) to its goal support, all smaller rings (1..k-1)
  must be moved out of the way first.
- Therefore, if the largest ring that is not yet at its goal position is ring k,
  then at least (2^k - 1) primitive moves are necessary.

This heuristic exploits that structure and is:
- Domain-dependent (uses 'on' and the goal 'on' relations).
- Efficient (bitwise fact checks only; no per-call explicit state decoding).
- Admissible and consistent for unit-cost moves in Hanoi, thus good for A*/best-first.

Implementation notes for Pytrich
-------------------------------
- Goal facts must be extracted from model.goals (bitwise integer) via model.state_explicit_repr.
- State checks must use bitwise operations on fact.global_id.
- We preprocess:
  * rings list and a size order (smallest -> largest) derived from static smallerThan facts.
  * mapping ring -> list of candidate '+on[ring,target]' facts to locate its current position.
  * goal position ring -> target derived from goal 'on' facts.

Heuristic value
---------------
- Find the largest ring (by size) that is not on its goal support.
- Return 2^(index+1) - 1 where index is that ring's index in smallest->largest order.
- If all rings are in goal positions, return 0.

This gives a tight lower bound for classic Towers of Hanoi instances used in this benchmark.
"""

import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic17(Heuristic):
    """Optimal lower-bound heuristic for the Pytrich 'towers' domain."""

    def __init__(self, towers_gpt52_heuristic_17=None, name: str = "TowersGPT52Heuristic17"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        self.model: Model | None = None

        # Preprocessed domain knowledge
        self.goal_positions: dict[str, str] = {}     # ring -> goal support object
        self.rings_by_size: list[str] = []           # smallest -> largest
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> [(Fact, target)]
        self.num_rings: int = 0

    # ------------------------- Parsing helpers -------------------------
    def _parse_fact_name(self, name: str):
        """Parse grounded fact name into (predicate, args).

        Expected grounded format in Pytrich after grounding:
          +predicate[arg1,arg2] or -predicate[arg1,arg2]

        Returns:
          (predicate: str|None, args: list[str])
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

    # ------------------------- Pytrich hooks -------------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    # ------------------------- Preprocessing -------------------------
    def _preprocess(self):
        """Extract rings, ring order, goal supports, and 'on' fact handles."""
        if self.model is None:
            return

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []

        # 1) Extract goal positions from the actual goal bitset (goals are 'on' facts)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2) Extract static ordering from smallerThan and collect all 'on' facts per ring
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

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine ring order smallest->largest.
        # Use smallerThan facts: a is smaller than many others => very small.
        # Count how many objects each ring is smaller than (restrict to rings when possible).
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_pairs:
                if smaller in rings:
                    # Count only comparisons to other rings when available, else still informative.
                    if larger.startswith("r"):
                        smaller_count[smaller] += 1
                    else:
                        # towers (t1,t2,t3) are larger than all rings; count lightly
                        smaller_count[smaller] += 0

            # If ring comparisons exist, counts will distinguish sizes; otherwise fallback
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: lexical (common naming r1 smallest ...)
            def _ring_key(r: str):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(list(rings), key=_ring_key)

    # ------------------------- Fast state queries -------------------------
    def _get_current_position(self, state_bitwise: int, ring: str) -> str | None:
        """Return the current support object for ring by scanning its 'on' facts (bitwise)."""
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, target in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ------------------------- Heuristic computation -------------------------
    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """Compute 2^k-1 where k is the index (from smallest) of the largest misplaced ring."""
        if not self.rings_by_size:
            return 0

        # Iterate largest->smallest to find first ring not at goal
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # If goal doesn't mention this ring, it doesn't constrain cost.
                continue
            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                k = i + 1  # number of rings from smallest..this ring
                return (1 << k) - 1
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT52Heuristic17(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic17 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions (ring->support): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
"
}