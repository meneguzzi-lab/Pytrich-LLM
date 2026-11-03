"""\
Domain-dependent heuristic for the Towers (HTN Towers of Hanoi) domain.

Key domain insight
-----------------
The primitive dynamics are exactly Towers of Hanoi:
  - A ring r can be moved iff r is the top ring on its tower and the destination
    top object is larger (smallerThan r dest_top).
  - The goal in provided instances is a full stack on target tower (e.g., r20 on t3).

The HTN methods (shiftTower/selectDirection/rotateTower/exchange/move_abstract)
implement the classic recursive algorithm. Therefore the optimal remaining
primitive moves from a state are tightly lower-bounded by:

  If the largest ring not yet in its goal position is of rank k (counting from
  smallest as 1), then at least (2^k - 1) moves are required.

This is admissible because to place that largest misplaced ring correctly, all
smaller rings must be moved out of its way at least once, and the minimal number
of moves to relocate k rings in Hanoi is 2^k - 1.

Implementation notes for Pytrich
-------------------------------
- Goal facts are stored as a bitwise integer in model.goals.
- Grounded fact strings are of the form "+pred[a,b]".
- Checking truth in a state must use bitwise tests with fact.global_id.

This heuristic focuses on state-based remaining effort. It is cheap (O(#rings))
per call and scales well up to 20 rings.
"""

import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic1(Heuristic):
    """Admissible, domain-dependent heuristic for the Towers domain.

    h(s) = 2^k - 1, where k is the number of smallest rings up to and including
    the largest ring that is not currently on its goal support.

    If all rings are on their goal supports, h(s)=0.
    """

    def __init__(self, towers_gpt52_heuristic_1=None, name="TowersGPT52Heuristic1"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed knowledge
        self.goal_positions = {}        # ring -> goal support object (from model.goals on-facts)
        self.rings_by_size = []         # [smallest, ..., largest]
        self.on_facts_by_ring = {}      # ring -> list[(Fact, target_obj)] for all +on[ring,*]
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact strings: "+pred[a,b]" or "-pred[a,b]".

        Returns (predicate, [args]) or (None, []).
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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """Extract rings, size order, all on-facts per ring, and goal support for each ring."""
        rings = set()
        smaller_than_pairs = []

        # 1) Extract goal "on" relations from the bitwise goal state.
        if getattr(self.model, "goals", 0):
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = target
                        rings.add(r)

        # 2) Scan all grounded facts to find rings, size info, and all possible on-facts.
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
                r, target = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Derive ring order smallest..largest.
        # In the training instances, smallerThan is rich: r1 < r2 < ... < rn and also ring < towers.
        # We can rank rings by how many objects they are smaller than: the smallest ring has the largest count.
        if smaller_than_pairs:
            count = defaultdict(int)
            for a, _b in smaller_than_pairs:
                if a in rings:
                    count[a] += 1
            self.rings_by_size = sorted(rings, key=lambda r: -count.get(r, 0))
        else:
            # Fallback: lexical order (works for r1..r20 naming)
            def ring_key(x: str):
                if x.startswith("r"):
                    try:
                        return int(x[1:])
                    except Exception:
                        return 10**9
                return 10**9

            self.rings_by_size = sorted(rings, key=ring_key)

    def _get_current_support(self, state_bitwise: int, ring: str):
        """Return the current support object for ring by testing +on[ring,*] facts bitwise."""
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, target in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest; first mismatch determines k.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If goal doesn't mention this ring, we ignore it (safe lower bound).
                continue
            cur_support = self._get_current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1  # number of rings from smallest up to this ring
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic1(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic1 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic1 Info:\n"
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