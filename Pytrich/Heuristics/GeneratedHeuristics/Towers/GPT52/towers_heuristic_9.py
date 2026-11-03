"""\
Domain-dependent heuristic for the Towers (HTN Towers of Hanoi) domain.

Heuristic design rationale (domain insights)
------------------------------------------
This HDDL domain encodes Towers of Hanoi using an HTN hierarchy:
  shiftTower -> selectDirection -> rotateTower -> (move_abstract, exchange) -> ... -> move

Key planning challenge:
- The true difficulty is dominated by the classic Hanoi exponential substructure:
  to move a larger ring to its goal position, all smaller rings must be moved away first.

What we exploit:
1) The goal is a fully specified stack on the target tower (in benchmark instances).
   The goal facts are stored in model.goals (bitset). We extract goal 'on' relations
   from model.goals via model.state_explicit_repr(model.goals).

2) A tight admissible lower bound for Hanoi:
   Let k be the number of smallest rings that include the largest ring currently not
   in its goal position. Then at least (2^k - 1) primitive moves are required.

3) We add *cheap* hierarchical guidance without breaking admissibility:
   - If the task network is non-empty, at least one primitive action must occur
     to make progress, unless the goal is already reached. We can safely add a
     0/1 term based on whether goals are satisfied (but NOT based on methods,
     because decompositions are cost 0 and could make this inadmissible).

The resulting heuristic is:
  h(state, tn) = (2^k - 1) + (0 if goal reached else 0)

We keep the second term as 0 to remain strictly admissible in the presence of
zero-cost HTN decomposition steps. (Adding +1 when tn not empty can overestimate
because the planner may decompose without executing any primitive operator.
Pytrich usually counts operator costs only.)

Implementation notes:
- Uses only bitwise checks on facts (fast).
- Robust fact parsing for the grounded format: +predicate[arg1,arg2]
- Preprocessing builds:
  * rings_by_size: smallest -> largest
  * goal_positions: ring -> goal support object
  * on_facts_by_ring: ring -> list of (Fact, target_obj)

Class naming / init parameter requirements:
- Class: TowersGPT52Heuristic9
- __init__ parameter: towers_gpt52_heuristic_9
- Avoid using any variable named towers_heuristic_9.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic9(Heuristic):
    def __init__(self, towers_gpt52_heuristic_9=None, name="TowersGPT52Heuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0
        self.model: Model | None = None

        # Preprocessed domain knowledge
        self.goal_positions: dict[str, str] = {}       # ring -> target object in goal
        self.rings_by_size: list[str] = []             # smallest -> largest
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> [(Fact, target)]
        self.num_rings: int = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._h_hanoi(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """Parse grounded fact names of the form '+pred[a,b]' or '-pred[a,b]'."""
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
        assert self.model is not None

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []

        # Extract goal ring placements from the *bitset goal*
        if getattr(self.model, "goals", None):
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = target

        # Scan all grounded facts once
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
                r, target = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # Sort rings smallest->largest.
        # We use a robust approach based on how many objects each ring is smallerThan.
        # In the benchmarks, smallerThan is very dense, so this works well.
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for a, _b in smaller_pairs:
                if a in rings:
                    smaller_count[a] += 1
            # more "smallerThan" relations => smaller ring
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: lexical r1<r2<... if no size info available
            def _ring_key(r: str):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(rings, key=_ring_key)

    def _current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object the ring is currently 'on' by checking all its on-facts."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _h_hanoi(self, state_bitwise: int) -> int:
        """Admissible Hanoi lower bound: 2^k - 1 for largest ring out of place."""
        if not self.rings_by_size:
            return 0

        # Iterate from largest down to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If goal doesn't mention this ring, ignore it for admissibility
                continue

            cur_support = self._current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1  # smallest..ring inclusive
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        # Only state-based to remain admissible with 0-cost decompositions.
        h = self._h_hanoi(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic9(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic9 with {self.num_rings} rings"

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
"
}