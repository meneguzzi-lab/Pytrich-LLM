import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic16(Heuristic):
    """Domain-dependent heuristic for the HDDL 'towers' (Towers of Hanoi) domain.

    Rationale / domain insights
    --------------------------
    The domain encodes Towers of Hanoi with:
      - Primitive action: move(r, o1, t1, o2, t2)
      - Abstract tasks: shiftTower / rotateTower / exchange / move_abstract ...

    Even though the search is HTN, the real cost driver is the number of primitive
    'move' actions required to place rings into their goal configuration.

    Key property (classic Hanoi lower bound):
      If the largest ring that is not yet in its goal position has index k (counting
      from smallest=1), then at least (2^k - 1) moves are required.

    Why this works well here:
      - Training instances represent full stack goals (on r1 r2, ..., on rN t3).
      - The only way to reposition ring rk is to clear all smaller rings first.
      - This lower bound is admissible and typically very informative, thus reducing
        node expansions dramatically.

    Implementation notes (Pytrich-specific)
    --------------------------------------
    - We must NOT use model.state_explicit_repr(state) inside __call__ (too slow).
    - We precompute, for each ring, the list of grounded '+on[ring,target]' facts
      (Fact objects) to test with fast bitwise checks.
    - Goals are stored in model.goals as a bitwise integer; we extract goal 'on' facts
      once during initialize().

    The heuristic returned is:
      0 if all rings are at their goal support;
      otherwise (2^k - 1) where k is the number of smallest rings up to the largest
      ring that is misplaced.

    This heuristic is admissible and consistent for unit-cost move actions.
    """

    def __init__(self, towers_gpt52_heuristic_16=None, name="TowersGPT52Heuristic16"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0
        self.model: Model | None = None

        # Preprocessed domain info
        self.goal_positions: dict[str, str] = {}          # ring -> target object in goal
        self.on_facts_by_ring: dict[str, list[tuple]] = {}  # ring -> [(Fact, target), ...]
        self.rings_by_size: list[str] = []               # smallest -> largest
        self.num_rings: int = 0

    # --------- parsing / preprocessing ---------
    def _parse_fact_name(self, name: str):
        """Parse grounded fact name.

        Expected grounded format: "+predicate[arg1,arg2]" or "-predicate[...]".
        Returns (predicate, [args...]) or (None, []).
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

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._h(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        assert self.model is not None

        # 1) Extract goal ring->target from model.goals (bitwise)
        if getattr(self.model, "goals", None):
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2) Collect rings and all possible '+on[ring,*]' grounded facts (Fact objects)
        rings = set()
        smaller_pairs = []  # (smaller, larger)

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "on" and len(args) == 2:
                r, target = args
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, target))

            elif pred == "smallerThan" and len(args) == 2:
                a, b = args
                # smallerThan relates rings and towers (static)
                if a.startswith("r"):
                    rings.add(a)
                    if b.startswith("r"):
                        rings.add(b)
                    smaller_pairs.append((a, b))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine ring order (smallest -> largest)
        # Use smallerThan counts: smallest ring is "smallerThan" many others.
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for a, b in smaller_pairs:
                if a in rings:
                    smaller_count[a] += 1
            # More "smallerThan" relations => smaller ring
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: lexical r1,r2,...
            def keyf(x: str):
                try:
                    return int(x[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(rings, key=keyf)

    # --------- fast state queries ---------
    def _current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object this ring is currently on, using bitwise checks."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # --------- heuristic computation ---------
    def _h(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Check from largest to smallest for the first ring not at goal support.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_target = self.goal_positions.get(ring)
            if goal_target is None:
                # If no explicit goal for this ring, treat it as already satisfied.
                continue
            cur_target = self._current_support(state_bitwise, ring)
            if cur_target != goal_target:
                k = i + 1  # number of smallest rings up to this (misplaced) ring
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        hval = self._h(node.state)
        super().update_info(hval)
        return hval

    def __repr__(self):
        return f"TowersGPT52Heuristic16(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic16 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic16 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions (ring->support): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
