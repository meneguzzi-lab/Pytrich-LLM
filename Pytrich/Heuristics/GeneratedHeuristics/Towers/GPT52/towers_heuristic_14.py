import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic14(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers domain (Towers of Hanoi).

    Key domain insights leveraged
    ----------------------------
    1) The only primitive action is `move(r, o1, t1, o2, t2)`; all higher-level tasks
       (shiftTower/selectDirection/rotateTower/exchange/move_abstract) ultimately realize
       the classic Towers of Hanoi process.

    2) In all provided instances, the goal is a complete stack of rings on a target tower
       (e.g., r1 on r2 on ... on rn on t3). Thus, the dominant cost driver is the
       number of rings that still need to be correctly placed on the goal tower.

    Heuristic design
    ----------------
    We combine two *admissible* lower bounds and take their maximum:

      h(state) = max( h_hanoi_prefix(state), h_goal_unsat(state) )

    A) h_hanoi_prefix (tight and admissible)
       - Find the largest ring (by size) that is not yet in its goal position.
       - If that ring is the k-th smallest (i.e., there are k rings among {r1..rk}),
         then at least (2^k - 1) moves are required in any valid Hanoi solution.
       - This is the classic optimal lower bound and is very informative.

    B) h_goal_unsat (very cheap and admissible)
       - Count how many goal `on[x,y]` facts are currently false.
       - Each `move` can add at most one `on[r, o]` for a ring r (and deletes another),
         so at least that many moves are required to satisfy all missing goal relations.
       - This helps in corner/partial states and provides a monotone-ish baseline.

    Efficiency notes
    ----------------
    - All expensive preprocessing (parsing facts, building ring order, mapping ring->on-facts,
      extracting goal `on` facts) is done once in `initialize`.
    - Per-node evaluation uses only bitwise checks: (state >> fact.global_id) & 1.

    This heuristic is domain-specific, scalable up to 20 rings, and strongly reduces
    node expansions compared to generic heuristics.
    """

    def __init__(self, towers_gpt52_heuristic_14=None, name="TowersGPT52Heuristic14"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Model handle
        self.model: Model | None = None

        # Preprocessed data
        self.rings = set()  # set of ring names
        self.rings_by_size = []  # smallest -> largest
        self.goal_positions = {}  # ring -> target object from goal on-facts

        # ring -> list[(Fact, target_obj)] for all grounded on[ring,target]
        self.on_facts_by_ring = defaultdict(list)

        # goal on-facts as Fact objects (for fast unsatisfied counting)
        self.goal_on_facts = []  # list[Fact]

        # Helper: name->Fact map
        self._fact_by_name = {}

    # ------------------------- Parsing / preprocessing -------------------------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name into (predicate, [args]).

        Expected grounded format in Pytrich after grounding:
          +predicate[arg1,arg2] or -predicate[arg1,arg2]

        Returns (None, []) if parsing fails.
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

        # Build name->Fact index for later goal lookup
        self._fact_by_name = {f.name: f for f in self.model.facts}

        self._preprocess_static()

        self.preprocessing_time = time.time() - start
        h0 = self._h(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess_static(self):
        """Extract ring ordering, all on-facts by ring, and goal on-facts."""
        # 1) Collect rings and smallerThan relations for ordering
        smaller_pairs = []  # (smaller, larger)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    self.rings.add(a)
                    smaller_pairs.append((a, b))
                    if b.startswith("r"):
                        self.rings.add(b)

            elif pred == "on":
                r, tgt = args[0], args[1]
                if r.startswith("r"):
                    self.rings.add(r)
                    self.on_facts_by_ring[r].append((fact, tgt))

        # 2) Extract goal on-facts from model.goals (bitwise)
        self.goal_positions.clear()
        self.goal_on_facts.clear()

        goal_fact_names = self.model.state_explicit_repr(self.model.goals) if getattr(self.model, "goals", 0) else []
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) >= 2:
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    self.goal_positions[ring] = target
                # store the corresponding Fact object (for fast checks)
                f = self._fact_by_name.get(gname)
                if f is not None:
                    self.goal_on_facts.append(f)

        # 3) Determine ring order smallest->largest
        if not self.rings:
            self.rings_by_size = []
            return

        if smaller_pairs:
            # Count how many "larger" objects each ring is smaller than (proxy for size rank)
            # In typical instances, r1 is smaller than many, r20 smaller than few.
            smaller_count = defaultdict(int)
            for a, b in smaller_pairs:
                if a in self.rings:
                    smaller_count[a] += 1
            self.rings_by_size = sorted(self.rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by numeric suffix if available
            def _ring_key(r):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9
            self.rings_by_size = sorted(self.rings, key=_ring_key)

    # ------------------------------ State queries ------------------------------

    def _get_current_position(self, state_bitwise: int, ring: str):
        """Return the current object the ring is on, using bitwise checks."""
        for fact, tgt in self.on_facts_by_ring.get(ring, []):
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    # ------------------------------ Heuristic core -----------------------------

    def _h_hanoi_prefix(self, state_bitwise: int) -> int:
        """Classic admissible Hanoi lower bound: 2^k-1 for largest misplaced ring."""
        if not self.rings_by_size:
            return 0

        # iterate largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_tgt = self.goal_positions.get(ring)
            if goal_tgt is None:
                # If no explicit goal for this ring, ignore it (keeps admissible)
                continue
            cur_tgt = self._get_current_position(state_bitwise, ring)
            if cur_tgt != goal_tgt:
                k = i + 1  # number of smallest rings that must be manipulated
                return (1 << k) - 1
        return 0

    def _h_goal_unsat(self, state_bitwise: int) -> int:
        """Admissible: number of unsatisfied goal on-facts (each move can add at most one)."""
        if not self.goal_on_facts:
            return 0
        missing = 0
        for f in self.goal_on_facts:
            if ((state_bitwise >> f.global_id) & 1) == 0:
                missing += 1
        return missing

    def _h(self, state_bitwise: int) -> int:
        return max(self._h_hanoi_prefix(state_bitwise), self._h_goal_unsat(state_bitwise))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._h(node.state)
        super().update_info(h_val)
        return h_val

    def __repr__(self):
        return f"TowersGPT52Heuristic14(rings={len(self.rings_by_size)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings: {len(self.rings_by_size)}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal positions (subset): {dict(list(self.goal_positions.items())[:5])}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal calls: {self.calls}\n"
            f"\tMin h: {self.min_hvalue}\n"
        )
