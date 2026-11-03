import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic8(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers domain (Towers of Hanoi).

    Rationale / domain insights
    --------------------------
    The domain encodes classical Towers of Hanoi with 3 pegs (t1,t2,t3), rings (r1..rn),
    and a single primitive action:
        move(r, o1, t1, o2, t2)
    that moves the *top* ring of a tower onto the top object of another tower (ring or tower base).

    The HTN tasks (shiftTower/selectDirection/rotateTower/exchange/move_abstract) implement
    the recursive Hanoi strategy. However, during search we still need strong guidance on
    the *remaining primitive moves*.

    This heuristic combines two admissible lower bounds and returns their maximum:

      (A) Classic Hanoi lower bound (tight for standard instances):
          Find the largest ring that is not currently in its goal place (according to goal on-facts).
          If that ring has index i in size order (0=smallest), at least 2^(i+1)-1 moves remain.

      (B) Misplaced-rings counting bound (very cheap, helps in partial/atypical states):
          Each ring not at its goal position must be moved at least once => LB = count(misplaced).

    max(A,B) remains admissible and is usually as informative as A, while being more robust
    if goal extraction is partial or intermediate states are unusual.

    Efficiency considerations
    -------------------------
    - All expensive work is done in initialize().
    - During __call__, the heuristic uses only bitwise checks against pre-indexed facts.
    - We never call model.state_explicit_repr(state) inside __call__.

    Notes
    -----
    - Goals in Pytrich are stored as a bitwise integer in model.goals; we extract goal '+on[...]'
      facts from there during preprocessing.
    - Fact names are in grounded format: '+predicate[arg1,arg2]'.
    """

    def __init__(self, towers_gpt52_heuristic_8=None, name: str = "TowersGPT52Heuristic8"):
        super().__init__(name=name)
        self.towers_gpt52_heuristic_8 = towers_gpt52_heuristic_8

        self.model: Model | None = None
        self.preprocessing_time: float = 0.0

        # Preprocessed domain info
        self.goal_positions: dict[str, str] = {}          # ring -> target obj (from goal '+on[ring,target]')
        self.rings_by_size: list[str] = []               # smallest -> largest
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = {}  # ring -> list[(Fact, target)]
        self.num_rings: int = 0

    # -------------------------
    # Parsing / preprocessing
    # -------------------------
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

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._h(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        # 1) Extract goal positions from model.goals (bitwise goal state)
        if getattr(self.model, "goals", None):
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) == 2:
                    ring, target = args
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []

        # 2) Scan all facts once to build:
        #    - on_facts_by_ring for fast current-position lookup
        #    - ring set and ordering info from smallerThan
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "on" and len(args) == 2:
                ring, target = args
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

            elif pred == "smallerThan" and len(args) == 2:
                a, b = args
                # only rings participate in size ordering
                if a.startswith("r"):
                    rings.add(a)
                    smaller_pairs.append((a, b))
                    if b.startswith("r"):
                        rings.add(b)

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine ring order smallest->largest.
        # In instances, smallerThan is dense: smallerThan(ri, rj) for i<j.
        # Then: count how many objects each ring is smaller than; larger count => smaller ring.
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for a, _b in smaller_pairs:
                if a in rings:
                    smaller_count[a] += 1
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # fallback: lexicographic (r1,r2,...) if smallerThan not present
            def _ring_key(r: str):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(rings, key=_ring_key)

    # -------------------------
    # Fast state queries
    # -------------------------
    def _current_pos(self, state: int, ring: str) -> str | None:
        """Return the current support object of `ring` by checking which +on[ring,x] is true."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state >> fact.global_id) & 1:
                return target
        return None

    # -------------------------
    # Heuristic computation
    # -------------------------
    def _h(self, state: int) -> int:
        if not self.rings_by_size:
            return 0

        # (B) Count misplaced rings (cheap, robust)
        misplaced = 0
        for ring in self.rings_by_size:
            g = self.goal_positions.get(ring)
            if g is None:
                continue
            if self._current_pos(state, ring) != g:
                misplaced += 1

        # (A) Classic Hanoi bound: largest ring not in goal position => 2^k - 1
        classic = 0
        # iterate from largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            g = self.goal_positions.get(ring)
            if g is None:
                continue
            if self._current_pos(state, ring) != g:
                k = i + 1  # number of rings up to this (smallest..ring)
                classic = (1 << k) - 1
                break

        return max(classic, misplaced)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        hval = self._h(node.state)
        super().update_info(hval)
        return hval

    def __repr__(self):
        return f"TowersGPT52Heuristic8(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic8 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic8 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
