import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic7(Heuristic):
    """Domain-dependent heuristic for the HDDL *towers* domain (Towers of Hanoi).

    Rationale / domain insights
    --------------------------
    This domain encodes Hanoi via a hierarchy:
      shiftTower -> selectDirection (recurses down the stack via on)
      selectDirection -> rotateTower
      rotateTower -> move_abstract + exchange
      move_abstract -> move (primitive)

    The true cost is the number of primitive `move` actions.

    Key observation (classic Hanoi lower bound):
      To move a ring r that is not yet in its goal position, all smaller rings
      above it must be moved away at least once, which implies a minimum of
        2^k - 1
      moves, where k is the number of rings up to and including r (from smallest).

    Heuristic implemented
    ---------------------
    1) Preprocess (once):
       - Extract all rings and a size order (smallest -> largest) from static smallerThan.
       - Extract each ring's goal support object from the goal state's +on facts.
       - Build a fast lookup: for each ring, all grounded +on[ring,*] facts.
    2) For a given state:
       - Find the *largest* ring that is not on its goal support.
       - Return (2^(index+1) - 1).
       - If all rings are at goal supports, return 0.

    Properties
    ----------
    - Admissible and consistent for Hanoi-style instances in this domain.
    - Very fast per-call: only bit checks on a small set of candidate facts.
    - Scales well up to 20 rings (and beyond).

    Notes on Pytrich integration
    ----------------------------
    - Goals must be read from `model.goals` (bitwise) using `model.state_explicit_repr`.
    - State checks must be done via bit operations using fact.global_id.
    """

    def __init__(self, towers_gpt52_heuristic_7=None, name="TowersGPT52Heuristic7"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0
        self.model: Model | None = None

        # Preprocessed structures
        self.goal_positions: dict[str, str] = {}          # ring -> goal support object from goal +on
        self.rings_by_size: list[str] = []               # smallest -> largest
        self.on_facts_by_ring: dict[str, list[tuple]] = {}  # ring -> [(fact_obj, target_str), ...]
        self.num_rings: int = 0

    # ---------- Framework hooks ----------

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._h_state(initial_node.state)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._h_state(node.state)
        super().update_info(h)
        return h

    # ---------- Parsing helpers ----------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact names: +pred[a,b] / -pred[a,b].

        Returns: (predicate, [args...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                pred = name[:b]
                args_str = name[b + 1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        return None, []

    # ---------- Preprocessing ----------

    def _preprocess(self):
        assert self.model is not None

        rings: set[str] = set()
        smaller_pairs: list[tuple[str, str]] = []  # (smaller, larger)

        # 1) Extract ring goal supports from the *goal state* (+on facts)
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = target
                        rings.add(r)
                        if target.startswith("r"):
                            rings.add(target)

        # 2) Scan all grounded facts once to collect:
        #    - smallerThan relations (static)
        #    - all possible on facts for each ring (for fast position query)
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

        # 3) Determine a size order smallest -> largest.
        #    In these instances, smallerThan[ri,rj] exists for all i<j.
        #    The smallest ring has the highest "smaller-than count".
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for a, _b in smaller_pairs:
                smaller_count[a] += 1
            # Higher count => smaller ring
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: lexicographic; still safe but weaker.
            self.rings_by_size = sorted(list(rings))

    # ---------- State queries ----------

    def _current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object that `ring` is currently on (support), or None if unknown."""
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, target in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ---------- Heuristic computation ----------

    def _h_state(self, state_bitwise: int) -> int:
        """Classic Hanoi lower bound: find largest misplaced ring => 2^k - 1."""
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If a ring has no explicit goal support, ignore it (keeps admissibility).
                continue
            cur_support = self._current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1
                return (1 << k) - 1

        return 0

    # ---------- Reporting ----------

    def __repr__(self):
        return f"TowersGPT52Heuristic7(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic7 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic7 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal supports (from goal +on): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
