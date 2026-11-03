import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic12(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers domain (HTN Towers of Hanoi).

    Rationale / domain insights
    --------------------------
    The provided HDDL encodes Towers of Hanoi with HTN methods:
      shiftTower -> selectDirection -> rotateTower -> (move_abstract, exchange) -> ... -> move

    The only primitive action is `move`, which moves exactly one ring.
    Therefore, the true cost-to-go equals the remaining number of ring moves.

    Key insight for a tight admissible lower bound:
      In Towers of Hanoi, if the largest ring that is not yet in its goal position has
      index k (counting from smallest=1), then at least (2^k - 1) moves are required
      from ANY state to place all rings 1..k correctly (because that largest ring
      cannot be moved unless all smaller rings are moved out of the way).

    This heuristic computes:
      - the goal support object of each ring from `model.goals` (bitwise goal state)
      - a size order of rings from static `smallerThan` facts
      - the current support object of each ring using fast bitwise checks over grounded `on` facts

    Then it returns:
      h(s) = 0 if all rings are on their goal supports
      else h(s) = 2^(#rings up to largest misplaced ring) - 1

    Properties
    ----------
    - Admissible and consistent for unit-cost `move`.
    - Very fast per node: O(#rings + #on-facts-per-ring) but typically O(#rings).
    - Scales well up to 20 rings (and beyond) due to pure bit operations.

    Notes for Pytrich
    -----------------
    - NEVER use model.state_explicit_repr(state) inside __call__.
    - Goals must be read from model.goals using model.state_explicit_repr(model.goals) once.

    """

    def __init__(self, towers_gpt52_heuristic_12=None, name="TowersGPT52Heuristic12"):
        super().__init__(name=name)
        self.model: Model | None = None
        self.preprocessing_time = 0.0

        # Preprocessed structures
        self.goal_positions: dict[str, str] = {}          # ring -> goal support obj (ring or tower)
        self.on_facts_by_ring: dict[str, list] = {}       # ring -> list[(Fact, target_obj)]
        self.rings_by_size: list[str] = []                # smallest -> largest
        self.num_rings: int = 0

    # ------------------------- Parsing helpers -------------------------
    def _parse_fact_name(self, name: str):
        """Parse grounded fact strings.

        Expected grounded format:
          +predicate[arg1,arg2] or -predicate[arg1,arg2]
        Also supports (predicate arg1 arg2) defensively.
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

        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    @staticmethod
    def _is_ring(sym: str) -> bool:
        # Training instances name rings as r1..r20; keep it simple and robust.
        return isinstance(sym, str) and sym.startswith("r")

    # ------------------------- Preprocessing -------------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        assert self.model is not None

        rings = set()
        smaller_pairs: list[tuple[str, str]] = []  # (smaller_ring, larger_obj)

        # 1) Extract goal positions from the *bitwise* goal state.
        if hasattr(self.model, "goals") and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if self._is_ring(r):
                        self.goal_positions[r] = target
                        rings.add(r)

        # 2) Scan all grounded facts once to collect:
        #    - size ordering from smallerThan (static)
        #    - all possible on-facts per ring (dynamic)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == "smallerThan":
                smaller, larger = args[0], args[1]
                if self._is_ring(smaller):
                    rings.add(smaller)
                    smaller_pairs.append((smaller, larger))
                if self._is_ring(larger):
                    rings.add(larger)

            elif pred == "on":
                r, target = args[0], args[1]
                if self._is_ring(r):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Derive a total size order. Preferred: count how many things each ring is
        #    declared smaller than. In these instances, smallerThan is dense enough
        #    to induce a strict ordering.
        if smaller_pairs:
            smaller_count = defaultdict(int)
            for s, _l in smaller_pairs:
                if s in rings:
                    smaller_count[s] += 1

            # Smallest ring has the *largest* count (smaller than many others).
            self.rings_by_size = sorted(rings, key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: sort by numeric suffix if present (r1 < r2 < ...)
            def key(r: str):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(rings, key=key)

    # ------------------------- Fast state queries -------------------------
    def _get_current_position(self, state_bitwise: int, ring: str) -> str | None:
        """Return the object the ring is currently on by testing 'on' facts bitwise."""
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, target in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ------------------------- Heuristic core -------------------------
    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Check from largest to smallest (reverse of rings_by_size).
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # If goal doesn't mention this ring, don't force it; return 0 contribution.
                continue

            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                # At least all rings 0..i must be moved in any solution.
                k = i + 1
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic12(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic12 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
