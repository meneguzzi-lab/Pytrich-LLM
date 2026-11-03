import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT52Heuristic4(Heuristic):
    """Domain-dependent heuristic for the HDDL Towers domain (Towers of Hanoi).

    Key domain insights leveraged
    ----------------------------
    1) The only primitive operator is `move`, which moves exactly one ring.
       Therefore plan cost (number of primitive steps) equals the number of ring moves.

    2) Classic optimal lower bound structure:
       If the largest ring that is not yet in its goal position has index i (counting from
       smallest=1 to largest), then at least (2^i - 1) moves remain.
       Reason: to move that ring into place, all smaller rings must be moved away (and later
       re-stacked), which requires at least 2^i - 1 moves in any legal Hanoi-like system.

    3) This domain encodes size via `smallerThan` and positions via `on`. Goals are given as
       bitwise facts in `model.goals` (not `goal_on`). We extract goal `+on[...]` facts from
       `model.state_explicit_repr(model.goals)`.

    4) Efficiency: during search we avoid state-to-explicit conversions. We pre-index all
       grounded `on` facts by ring and use bitwise checks on their global_id.

    Returned heuristic
    ------------------
    h(state) = 0 if all rings are already at their goal `on` location.
              else (2^k - 1) where k = number of rings up to and including the largest
                   ring currently not at its goal location.

    This is admissible and typically very informative on this benchmark family.

    Notes on hierarchy
    ------------------
    Although the domain is hierarchical (shiftTower/selectDirection/rotateTower/exchange),
    the decomposition ultimately expands to a sequence of primitive `move` actions whose
    optimal count is governed by the Hanoi recurrence. Thus a state-based lower bound that
    captures the Hanoi structure is also a strong lower bound for remaining refinement cost.
    """

    def __init__(self, towers_gpt52_heuristic_4=None, name: str = "TowersGPT52Heuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        self.model: Model | None = None

        # ring -> goal support object (tower or ring)
        self.goal_positions: dict[str, str] = {}

        # ring -> list[(Fact, target_obj)] for all grounded on[ring,target_obj]
        self.on_facts_by_ring: dict[str, list[tuple[object, str]]] = defaultdict(list)

        # rings sorted smallest -> largest
        self.rings_by_size: list[str] = []

        self.num_rings: int = 0

    # ------------------------- parsing & preprocessing -------------------------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact names.

        Expected grounded format (Pytrich):
            +predicate[arg1,arg2] or -predicate[arg1,arg2]

        Returns:
            (predicate, [args...]) or (None, [])
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

        h0 = self._h_state(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        assert self.model is not None

        rings = set()
        smaller_than_pairs: list[tuple[str, str]] = []

        # Extract goal on-facts from the bitwise goal state
        if hasattr(self.model, "goals") and self.model.goals:
            for gf in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gf)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = target
                        rings.add(r)

        # Index all grounded facts we need for fast bitwise checks
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "on" and len(args) >= 2:
                r, target = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring[r].append((fact, target))

            elif pred == "smallerThan" and len(args) >= 2:
                a, b = args[0], args[1]
                if a.startswith("r"):
                    rings.add(a)
                    smaller_than_pairs.append((a, b))
                if b.startswith("r"):
                    rings.add(b)

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # Sort rings smallest->largest.
        # In these instances, smallerThan(r_i, r_j) exists for i<j.
        # So the smallest ring appears as "smaller" in the most such relations.
        smaller_count = defaultdict(int)
        for a, b in smaller_than_pairs:
            if a in rings and b.startswith("r"):
                smaller_count[a] += 1

        if smaller_count:
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: lexical (r1,r2,...)
            def _ring_key(r: str):
                try:
                    return int(r[1:])
                except Exception:
                    return 10**9

            self.rings_by_size = sorted(list(rings), key=_ring_key)

    # ------------------------------- fast queries ------------------------------

    def _get_current_support(self, state_bitwise: int, ring: str) -> str | None:
        """Return current support object of ring by checking the unique true on-fact."""
        for fact, target in self.on_facts_by_ring.get(ring, []):
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # -------------------------------- heuristic --------------------------------

    def _h_state(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Find largest ring (scan from largest to smallest) that is not at its goal support.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_support = self.goal_positions.get(ring)
            if goal_support is None:
                # If ring not mentioned in the goal, we treat it as already settled.
                continue

            cur_support = self._get_current_support(state_bitwise, ring)
            if cur_support != goal_support:
                k = i + 1  # number of rings from smallest..this ring
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._h_state(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT52Heuristic4(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT52Heuristic4 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT52Heuristic4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal positions (from model.goals): {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
