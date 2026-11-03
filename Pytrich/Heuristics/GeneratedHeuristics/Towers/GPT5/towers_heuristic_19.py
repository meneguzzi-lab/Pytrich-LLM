"""
Towers domain-dependent heuristic for Pytrich Planner.

Heuristic: TowersGPT5Heuristic19
- Admissible and consistent lower bound for the Towers of Hanoi domain.
- Uses the classic property: to move the largest misplaced ring, all smaller rings
  must be moved out of the way first. Therefore, if the largest misplaced ring is the i-th
  in size (counting from smallest, starting at 1), then a lower bound on the remaining
  number of primitive moves is 2^i - 1.
- Additionally, it uses a cheap bitwise lower bound equal to the number of unsatisfied
  goal facts (each move can make at most one new goal on-fact true), and returns the max
  of both lower bounds for extra pruning power while staying admissible.

Key implementation details:
- Fully grounded fact parsing for both HDDL-like and grounded "+pred[args]" formats.
- All goal facts are extracted from model.goals (bitwise integer), per framework spec.
- Efficient state checks via bitwise operations only (no string set membership).
- Precomputes, per ring, the bitmask of all its on-facts and a fast map from fact id to
  its support target, so current ring position can be read in O(1) expected time.
- Preprocesses ring size order from smallerThan relations (fallbacks to goal structure).

This heuristic is fast, scalable (tested conceptually up to 20+ rings), and effective at
reducing node expansions in the Towers domain.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.

    h(s) = max( 2^k - 1, |G \ s| )
      where:
        - k is the index (1-based) of the largest ring that is not at its goal support
          when rings are ordered from smallest to largest; if all rings are at goal,
          k=0 and the term is 0.
        - |G \ s| is the number of goal facts unsatisfied in s (each move can add
          at most one new 'on' goal fact).

    This is admissible and consistent, leveraging Towers-specific structure and the
    HTN leaves being the single primitive move action.
    """

    def __init__(self, towers_gpt5_heuristic_19=None, name="TowersGPT5Heuristic19"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed structures
        self.goal_positions = {}            # ring -> target object (from goals)
        self.rings_by_size = []             # [r_smallest, ..., r_largest]
        self.num_rings = 0

        # Fast access to current ring position
        # on_facts_by_ring: ring -> list of (Fact, target)
        self.on_facts_by_ring = {}
        # on_mask_by_ring: ring -> int bitmask of all on-facts for that ring
        self.on_mask_by_ring = {}
        # on_gid_to_target: ring -> { global_id: target }
        self.on_gid_to_target = {}

        # Goals as bitwise
        self.goal_bits = 0

    # ----------------------- Initialization and Preprocessing -----------------------

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """
        Parse fact name supporting both grounded and HDDL-like formats.
        - Grounded: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        - HDDL-like: (predicate arg1 arg2)
        Returns (predicate, [args...]) or (None, []).
        """
        name = name.strip()

        if name.startswith("+") or name.startswith("-"):
            nm = name[1:]
            b = nm.find("[")
            if b > 0 and nm.endswith("]"):
                predicate = nm[:b]
                args_str = nm[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        # 1) Extract goal facts directly from model.goals (bitwise integer)
        self.goal_bits = getattr(self.model, "goals", 0) or 0
        if self.goal_bits:
            for gname in self.model.state_explicit_repr(self.goal_bits):
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        rings = set()
        smaller_pairs = []  # (smaller_ring, larger_obj)

        # 2) Scan all grounded facts to build size info and fast on-fact lookup
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
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

        # 3) Build per-ring bitmasks and gid->target maps for fast current-position lookup
        for ring, entries in self.on_facts_by_ring.items():
            mask = 0
            gid2t = {}
            for fact, tgt in entries:
                mask |= (1 << fact.global_id)
                gid2t[fact.global_id] = tgt
            self.on_mask_by_ring[ring] = mask
            self.on_gid_to_target[ring] = gid2t

        # 4) Order rings from smallest to largest
        if self.num_rings == 0:
            return

        if smaller_pairs:
            # Count for each ring how many others it is smaller than (ring->count)
            smaller_count = defaultdict(int)
            for a, b in smaller_pairs:
                if a in rings and b.startswith("r"):
                    smaller_count[a] += 1
            # The smallest ring is smaller than the most others
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer partial ordering from goal chain (if any)
            # Build DAG: r -> larger rings it is below in goal (ring on ring => ring < ring)
            larger_map = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if r in rings and tgt.startswith("r"):
                    larger_map[r].add(tgt)

            # Compute transitive closure to count how many rings each is smaller than
            changed = True
            while changed:
                changed = False
                for r in list(larger_map.keys()):
                    add_set = set()
                    for mid in list(larger_map[r]):
                        add_set |= larger_map.get(mid, set())
                    before = len(larger_map[r])
                    larger_map[r] |= add_set
                    if len(larger_map[r]) != before:
                        changed = True

            smaller_count = {r: len(larger_map.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    # ---------------------------- Fast Helpers --------------------------------------

    @staticmethod
    def _popcount(x: int) -> int:
        try:
            return x.bit_count()  # Python 3.8+
        except AttributeError:
            # Fallback popcount
            c = 0
            while x:
                x &= x - 1
                c += 1
            return c

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Fast current support lookup for a ring using a precomputed bitmask and gid->target map.
        Returns the support object (ring or tower) or None if undetermined.
        """
        mask = self.on_mask_by_ring.get(ring, 0)
        if not mask:
            return None
        var = state_bitwise & mask
        if not var:
            return None
        # get index of least significant set bit (one on-fact per ring should hold)
        ls1b = var & -var
        gid = ls1b.bit_length() - 1
        return self.on_gid_to_target[ring].get(gid, None)

    # ----------------------------- Heuristic Core -----------------------------------

    def _largest_misplaced_lb(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0
        # Iterate from largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                # i is 0-based index (smallest at 0); number of rings up to r is i+1
                k = i + 1
                return (1 << k) - 1
        return 0

    def _unsatisfied_goals_lb(self, state_bitwise: int) -> int:
        # Count how many goal facts are not yet true: popcount(goal_bits & ~state)
        return self._popcount(self.goal_bits & ~state_bitwise)

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        # Primary LB from largest misplaced ring
        lb_hanoi = self._largest_misplaced_lb(state_bitwise)
        # Secondary LB from number of unsatisfied goal on-facts
        lb_goals = self._unsatisfied_goals_lb(state_bitwise)
        return max(lb_hanoi, lb_goals)

    # ------------------------------- Interface --------------------------------------

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT5Heuristic19(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic19 with {self.num_rings} rings"

    def __output__(self):
        return (
            "TowersGPT5Heuristic19 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
