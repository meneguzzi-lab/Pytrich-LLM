import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Key ideas and rationale:
    - In 3-peg Towers of Hanoi, the minimum number of primitive moves to reposition a k-disk
      tower (from smallest up to the largest unsettled disk) is 2^k - 1. This is a classic,
      admissible lower bound and is tight for many states.
    - We combine two admissible bounds and take their maximum to improve guidance:
        H1 (largest-misaligned bound):
            2^k - 1 where k is the index (1-based) of the largest ring that is not at its goal
            position, counting from the smallest ring upward.
        H2 (per-ring necessity bound):
            The number of rings that are not currently at their goal position. Each such ring
            must be moved at least once, since the only action that changes on[r,*] for a given
            ring r is moving that ring itself.
      The heuristic value is max(H1, H2), which remains admissible and typically stronger than
      either bound alone.

    Implementation details:
    - Fully grounded facts are stored bitwise. We parse and preprocess once, then use bitwise
      checks in constant time per fact during heuristic computation.
    - Goals are read from model.goals (a bitwise integer). We do NOT read goal_on predicates
      from the initial state. Instead, we rely on the final goal facts (typically +on[r,OBJ]).
    - We map, per ring r, all possible grounded on[r,x] facts and their Fact objects to check
      current positions quickly via bitwise operations.
    - Ring size ordering is inferred from smallerThan facts and, if needed, from goal on-chains.
      We produce a list rings_by_size sorted from smallest to largest.

    Notes on hierarchy:
    - The domain uses abstract tasks (shiftTower, selectDirection, rotateTower, exchange) that
      ultimately decompose to primitive move actions. Our heuristic estimates the number of
      remaining primitive moves required, independent of the specific decomposition path.
      This respects the hierarchical structure while staying efficiently computable.
    """

    def __init__(self, towers_gpt5_heuristic_8=None, name="TowersGPT5Heuristic8"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}       # ring -> goal support (OBJ)
        self.rings_by_size = []        # [r_smallest, ..., r_largest]
        self.on_facts_by_ring = {}     # ring -> list of (Fact, target_obj)
        self.num_rings = 0

        # Internal references
        self.model: Model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """
        Parse grounded or HDDL-like fact names into (predicate, [args]).
        Handles two formats:
          - Grounded:  "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]"
          - HDDL-like: "(predicate arg1 arg2)"
        """
        s = name.strip()
        # Grounded format
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]  # drop +/-
            lb = s.find("[")
            if lb > 0 and s.endswith("]"):
                pred = s[:lb]
                args_str = s[lb + 1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        # HDDL-like format
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """
        Extracts:
        - goal_positions: from model.goals (bitwise integer -> explicit names via state_explicit_repr)
        - on_facts_by_ring: mapping ring -> list of all (on[r,x], x)
        - rings_by_size: rings sorted from smallest to largest using smallerThan and/or goal chains
        """
        rings = set()
        smaller_pairs = []  # (smaller_ring, larger_obj)

        # 1) Read goal positions from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == 'on' and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith('r'):
                        self.goal_positions[r] = tgt
                        rings.add(r)

        # 2) Scan all facts to build on_facts_by_ring and collect smallerThan info
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue

            if pred == 'on':
                r, tgt = args[0], args[1]
                if r.startswith('r'):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, tgt))

            elif pred == 'smallerThan':
                sm, lg = args[0], args[1]
                if sm.startswith('r'):
                    rings.add(sm)
                    smaller_pairs.append((sm, lg))
                    if lg.startswith('r'):
                        rings.add(lg)

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3) Determine ring size order: smallest -> largest
        #    Primary: count how many objects each ring is smaller than (from smallerThan)
        #    Fallback/augment: infer order from goal on-chain (if on[r_i, r_{i+1}] then r_i < r_{i+1} )
        smaller_count = defaultdict(int)
        if smaller_pairs:
            for sm, lg in smaller_pairs:
                if sm in rings:
                    smaller_count[sm] += 1
        else:
            # Infer from goal chain if smallerThan is absent
            # Build direct relation: r < r2 if goal has on[r, r2]
            direct_smaller = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if tgt and tgt.startswith('r'):
                    direct_smaller[r].add(tgt)
            # Transitive closure to estimate counts
            changed = True
            while changed:
                changed = False
                for r in list(direct_smaller.keys()):
                    to_add = set()
                    for r2 in direct_smaller[r]:
                        to_add |= direct_smaller.get(r2, set())
                    new = to_add - direct_smaller[r]
                    if new:
                        direct_smaller[r] |= new
                        changed = True
            for r in rings:
                smaller_count[r] = len(direct_smaller.get(r, set()))

        # Sort by decreasing smaller_count so that most-small ring (smaller than many) comes first
        self.rings_by_size = sorted(list(rings), key=lambda rr: -smaller_count.get(rr, 0))

        # If counts tie or missing, provide stable tiebreaker by name to avoid nondeterminism
        # (already stable in Python, but we can ensure deterministic behavior by re-sorting with name)
        # Not strictly necessary; left out to keep it minimal.

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the current support (OBJ) for the given ring by checking which on[r,*] fact is true.
        Uses fast bitwise check: (state >> fact.global_id) & 1
        """
        lst = self.on_facts_by_ring.get(ring)
        if not lst:
            return None
        for fact, tgt in lst:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # H1: Largest-misaligned bound (2^k - 1)
        h1 = 0
        largest_index = -1
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(r)
            if goal_pos is None:
                continue  # ring has no explicit goal; treat as already satisfied
            cur_pos = self._get_current_position(state_bitwise, r)
            if cur_pos != goal_pos:
                largest_index = i
                break
        if largest_index >= 0:
            k = largest_index + 1  # number of rings from smallest up to that ring
            h1 = (1 << k) - 1

        # H2: Count how many rings are not at their goal position (each must move at least once)
        h2 = 0
        for r in self.rings_by_size:
            goal_pos = self.goal_positions.get(r)
            if goal_pos is None:
                continue
            cur_pos = self._get_current_position(state_bitwise, r)
            if cur_pos != goal_pos:
                h2 += 1

        return max(h1, h2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT5Heuristic8(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic8 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic8 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
