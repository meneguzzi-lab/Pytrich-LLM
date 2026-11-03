import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic15(Heuristic):
    """
    TowersGPT5Heuristic15

    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - Primitive cost driver is the number of 'move' actions. All other HTN decompositions
      expand into these moves without adding action cost.
    - For a 3-peg Towers of Hanoi, to place the largest misplaced ring correctly (and finish),
      you must move all smaller rings out of the way and then rebuild them on top. This yields
      the classic admissible lower bound 2^k - 1 where k is the number of rings from the
      smallest up to and including the largest misplaced ring.
    - We leverage the grounded model:
        * Goals are provided as a bitwise integer; we decode them via model.state_explicit_repr.
        * We parse facts of the form +predicate[arg1,arg2].
        * We use bitwise checks for state membership: (state >> fact.global_id) & 1.
    - We also compute a safe secondary lower bound: the count of rings not in their goal position.
      Taking the max with the classic 2^k - 1 bound can help on very small or near-goal plateaus
      while remaining admissible.

    Efficiency considerations:
    - All static preprocessing (ring order, goals, and per-ring 'on' facts) is done once in
      initialize().
    - At evaluation time, we use only bitwise checks; no string searches over the state.

    This heuristic is admissible and designed to scale well from the smallest to the largest
    provided instances.
    """

    def __init__(self, towers_gpt5_heuristic_15=None, name: str = "TowersGPT5Heuristic15"):
        super().__init__(name=name)
        self.preprocessing_time: float = 0.0

        # Populated in initialize/_preprocess
        self.model: Optional[Model] = None
        self.goal_positions: Dict[str, str] = {}      # ring -> target object (from goal 'on' facts)
        self.rings_by_size: List[str] = []            # rings sorted smallest -> largest
        self.on_facts_by_ring: Dict[str, List[Tuple[object, str]]] = {}  # ring -> list of (Fact, target)
        self.num_rings: int = 0

    # -------------------- Framework API -------------------- #
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    # -------------------- Parsing Helpers -------------------- #
    def _parse_fact_name(self, name: str):
        """
        Parse grounded fact names of the form '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Also tolerates '(predicate arg1 arg2)' if encountered (not expected here but harmless).
        Returns: (predicate, [args...]) or (None, []).
        """
        name = name.strip()

        # Grounded format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            idx = name.find("[")
            if idx > 0 and name.endswith("]"):
                predicate = name[:idx]
                args_str = name[idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Optional HDDL-style support
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    # -------------------- Preprocessing -------------------- #
    def _preprocess(self):
        rings = set()
        smaller_than_pairs = []  # (smaller, larger)

        # 1) Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, "goals") and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith("r"):  # ring
                        self.goal_positions[r] = target

        # 2) Traverse all grounded facts to collect:
        #    - ring set
        #    - smallerThan relations (to sort rings by size)
        #    - per-ring 'on' facts with their Fact objects for fast bitwise checking
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
            return

        # 3) Sort rings by size (smallest -> largest) using smallerThan counts
        if smaller_than_pairs:
            # Count how many times each ring appears as 'smaller' to induce size order
            smaller_count = defaultdict(int)
            for s, _ in smaller_than_pairs:
                if s in rings:
                    smaller_count[s] += 1
            # More 'smallerThan' targets imply a smaller ring
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer from goals by reading chains r_i on r_{i+1}
            below_map = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if tgt.startswith("r"):
                    below_map[r].add(tgt)
            # Transitively close
            changed = True
            while changed:
                changed = False
                for r in list(below_map.keys()):
                    for b in list(below_map[r]):
                        for bb in below_map.get(b, set()):
                            if bb not in below_map[r]:
                                below_map[r].add(bb)
                                changed = True
            # Sort by how many rings each is smaller than
            smaller_count = {r: len(below_map.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    # -------------------- Heuristic Core -------------------- #
    def _get_current_position(self, state_bitwise: int, ring: str) -> Optional[str]:
        """
        Return the current object the ring is 'on', by scanning pre-collected 'on' facts for that ring
        and checking which one holds in the bitwise state.
        """
        candidates = self.on_facts_by_ring.get(ring)
        if not candidates:
            return None
        for fact, target in candidates:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _largest_misplaced_index(self, state_bitwise: int) -> int:
        """
        Return the index (in rings_by_size) of the largest ring that is not yet at its goal position.
        If all rings are already at goal, return -1.
        """
        if not self.rings_by_size:
            return -1
        # Iterate largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                return i
        return -1

    def _misplaced_count(self, state_bitwise: int) -> int:
        cnt = 0
        for r in self.rings_by_size:
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                cnt += 1
        return cnt

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Admissible lower bound of remaining moves:
        - Primary bound: 2^(k) - 1, where k is the number of rings from smallest up to and including
          the largest misplaced ring. This follows the classic 3-peg Towers of Hanoi lower bound.
        - Secondary bound: number of rings not at their goal position. We take the max with the
          primary bound for a slightly tighter (still admissible) estimate.
        """
        if not self.rings_by_size:
            return 0

        largest_idx = self._largest_misplaced_index(state_bitwise)
        if largest_idx < 0:
            return 0

        # k = number of rings up to and including largest misplaced (index is 0-based)
        k = largest_idx + 1
        primary_lb = (1 << k) - 1  # 2^k - 1

        secondary_lb = self._misplaced_count(state_bitwise)

        return max(primary_lb, secondary_lb)

    # -------------------- Introspection -------------------- #
    def __repr__(self):
        return f"TowersGPT5Heuristic15(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic15 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic15 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
