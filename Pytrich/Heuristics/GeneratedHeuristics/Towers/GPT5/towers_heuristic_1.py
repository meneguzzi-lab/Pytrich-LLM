import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - In Towers of Hanoi, to move the largest misplaced ring, all smaller rings above it
      must be displaced first. This yields a classic tight lower bound: 2^k - 1 moves,
      where k is the number of rings from the smallest up to the largest misplaced one.
    - Goals in Pytrich are provided as a bitwise integer. We extract ring goal positions
      from model.goals using model.state_explicit_repr(model.goals), NOT from goal_on facts.
    - We preprocess the grounded model once, building:
        * goal_positions: ring -> goal object it must be on (ring or tower)
        * rings_by_size: list of rings (smallest -> largest) using smallerThan facts
        * on_facts_by_ring: ring -> list of (fact, target_obj) to quickly check current state
    - The heuristic value h(n) is computed as:
        h1 = 2^k - 1, where k indexes the largest ring not currently at its goal position
        h2 = number of rings not yet at their goal position (each must move at least once)
      We return max(h1, h2), which remains admissible and often increases informativeness
      over using only h1.
    - Efficiency: All state checks are done via bitwise operations using fact.global_id.

    Admissibility: The estimate never exceeds the true minimal number of primitive moves,
    because:
      * 2^k - 1 is a classic admissible lower bound for Towers of Hanoi,
      * The count of misplaced rings is also a valid lower bound (each incorrect ring
        must be moved at least once), and taking max of lower bounds keeps it admissible.

    Hierarchical awareness: While we do not simulate task decompositions at runtime, this
    heuristic is a lower bound on the number of primitive 'move' actions needed to complete
    any valid decomposition of the shiftTower task network.
    """

    def __init__(self, towers_gpt5_heuristic_1=None, name: str = "TowersGPT5Heuristic1"):
        super().__init__(name=name)
        self.preprocessing_time: float = 0.0

        # Domain knowledge extracted during preprocessing
        self.goal_positions: Dict[str, str] = {}      # ring -> target object (from goal facts)
        self.rings_by_size: List[str] = []            # rings sorted smallest -> largest
        self.on_facts_by_ring: Dict[str, List[Tuple[object, str]]] = {}  # ring -> list of (Fact, target)
        self.num_rings: int = 0

        # Cache for faster membership tests
        self._rings_set: Set[str] = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time

        # Initial heuristic value for the root
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.

        Handles the grounded format used by Pytrich facts:
          +predicate[arg1,arg2] or -predicate[arg1,arg2]
        Also tolerates classic PDDL/HDDL format: (predicate arg1 arg2)
        """
        name = name.strip()

        # Grounded format
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # HDDL-style format (tolerant path)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        """
        Extract domain structure from grounded facts and goals.
        - Build goal_positions from model.goals using explicit state representation.
        - Build rings set, on_facts_by_ring mapping, and derive rings_by_size.
        """
        rings: Set[str] = set()
        smaller_than_pairs: List[Tuple[str, str]] = []  # (smaller_ring, larger_obj)

        # 1) Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                predicate, args = self._parse_fact_name(gname)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):  # rings are named r1, r2, ...
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2) Scan all facts to learn about rings, size ordering (smallerThan), and 'on' facts
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller, larger = args[0], args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_pairs.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self._rings_set = rings
        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3) Sort rings by size (smallest -> largest)
        # Prefer using smallerThan counts if available
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than_pairs:
                if smaller in rings:
                    # Count how many objects each ring is smaller than
                    if larger.startswith("r") or larger.startswith("t"):
                        smaller_count[smaller] += 1
            # Larger count => smaller ring (on top of more things)
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer from goal chain (if provided)
            # If ring a is on ring b in goal, then a < b
            is_smaller_than = defaultdict(set)
            for ring, target in self.goal_positions.items():
                if target.startswith("r"):
                    is_smaller_than[ring].add(target)

            # Compute a transitive closure (small set of up to ~20 rings -> cheap)
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller_than.keys()):
                    current = list(is_smaller_than[r])
                    for mid in current:
                        for large in is_smaller_than.get(mid, set()):
                            if large not in is_smaller_than[r]:
                                is_smaller_than[r].add(large)
                                changed = True

            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> Optional[str]:
        """
        Return the object 'o' such that 'on(ring, o)' is true in the given state.
        Uses fast bitwise checks over pre-indexed Fact objects.
        """
        flist = self.on_facts_by_ring.get(ring)
        if not flist:
            return None
        for fact, target in flist:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Compute max of two admissible lower bounds:
        - h1: 2^k - 1 for the largest ring not at its goal position
        - h2: number of rings not at their goal position (each must move at least once)
        """
        if not self.rings_by_size:
            return 0

        largest_misplaced_index = -1
        misplaced_count = 0

        for i in range(len(self.rings_by_size)):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # No explicit goal for this ring -> ignore it in the estimate
                continue
            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                misplaced_count += 1
                largest_misplaced_index = i  # rings_by_size is smallest -> largest

        if largest_misplaced_index < 0:
            # All goal-tracked rings are already at goal
            return 0

        # Number of rings up to and including the largest misplaced one
        k = largest_misplaced_index + 1
        h1 = (1 << k) - 1  # 2^k - 1
        h2 = misplaced_count
        return max(h1, h2)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT5Heuristic1(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic1 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic1 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest to largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
