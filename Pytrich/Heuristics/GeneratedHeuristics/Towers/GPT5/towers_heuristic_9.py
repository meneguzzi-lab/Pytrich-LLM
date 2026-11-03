import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - The Towers domain models the classic 3-peg Towers of Hanoi problem via HTN tasks.
    - The only primitive action is "move", which moves the top ring from one object to another
      respecting smallerThan constraints (i.e., no larger ring on top of a smaller ring).
    - The HTN hierarchy (shiftTower -> selectDirection/rotateTower/exchange -> move_abstract -> move)
      ultimately refines to a sequence of primitive moves; costs are determined by primitive moves only.

    Key property leveraged:
    - For k active (not yet fixed) rings, the minimum number of primitive moves required is 2^k - 1.
      This is the optimal lower bound for 3 pegs and is admissible.

    Hierarchy-aware estimation:
    - We compute the number of rings that are already fixed in their final position, starting from the
      largest ring downward (a suffix of the goal stack). Those rings need not move again in any optimal
      solution. The remaining k = total_rings - fixed_suffix_length rings necessarily require at least
      2^k - 1 moves.
    - This implicitly considers the hierarchical structure, since any completion of the HTN must include
      those primitive moves; we do not add costs for abstract decompositions because plan cost is defined
      over primitive moves only.

    Efficiency considerations:
    - All state checks use the model's bitwise representation (no string-in-list searches).
    - We preprocess:
        * goal positions for each ring from model.goals (bitwise integer -> explicit names)
        * mapping from each ring to its grounded 'on' facts for fast current position lookup
        * ring ordering (smallest to largest) using smallerThan facts

    Admissibility and consistency:
    - Admissible: never overestimates true remaining move count.
    - Consistent: Moving one step reduces the lower bound by at most the step cost (1), thanks to the
      known structure of the Hanoi problem and the monotone nature of the 2^k - 1 bound when k decreases.
    """

    def __init__(self, towers_gpt5_heuristic_9=None, name="TowersGPT5Heuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}       # ring -> goal support (OBJ)
        self.rings_by_size = []        # list of rings sorted smallest -> largest
        self.on_facts_by_ring = {}     # ring -> list of tuples (Fact, target_obj)
        self.num_rings = 0

        # Caches
        self._current_pos_cache = {}   # optional per-state cache (state_id -> {ring: pos}); not used by default

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # Preprocess static/domain info
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Initial h-value
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse grounded fact names in either supported format:
        1) Grounded format: "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]"
        2) HDDL-like format: "(predicate arg1 arg2)"
        Returns: (predicate, [args]) or (None, []).
        """
        name = name.strip()
        # Format 1: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bidx = name2.find("[")
            if bidx > 0 and name2.endswith("]"):
                predicate = name2[:bidx]
                args_str = name2[bidx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        # Format 2: (predicate arg1 arg2)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """
        Extract domain structure from grounded facts and goals.

        - Goal positions for rings are read from model.goals (bitwise integer) via state_explicit_repr.
        - The ordering of rings (smallest -> largest) is computed from smallerThan facts if available;
          otherwise, approximated using goal stacking relations.
        - For each ring, collect all grounded 'on' facts for quick current-position lookup.
        """
        rings = set()
        smaller_pairs = []  # list of (smaller_ring, larger_obj)

        # 1) Extract goal positions from the bitwise goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                predicate, args = self._parse_fact_name(gname)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)
                        if target.startswith("r"):
                            rings.add(target)

        # 2) Scan all facts to gather 'on' facts for rings and collect smallerThan relationships
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller, larger = args[0], args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_pairs.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) Determine ring order (smallest -> largest)
        if smaller_pairs:
            # Count how many objects each ring is smaller than; more -> smaller ring
            smaller_count = defaultdict(int)
            for s, l in smaller_pairs:
                if s in rings:
                    smaller_count[s] += 1
            # Sort so that ring with highest smaller_count is smallest
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer partial order from goal stack
            # If r is on x in goal and x is a ring, then r < x.
            is_smaller_than = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if tgt and tgt.startswith("r"):
                    is_smaller_than[r].add(tgt)
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller_than.keys()):
                    for l in list(is_smaller_than[r]):
                        for ll in is_smaller_than.get(l, set()):
                            if ll not in is_smaller_than[r]:
                                is_smaller_than[r].add(ll)
                                changed = True
            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the current support object that 'ring' is on by checking which 'on' fact is true.
        Uses fast bitwise checks over pre-collected Fact objects.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, target in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Compute an admissible lower bound on remaining primitive moves.

        Algorithm:
        - Identify the suffix of largest rings that are already in their goal positions (fixed base).
        - Let k = total_rings - fixed_suffix_length. Then h = 2^k - 1.

        Rationale:
        - Any plan that finishes must move all k active (not-yet-fixed) rings; the optimal lower bound
          for 3 pegs is 2^k - 1. Rings already fixed at the bottom do not need to move again in any
          optimal solution, thanks to the monotonic structure of Towers of Hanoi.
        """
        if not self.rings_by_size:
            return 0

        fixed_suffix = 0
        # rings_by_size is smallest -> largest; iterate largest -> smallest
        for ring in reversed(self.rings_by_size):
            cur_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)
            if cur_pos is not None and goal_pos is not None and cur_pos == goal_pos:
                fixed_suffix += 1
            else:
                break

        k_active = max(0, len(self.rings_by_size) - fixed_suffix)
        if k_active == 0:
            return 0
        return (1 << k_active) - 1

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT5Heuristic9(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic9 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic9 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
