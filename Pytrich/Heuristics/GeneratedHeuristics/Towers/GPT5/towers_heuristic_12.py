import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) HDDL domain.

    Core idea (admissible, fast, and scales):
    - The minimum number of primitive moves (actions) needed to move a stack of k rings
      while respecting size constraints is 2^k - 1. This is the classic lower bound
      for Towers of Hanoi and remains a valid lower bound under HTN decomposition
      (since decompositions are zero-cost and the primitive action cost dominates).

    Heuristic computation:
    1) Preprocess the grounded model once:
       - Extract the target/goal position for each ring from model.goals (bitwise integer),
         not from goal_on predicates.
       - Identify all ring symbols and sort them from smallest to largest using smallerThan facts.
       - Index all dynamic 'on' facts by ring for fast bitwise state queries.

    2) For any search node state:
       - Find the largest ring that is not at its goal position.
       - If that ring is the i-th in the order (0-based from smallest), then at least
         the i+1 smallest rings must be moved to resolve it, which requires
         2^(i+1) - 1 moves. Return this as the heuristic value.
       - If all rings are at their goal positions, return 0.

    Why this considers the HTN structure:
    - The hierarchical tasks in this domain (shiftTower -> selectDirection -> rotateTower -> move_abstract -> move)
      ultimately reduce to primitive 'move' actions. The above lower bound directly estimates
      the number of remaining move actions, which is the dominant part of cost in this domain.
      Since decompositions are zero-cost in the usual HTN planning cost model, adding any extra
      cost for abstract tasks would be inadmissible. This heuristic therefore respects the
      hierarchical structure while remaining admissible and computationally efficient.

    Implementation details critical for Pytrich:
    - Goals are accessed via model.goals bitset and converted to explicit names using
      model.state_explicit_repr(model.goals).
    - State membership is checked using fast bitwise operations on Fact.global_id.
    - Fact name parsing supports the grounded form: "+predicate[arg1,arg2]".

    This heuristic is consistent and admissible, suitable for optimal search.
    """

    def __init__(self, towers_gpt5_heuristic_12=None, name="TowersGPT5Heuristic12"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}        # ring -> goal support object (ring or tower)
        self.rings_by_size = []         # [r_smallest, ..., r_largest]
        self.on_facts_by_ring = {}      # ring -> list of (Fact, support_obj)
        self.num_rings = 0

        # Optional cache (state_bitwise -> h-value). Conservatively sized by usage.
        self._h_cache = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """
        Parse grounded fact name formats used by Pytrich after grounding.
        Supported format (critical): +predicate[arg1,arg2]
        Also safely ignores '-' prefix if present.
        Returns: (predicate, [arg1, arg2, ...]) or (None, []) on failure.
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

    def _preprocess(self):
        """
        Extract:
        - Goal positions of rings from model.goals (bitwise integer -> explicit names)
        - Set of rings and their size order using smallerThan facts
        - Index all 'on' facts per ring for O(1) state checks via bitwise operations
        """
        rings = set()
        ring_smaller_pairs = []  # list of (ring_smaller, ring_larger)

        # 1) Goal positions come from model.goals bitset (NOT from goal_on predicate)
        self.goal_positions.clear()
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith("r"):  # ring symbol heuristic
                        self.goal_positions[r] = tgt

        # Fallback: if for some reason goals bitset didn't yield ring goals,
        # try to read static goal_on from facts (as seen in some instances' init)
        if not self.goal_positions:
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if pred == "goal_on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith("r"):
                        self.goal_positions[r] = tgt

        # 2) Collect rings and index 'on' facts (dynamic) and size relations
        self.on_facts_by_ring.clear()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue

            if pred == "on":
                r, obj = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, obj))

            elif pred == "smallerThan":
                smaller, larger = args[0], args[1]
                # Track ring symbols
                if smaller.startswith("r"):
                    rings.add(smaller)
                if larger.startswith("r"):
                    rings.add(larger)
                    # Only keep ring->ring pairs for ordering
                    if smaller.startswith("r"):
                        ring_smaller_pairs.append((smaller, larger))

        self.num_rings = len(rings)

        # 3) Sort rings from smallest to largest
        self.rings_by_size = []
        if self.num_rings == 0:
            return

        if ring_smaller_pairs:
            # Count how many larger rings each ring is smaller than
            smaller_count = defaultdict(int)
            for s, l in ring_smaller_pairs:
                if s in rings and l in rings:
                    smaller_count[s] += 1
            # Rings with higher count are smaller (on top of more rings)
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer partial order from goals (if r_i on r_{i+1} in goal, then r_i < r_{i+1})
            is_smaller_than = defaultdict(set)  # ring -> set(rings larger than it)
            for r, tgt in self.goal_positions.items():
                if r.startswith("r") and tgt and tgt.startswith("r"):
                    is_smaller_than[r].add(tgt)
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for a in list(is_smaller_than.keys()):
                    to_add = set()
                    for b in list(is_smaller_than[a]):
                        to_add |= is_smaller_than.get(b, set())
                    new_items = to_add - is_smaller_than[a]
                    if new_items:
                        is_smaller_than[a] |= new_items
                        changed = True
            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the current support object for a given ring in the provided state
        by checking which 'on[ring, X]' fact is true using fast bitwise lookup.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, tgt in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        # Quick cache lookup (optional, safe)
        cached = self._h_cache.get(state_bitwise)
        if cached is not None:
            return cached

        if not self.rings_by_size:
            self._h_cache[state_bitwise] = 0
            return 0

        # Iterate from largest to smallest. The first (largest) ring that is not at goal
        # determines the number of smallest rings that must necessarily move.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                k = i + 1  # number of smallest rings up to (and including) r
                h = (1 << k) - 1  # 2^k - 1
                self._h_cache[state_bitwise] = h
                return h

        # All rings are already at goal positions
        self._h_cache[state_bitwise] = 0
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGPT5Heuristic12(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic12 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
