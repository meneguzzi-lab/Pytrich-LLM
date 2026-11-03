"""
Towers domain-dependent heuristic for Pytrich Planner.

Class: TowersGPT5Heuristic0 (inherits from Heuristic)

Design rationale and domain insights:
- This domain encodes the classic Towers of Hanoi problem in HDDL, with hierarchical tasks
  ultimately decomposing to primitive move actions. The true optimal cost (in number of moves)
  to transfer a stack of k rings is 2^k - 1.
- We leverage this by identifying the largest ring that is not yet in its goal position.
  To move that ring, all smaller rings must be moved at least once, hence a tight lower bound
  of 2^k - 1 where k is the number of rings up to and including that ring (smallest..that ring).
- This gives an admissible and very informative heuristic that scales well to large instances.

Implementation highlights:
- Preprocess model facts once (in initialize) to:
  * Extract goal positions from model.goals (bitwise int); do NOT read goal_on from state.
  * Build a size ordering over rings using smallerThan facts.
  * Index dynamic on-facts per ring for fast bitwise checks.
- For each state, scan rings from largest to smallest; the first ring not at goal yields
  the bound 2^k - 1; if all are at goal, return 0.
- Uses bitwise checks only; never reconstructs full explicit state during search.

Notes:
- This heuristic is admissible and consistent for Towers of Hanoi.
- It is independent of the current HTN task network, but already provides near-optimal guidance
  because the domain is perfectly captured by the ring positions alone.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic0(Heuristic):
    """
    Optimal domain-dependent heuristic for the Towers (Hanoi) domain.

    Heuristic value h(s):
      - Let r* be the largest ring not at its goal position in state s.
      - Let k be the number of rings from the smallest up to and including r*.
      - Return 2^k - 1. If all rings are at goal positions, return 0.

    Why admissible/tight:
      - To move the largest misplaced ring r*, all smaller rings must be moved
        out of the way at least once, which costs at least 2^k - 1 moves in Towers of Hanoi.
      - This is the classic optimal lower bound and is equal to the optimal solution length
        when moving a k-ring stack.
    """

    def __init__(self, towers_gpt5_heuristic_0=None, name="TowersGPT5Heuristic0"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}        # ring -> target object (from goal 'on' facts)
        self.rings_by_size = []         # list[str]: rings sorted from smallest to largest
        self.on_facts_by_ring = {}      # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0

        # Optional caches
        self._ring_index = {}           # ring -> index in rings_by_size (0=smallest)

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # Preprocess once
        self._preprocess()

        self.preprocessing_time = time.time() - start_time
        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles grounded Pytrich format: 
          +predicate[arg1,arg2] or -predicate[arg1,arg2]
        and also the HDDL textual format: 
          (predicate arg1 arg2)
        """
        name = name.strip()
        # Grounded format: +pred[a,b] or -pred[a,b]
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            b = name2.find("[")
            if b > 0 and name2.endswith("]"):
                pred = name2[:b]
                args_str = name2[b+1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        # HDDL format: (pred a b)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """
        Extract domain structure from grounded facts and the goal bitset.
        - Read goal positions from model.goals (bitwise integer), not from goal_on.
        - Build ring set and size ordering using smallerThan facts.
        - Index all 'on' facts per ring for fast state checks via bitwise.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller_ring, larger_obj)

        # 1) Extract goal 'on' facts from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            for goal_name in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(goal_name)
                if pred == 'on' and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith('r'):
                        self.goal_positions[ring] = target

        # 2) Index facts and gather ring set and ordering info
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == 'smallerThan':
                small, large = args[0], args[1]
                if small.startswith('r'):
                    rings.add(small)
                    smaller_than_pairs.append((small, large))
                    if large.startswith('r'):
                        rings.add(large)

            elif pred == 'on':
                ring, target = args[0], args[1]
                if ring.startswith('r'):
                    rings.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            self.rings_by_size = []
            self._ring_index = {}
            return

        # 3) Sort rings by size using smallerThan counts (more supersets => smaller ring)
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for s, l in smaller_than_pairs:
                if s in rings:
                    smaller_count[s] += 1
            # Highest count => smallest
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer partial order from goals (if ring on ring => smaller)
            is_smaller_than = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if tgt.startswith('r'):
                    is_smaller_than[r].add(tgt)
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller_than.keys()):
                    larger_set = list(is_smaller_than[r])
                    for L in larger_set:
                        for LL in is_smaller_than.get(L, set()):
                            if LL not in is_smaller_than[r]:
                                is_smaller_than[r].add(LL)
                                changed = True
            # Count supersets; more supersets => smaller
            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

        self._ring_index = {r: i for i, r in enumerate(self.rings_by_size)}

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the current object that 'ring' is on in the given state, or None if unknown.
        Uses bitwise check: (state >> fact.global_id) & 1
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
        Compute the 2^k - 1 bound from the largest misplaced ring.
        Rings are scanned from largest to smallest (reverse order of rings_by_size).
        """
        if not self.rings_by_size:
            return 0

        # Iterate largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            cur_pos = self._get_current_position(state_bitwise, ring)

            if goal_pos is None:
                # If no explicit goal for this ring, treat it as satisfied (safe/admissible)
                continue

            if cur_pos != goal_pos:
                # i is index (0-based) in smallest->largest list, so need i+1 rings to move
                k = i + 1
                return (1 << k) - 1

        # All rings are at their goal positions
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT5Heuristic0(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic0 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic0 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
