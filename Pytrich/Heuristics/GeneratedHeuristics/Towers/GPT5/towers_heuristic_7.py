import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic7(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - In three-peg Towers of Hanoi, the minimum number of primitive moves (action 'move')
      required to move a stack of k rings is 2^k - 1. This holds irrespective of
      intermediate placements due to the recursive structure of the optimal solution.
    - To finish the problem from any state, find the largest ring that is not yet at
      its goal location. Moving that ring to its goal necessarily requires relocating
      all smaller rings at least once, giving a tight lower bound of 2^k - 1 where
      k is the number of rings up to and including that largest misplaced ring
      (counting from the smallest).
    - This heuristic leverages domain structure and hierarchy implicitly because the
      HTN decomposition (shiftTower -> selectDirection/rotateTower/exchange -> move)
      ultimately yields a sequence of primitive 'move' actions, and the lower bound
      is computed in terms of primitive actions. It is admissible and consistent.

    Implementation notes:
    - Goals must be read from model.goals (bitwise integer) via model.state_explicit_repr.
    - Current state must be queried via fast bitwise checks using Fact.global_id.
    - We preprocess:
        * Goal positions for each ring (from '+on[ring,obj]' goal facts)
        * A total order of rings from smallest to largest using 'smallerThan' facts.
          If incomplete, we fall back to ordering inferred from the goal 'on' chain.
        * All 'on' facts per ring to quickly detect a ring's current position.
    - Heuristic value for a state is: 2^k - 1 for the largest misplaced ring.
    - This is efficient and scales well to the largest provided instances.
    """

    def __init__(self, towers_gpt5_heuristic_7=None, name: str = "TowersGPT5Heuristic7"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}       # ring -> goal object (tower or ring)
        self.rings_by_size = []        # [smallest, ..., largest]
        self.on_facts_by_ring = {}     # ring -> List[(Fact, target_obj)]
        self.num_rings = 0
        self._rings_set = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """
        Parse grounded fact names. Supports formats:
        - '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'
        - '(predicate arg1 arg2)'
        Returns (predicate, [args]) or (None, []).
        """
        name = name.strip()
        if not name:
            return None, []

        if name.startswith("+") or name.startswith("-"):
            s = name[1:]
            b = s.find("[")
            if b > 0 and s.endswith("]"):
                pred = s[:b]
                args_str = s[b+1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
            return None, []

        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        # 1) Extract ring goal positions from model.goals (bitwise integer)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == 'on' and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith('r'):
                        self.goal_positions[ring] = target
                        self._rings_set.add(ring)

        # 2) Scan all facts to collect ordering and 'on' mappings
        larger_by_ring = defaultdict(set)  # ring -> set of larger rings

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == 'smallerThan':
                a, b = args[0], args[1]
                if a.startswith('r'):
                    self._rings_set.add(a)
                    if b.startswith('r'):
                        self._rings_set.add(b)
                        larger_by_ring[a].add(b)

            elif pred == 'on':
                ring, target = args[0], args[1]
                if ring.startswith('r'):
                    self._rings_set.add(ring)
                    self.on_facts_by_ring.setdefault(ring, []).append((fact, target))

        self.num_rings = len(self._rings_set)
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        # 3) If ordering incomplete, try to infer from goal 'on' chain
        if not larger_by_ring:
            for ring, tgt in self.goal_positions.items():
                if tgt and tgt.startswith('r'):
                    larger_by_ring[ring].add(tgt)

        # 4) Transitive closure to strengthen partial order
        if larger_by_ring:
            changed = True
            while changed:
                changed = False
                for r in list(larger_by_ring.keys()):
                    to_add = set()
                    for lr in larger_by_ring[r]:
                        to_add |= larger_by_ring.get(lr, set())
                    new = to_add - larger_by_ring[r]
                    if new:
                        larger_by_ring[r] |= new
                        changed = True

        # 5) Sort rings smallest->largest by number of larger rings
        counts = {r: len(larger_by_ring.get(r, set())) for r in self._rings_set}
        # Ring with higher count is smaller; break ties by ring name for determinism
        self.rings_by_size = sorted(self._rings_set, key=lambda r: (-counts.get(r, 0), r))

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the object that 'ring' is currently on in 'state_bitwise'.
        Uses bitwise checks over pre-collected on-facts for that ring.
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
        For each ring from largest to smallest, check if it is at its goal.
        Let i be the index (0-based from smallest) of the largest misplaced ring.
        Then a lower bound on remaining moves is 2^(i+1) - 1.
        If all rings are at their goal positions, return 0.
        """
        if not self.rings_by_size:
            return 0

        # Iterate largest -> smallest (reverse of smallest->largest ordering)
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # No explicit goal for this ring -> assume already satisfied
                continue

            cur_pos = self._get_current_position(state_bitwise, ring)
            if cur_pos != goal_pos:
                k = i + 1  # number of rings from smallest up to this ring
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    def __repr__(self):
        return f"TowersGPT5Heuristic7(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic7 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic7 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
