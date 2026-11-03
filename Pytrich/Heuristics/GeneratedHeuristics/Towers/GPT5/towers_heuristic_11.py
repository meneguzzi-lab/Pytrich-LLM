import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Towers (HDDL) domain in Pytrich.

    Core idea (admissible and consistent):
    - The Towers domain is a variant of the Towers of Hanoi with 3 towers and k rings.
    - The optimal number of primitive moves to relocate the largest misplaced ring (and all
      smaller rings stacked above it) is 2^k - 1, where k is the count of rings from the smallest
      up to (and including) that largest misplaced ring.
    - We compute a tight lower bound by identifying the largest ring that is not at its goal
      position (as specified by the model goals, NOT by goal_on in the init), then return 2^k - 1.

    Why this works and is effective:
    - To move a larger ring, all smaller rings must first be moved off it; this is the fundamental
      constraint of Towers of Hanoi. Thus, the number of moves required is at least 2^k - 1.
    - This bound is often exact in Towers-like problems and gives very strong guidance.

    Hierarchical-awareness:
    - Although HTN search reasons over abstract tasks (shiftTower, rotateTower, exchange, etc.),
      the cost is ultimately the number of primitive move actions.
    - This heuristic computes a lower bound on the remaining primitive actions needed, which
      is a lower bound on any HTN refinement cost. Methods have no intrinsic cost, so this
      lower bound is valid for the HTN.

    Implementation highlights:
    - Preprocess once: extract goal positions for each ring from model.goals (bitwise integer).
    - Build per-ring lists of all grounded on-facts to allow O(1) bitwise state checks.
    - Derive ring size order (smallest -> largest) from smallerThan relations (ring<ring). If not
      available, infer order from goal on-chains (r_i on r_{i+1}). Fall back to name sort.
    - Evaluate h(state) fast using bit operations: for the largest ring not at goal, return 2^k - 1.

    Notes about grounded fact format (CRITICAL):
    - Facts come as "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]". We parse both that and
      classic "(predicate arg1 arg2)" just to be robust.
    - Goals must be read from model.goals using model.state_explicit_repr(self.model.goals).
      Do NOT scan facts to find goal_on in init; these are not goals.
    """

    def __init__(self, towers_gpt5_heuristic_11=None, name="TowersGPT5Heuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}       # ring -> target object (from goal '+on[ring,target]')
        self.rings_by_size = []        # [r_smallest, ..., r_largest]
        self.on_facts_by_ring = {}     # ring -> List[(Fact, target_obj)] for all grounded on-facts of that ring
        self.num_rings = 0

        # Keep a quick set of ring names for filtering and speed
        self._rings_set = set()

    # ------------------------ Initialization & Preprocessing ------------------------

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        # Initial heuristic value
        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def _preprocess(self):
        # 1) Extract goals: model.goals is a bitwise integer. Convert to names, parse, and pick the '+on[ring,target]' pairs
        if hasattr(self.model, 'goals') and self.model.goals:
            for goal_name in self.model.state_explicit_repr(self.model.goals):
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == 'on' and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if self._is_ring(r):
                        self.goal_positions[r] = tgt

        # 2) Collect all rings and map each ring to its grounded on-facts for fast state lookup
        #    Also collect smallerThan relations among rings to derive ordering
        ring_smaller_graph = defaultdict(set)  # r_small -> set(r_larger)
        rings_seen = set()

        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == 'on':
                ring, target = args[0], args[1]
                if self._is_ring(ring):
                    rings_seen.add(ring)
                    self._rings_set.add(ring)
                    lst = self.on_facts_by_ring.get(ring)
                    if lst is None:
                        lst = []
                        self.on_facts_by_ring[ring] = lst
                    lst.append((fact, target))

            elif predicate == 'smallerThan':
                smaller, larger = args[0], args[1]
                if self._is_ring(smaller) and self._is_ring(larger):
                    rings_seen.add(smaller)
                    rings_seen.add(larger)
                    self._rings_set.add(smaller)
                    self._rings_set.add(larger)
                    ring_smaller_graph[smaller].add(larger)

        self.num_rings = len(rings_seen)

        # 3) Determine ordering smallest -> largest
        if self.num_rings == 0:
            self.rings_by_size = []
            return

        if any(ring_smaller_graph.values()):
            # Use counts of how many strictly larger rings each ring is smaller than (classic for Hanoi)
            counts = {r: len(ring_smaller_graph.get(r, set())) for r in rings_seen}
            # More larger rings above => ring is smaller
            self.rings_by_size = sorted(rings_seen, key=lambda r: -counts.get(r, 0))
        else:
            # Fallback: infer from goal chain if available: if on(r_i, r_{i+1}) in goal, then r_i < r_{i+1}
            inferred_smaller = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if self._is_ring(r) and self._is_ring(tgt):
                    inferred_smaller[r].add(tgt)
            # Compute transitive closure (simple fixed-point)
            changed = True
            while changed:
                changed = False
                for a in list(inferred_smaller.keys()):
                    to_add = set()
                    for b in inferred_smaller[a]:
                        to_add |= inferred_smaller.get(b, set())
                    new = to_add - inferred_smaller[a]
                    if new:
                        inferred_smaller[a] |= new
                        changed = True
            counts = {r: len(inferred_smaller.get(r, set())) for r in rings_seen}
            if any(counts.values()):
                self.rings_by_size = sorted(rings_seen, key=lambda r: -counts.get(r, 0))
            else:
                # Last resort: name sort ensures deterministic order
                self.rings_by_size = sorted(list(rings_seen))

    # ------------------------ Heuristic Computation ------------------------

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Return 2^k - 1, where k is the index (1-based) of the largest ring not at goal
        (counted from the smallest side). If all rings are at goal, return 0.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from largest to smallest: rings_by_size is [smallest ... largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                # If a ring has no explicit goal in model.goals, conservatively assume it's already OK
                # (cannot claim it must move, otherwise we may overestimate). This keeps admissibility.
                continue

            current_pos = self._get_current_position(state_bitwise, ring)
            if current_pos != goal_pos:
                # This ring (and all smaller ones below index i) must move at least once
                k = i + 1  # number of rings from smallest up to and including 'ring'
                return (1 << k) - 1

        return 0

    def _get_current_position(self, state_bitwise: int, ring: str):
        # Bitwise check through all grounded on-facts of this ring
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, target in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    # ------------------------ Utilities ------------------------

    def _is_ring(self, obj: str) -> bool:
        # In the provided instances all rings are named like r1, r2, ..., rN
        return isinstance(obj, str) and obj.startswith('r')

    def _parse_fact_name(self, name: str):
        """
        Parse fact in two tolerated formats:
        - Grounded: "+predicate[arg1,arg2]" (or with '-')
        - HDDL style: "(predicate arg1 arg2)"
        Returns: (predicate, [args...]) or (None, [])
        """
        if not name:
            return None, []
        s = name.strip()
        # Grounded form
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
            b = s.find("[")
            if b > 0 and s.endswith("]"):
                predicate = s[:b]
                args_str = s[b + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        # HDDL form
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    # ------------------------ Heuristic API ------------------------

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state)
        super().update_info(h)
        return h

    # ------------------------ Debug / Reporting ------------------------

    def __repr__(self):
        return f"TowersGPT5Heuristic11(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic11 with {self.num_rings} rings"

    def __output__(self):
        return (
            "TowersGPT5Heuristic11 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
