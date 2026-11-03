import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - The Towers domain is a classical hierarchical encoding of the 3-peg Towers of Hanoi.
      The primitive move action moves the current top ring from one object to another
      (object = ring or tower) obeying size constraints. The hierarchy (shiftTower ->
      selectDirection -> rotateTower/exchange -> move_abstract -> move) encodes the
      optimal recursive strategy for 3 pegs.
    - The key difficulty is that moving a larger ring requires clearing all smaller rings first.
      This induces an exponential lower bound: to be able to move the largest misplaced ring L,
      all rings smaller than L must be moved at least once. Thus a tight admissible lower bound
      is 2^k - 1 where k = number of rings smaller than L plus the ring L itself.
    - Goals are stored as a bitwise integer in model.goals; we must extract the intended final
      support (on-facts) for each ring from there. During search, we must check the current state
      via bitwise operations for speed.
    - We combine two admissible bounds and take their maximum:
        1) Exponential bound derived from the largest misplaced ring (as above)
        2) Count of misplaced rings (each misplaced ring must be moved at least once)
      The max remains admissible and often tighter than either alone.

    Efficiency considerations:
    - All expensive preprocessing is performed once in initialize().
    - During search we rely exclusively on bitwise checks and precompiled indices.

    Heuristic summary (admissible):
    h(n) = max( 2^k - 1 for the largest ring not at goal (k = #known-smaller rings + 1),
                 number of rings that are not on their goal support )
    """

    def __init__(self, towers_gpt5_heuristic_14=None, name: str = "TowersGPT5Heuristic14"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Will be filled at preprocessing
        self.goal_positions = {}          # ring -> target object (from goal on-facts)
        self.rings = set()                # set of ring names (e.g., {"r1", "r2", ...})
        self.on_facts_by_ring = {}        # ring -> list[(Fact, target_obj)] for every grounded on-fact
        self.num_smallers = defaultdict(int)  # ring -> number of (known) smaller rings (via transitive closure)
        self.num_rings = 0

        # Graphs for size ordering among rings
        self._smaller_edges = defaultdict(set)      # a -> set of b where a is smallerThan b (rings only)
        self._smaller_rev_edges = defaultdict(set)  # reverse edges for closure computation

    # -------------------------- Framework hooks --------------------------

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

    # ----------------------------- Preprocess -----------------------------

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Supports grounded format: "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]"
        and HDDL format: "(predicate arg1 arg2)".
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            core = name[1:]
            b = core.find("[")
            if b > 0 and core.endswith("]"):
                predicate = core[:b]
                args_str = core[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        # 1) Extract goal positions (from model.goals bitwise integer)
        if hasattr(self.model, "goals") and self.model.goals:
            for goal_name in self.model.state_explicit_repr(self.model.goals):
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):  # ring
                        self.goal_positions[ring] = target
                        self.rings.add(ring)
                        if target.startswith("r"):
                            # Goal chain implies size order: ring < target
                            self._add_smaller_edge(ring, target)

        # 2) Scan all grounded facts to collect rings, on-facts, and smallerThan ring-ring edges
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    self.rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

            elif predicate == "smallerThan":
                a, b = args[0], args[1]
                if a.startswith("r"):
                    self.rings.add(a)
                if b.startswith("r"):
                    self.rings.add(b)
                # Only ring-ring edges define ordering among rings
                if a.startswith("r") and b.startswith("r"):
                    self._add_smaller_edge(a, b)

        self.num_rings = len(self.rings)

        # 3) Compute transitive closure for number of smaller rings per ring
        self._compute_num_smallers()

    def _add_smaller_edge(self, smaller_ring: str, larger_ring: str):
        if smaller_ring == larger_ring:
            return
        self._smaller_edges[smaller_ring].add(larger_ring)
        self._smaller_rev_edges[larger_ring].add(smaller_ring)

    def _compute_num_smallers(self):
        """For each ring L, count how many rings are (known to be) smaller than L using
        transitive closure on the reverse smaller-than graph.
        """
        for L in self.rings:
            count = 0
            if L in self._smaller_rev_edges:
                # BFS/DFS in reverse graph to find all s with s ->* L
                visited = set()
                dq = deque(self._smaller_rev_edges[L])
                while dq:
                    s = dq.popleft()
                    if s in visited:
                        continue
                    visited.add(s)
                    count += 1
                    for pre in self._smaller_rev_edges.get(s, ()):  # continue transitively
                        if pre not in visited:
                            dq.append(pre)
            self.num_smallers[L] = count

    # ------------------------------ Heuristic ------------------------------

    def _get_current_position(self, state_bitwise: int, ring: str):
        """Return the current support object for a ring by checking 'on[ring, obj]' facts.
        Uses fast bitwise checks. Returns None if not found.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, target in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _largest_misplaced_k(self, state_bitwise: int) -> int:
        """Find k = (#known smaller rings of L) + 1 for the largest misplaced ring L.
        Only rings that appear in goal_positions are considered (others have no goal constraint).
        Returns 0 if all such rings are at goal positions.
        """
        k_best = 0
        for ring, goal in self.goal_positions.items():
            current = self._get_current_position(state_bitwise, ring)
            if current != goal:
                # This ring must eventually be moved. Its impact lower bound is 2^k - 1
                # where k = (#known smaller rings) + 1 (including itself).
                k = self.num_smallers.get(ring, 0) + 1
                if k > k_best:
                    k_best = k
        return k_best

    def _misplaced_count(self, state_bitwise: int) -> int:
        """Count how many rings with goal constraints are not currently on their goal support."""
        cnt = 0
        for ring, goal in self.goal_positions.items():
            current = self._get_current_position(state_bitwise, ring)
            if current != goal:
                cnt += 1
        return cnt

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if self.num_rings == 0 or not self.goal_positions:
            return 0

        # Bound 1: Exponential bound from the largest misplaced ring
        k = self._largest_misplaced_k(state_bitwise)
        bound_exp = (1 << k) - 1 if k > 0 else 0

        # Bound 2: At least one move per misplaced ring
        bound_cnt = self._misplaced_count(state_bitwise)

        return max(bound_exp, bound_cnt)

    # ------------------------------ Introspection ------------------------------

    def __repr__(self):
        return f"{self.name}(rings={self.num_rings})"

    def __str__(self):
        return f"{self.name} with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings: {sorted(self.rings, key=lambda r: self.num_smallers.get(r, 0))}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
