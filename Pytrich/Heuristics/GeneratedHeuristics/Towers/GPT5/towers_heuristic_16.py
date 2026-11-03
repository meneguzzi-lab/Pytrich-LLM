import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic16(Heuristic):
    """
    Domain-dependent heuristic for the Towers (HDDL Towers of Hanoi) domain.

    Rationale and domain insights:
    - The only primitive operator is 'move', which moves a single ring between supports
      (towers or larger rings). Every plan cost equals the number of 'move' actions.
    - Towers of Hanoi requires (at least) 2^k - 1 moves to reposition the largest
      misplaced ring together with all smaller rings above it. This is a classic,
      admissible lower bound that is tight for standard three-peg Hanoi.
    - The HTN hierarchy (shiftTower -> selectDirection -> rotateTower -> move_abstract -> move)
      does not introduce additional primitive cost beyond the number of 'move' actions.
      Therefore, the lower bound can be computed purely from the current state's ring placements
      relative to the goal placements (the goal is encoded by '+on[ring,target]' facts).

    Heuristic definition:
    - Preprocessing extracts for each ring r its unique goal support (target object) from Model.goals.
    - Rings are ordered by size (smallest to largest) using smallerThan facts from the initial state
      (or a safe fallback if needed).
    - During search, we identify the largest ring that is not yet on its goal support.
      If its index in the size ordering is i (0-based from smallest), at least 2^(i+1) - 1 moves
      are still necessary (to free i smaller rings, move this ring, and restack those i rings).

    Properties:
    - Admissible: it never overestimates the remaining number of 'move' actions.
    - Consistent: the estimate does not increase after applying an action.
    - Fast: all heavy lifting is preprocessed; state checks use bitwise tests on fact IDs.

    Implementation notes (critical to Pytrich):
    - Goal facts must be read from the bitwise integer model.goals using model.state_explicit_repr.
    - State membership must use bitwise checks (state >> fact.global_id) & 1.
    - Grounded fact strings have the form '+predicate[arg1,arg2]'. We parse these robustly.
    """

    def __init__(self, towers_gpt5_heuristic_16=None, name="TowersGPT5Heuristic16"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}           # ring -> goal target (OBJ)
        self.rings_by_size = []            # [r_smallest, ..., r_largest]
        self.ring_index = {}               # ring -> index in rings_by_size (0=smallest)
        self.on_facts_by_ring = {}         # ring -> list[(Fact f, target_obj)] for all 'on[ring,target]'
        self.goal_on_fact_by_ring = {}     # ring -> Fact corresponding to on[ring, goal_target]
        self.num_rings = 0
        self.model = None

    # -------------- Framework hooks --------------

    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        # Initial node heuristic
        h0 = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    # -------------- Preprocessing --------------

    def _preprocess(self):
        # 1) Extract goal placements from model.goals (bitwise integer)
        self._extract_goal_positions_from_goals()

        # 2) Build 'on' fact index per ring (from grounded facts universe)
        self._index_on_facts()

        # 3) If possible, map each ring directly to the specific 'on[ring,goal]' fact
        self._map_goal_on_facts()

        # 4) Determine rings and their size order using smallerThan facts from initial state
        self._build_ring_order()

        self.num_rings = len(self.rings_by_size)

    def _extract_goal_positions_from_goals(self):
        self.goal_positions.clear()
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_fact_names:
                pred, args = self._parse_fact_name(name)
                if pred == 'on' and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith('r'):
                        self.goal_positions[r] = tgt

        # Fallback: If goals bitset didn't expose 'on' goals, try 'goal_on' from initial state
        if not self.goal_positions and hasattr(self.model, 'initial_state'):
            for name in self.model.state_explicit_repr(self.model.initial_state):
                pred, args = self._parse_fact_name(name)
                if pred == 'goal_on' and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith('r'):
                        self.goal_positions[r] = tgt

    def _index_on_facts(self):
        # Build mapping: ring -> all possible grounded on[ring,*] facts (Fact object + target string)
        self.on_facts_by_ring.clear()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred != 'on' or len(args) < 2:
                continue
            r, tgt = args[0], args[1]
            if not r.startswith('r'):
                continue
            self.on_facts_by_ring.setdefault(r, []).append((fact, tgt))

    def _map_goal_on_facts(self):
        # For quick state checks: ring at goal? Use a single bit test for on[ring,goal_target]
        self.goal_on_fact_by_ring.clear()
        for r, tgt in self.goal_positions.items():
            for f, cand in self.on_facts_by_ring.get(r, []):
                if cand == tgt:
                    self.goal_on_fact_by_ring[r] = f
                    break

    def _build_ring_order(self):
        # Prefer using smallerThan facts that are TRUE in the initial state (static ordering)
        ring_set = set()
        smaller_rel = []  # (small, large) where both are rings

        if hasattr(self.model, 'initial_state'):
            for name in self.model.state_explicit_repr(self.model.initial_state):
                pred, args = self._parse_fact_name(name)
                if pred == 'smallerThan' and len(args) >= 2:
                    a, b = args[0], args[1]
                    if a.startswith('r'):
                        ring_set.add(a)
                        if b.startswith('r'):
                            ring_set.add(b)
                            smaller_rel.append((a, b))
                elif pred == 'on' and len(args) >= 2:
                    a, b = args[0], args[1]
                    if a.startswith('r'):
                        ring_set.add(a)
                        if b.startswith('r'):
                            ring_set.add(b)

        # If we still have no rings (extreme edge case), infer from goal positions
        if not ring_set and self.goal_positions:
            for r, tgt in self.goal_positions.items():
                ring_set.add(r)
                if tgt and tgt.startswith('r'):
                    ring_set.add(tgt)

        if not ring_set:
            # Nothing to do
            self.rings_by_size = []
            self.ring_index = {}
            return

        if smaller_rel:
            # Sort rings by how many larger rings each is smaller than (smallest has highest count)
            smaller_count = defaultdict(int)
            for s, l in smaller_rel:
                if s in ring_set:
                    smaller_count[s] += 1
            # Default 0 for rings appearing only as larger
            self.rings_by_size = sorted(list(ring_set), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: infer partial order from goal chain (r on r_next => r < r_next)
            covers = defaultdict(set)  # r -> set of rings r is smaller than
            for r, tgt in self.goal_positions.items():
                if tgt and tgt.startswith('r'):
                    covers[r].add(tgt)
                    ring_set.add(r)
                    ring_set.add(tgt)
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(covers.keys()):
                    to_add = set()
                    for x in list(covers[r]):
                        to_add |= covers.get(x, set())
                    before = len(covers[r])
                    covers[r] |= to_add
                    if len(covers[r]) > before:
                        changed = True
            smaller_count = {r: len(covers.get(r, set())) for r in ring_set}
            self.rings_by_size = sorted(list(ring_set), key=lambda r: -smaller_count.get(r, 0))

        # Index map
        self.ring_index = {r: i for i, r in enumerate(self.rings_by_size)}

    # -------------- Heuristic core --------------

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        if not self.rings_by_size:
            return 0

        # Inspect rings from largest to smallest (reverse of smallest->largest list)
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]

            # If we know the dedicated on[ring,goal] fact, test it directly
            f_goal = self.goal_on_fact_by_ring.get(r)
            if f_goal is not None:
                at_goal = (state_bitwise >> f_goal.global_id) & 1
                if not at_goal:
                    # Need to move i+1 rings at least: 2^(i+1) - 1
                    return (1 << (i + 1)) - 1
                continue

            # Fallback (rare): read current position and compare to goal
            curr_pos = self._get_current_position(state_bitwise, r)
            goal_pos = self.goal_positions.get(r)
            if curr_pos != goal_pos:
                return (1 << (i + 1)) - 1

        # All rings are on their goal supports
        return 0

    # -------------- Utilities --------------

    def _get_current_position(self, state_bitwise: int, ring: str):
        facts = self.on_facts_by_ring.get(ring, [])
        for f, tgt in facts:
            if (state_bitwise >> f.global_id) & 1:
                return tgt
        return None

    @staticmethod
    def _parse_fact_name(name: str):
        # Grounded form: '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
            b = s.find('[')
            if b > 0 and s.endswith(']'):
                pred = s[:b]
                args_str = s[b+1:-1]
                args = args_str.split(',') if args_str else []
                return pred, args
        # Also support (predicate arg1 arg2)
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    # -------------- Introspection --------------

    def __repr__(self):
        return f"TowersGPT5Heuristic16(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic16 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic16 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
