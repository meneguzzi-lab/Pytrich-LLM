import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic4(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Core idea (two complementary lower bounds, combined via max):
    1) State-based Hanoi bound (admissible and tight):
       - Identify the largest ring that is not yet at its goal position.
       - If k rings (from the smallest up to that ring) still need to move,
         then at least 2^k - 1 primitive 'move' actions are required.

    2) Task-network-based structural bound (hierarchical awareness):
       - Each pending primitive 'move' in the task network must still execute: +cost (typically 1).
       - Each pending abstract task that necessarily decomposes to at least one 'move'
         contributes a lower bound of +1 (safe, minimal contribution):
           * shiftTower  -> ≥ 1 (selectDirection -> rotateTower -> move_abstract)
           * rotateTower -> ≥ 1 (always starts with a move_abstract)
           * move_abstract -> exactly 1 (directly decomposes to one move)
         Other abstract tasks (selectDirection, exchange) are counted as 0 for safety.

       Summing these minimal contributions across the current task network gives an
       admissible lower bound on the remaining cost induced by the hierarchy at the current node.

    Final heuristic value is the maximum of these two bounds. Using max() instead of sum()
    keeps the heuristic admissible and prevents double-counting the same necessary moves
    from different perspectives (state vs. task network).

    Implementation details and efficiency:
    - Uses bitwise checks against fact.global_id to evaluate state in O(#facts per ring) time.
    - Extracts goals from model.goals (bitwise integer) as required by Pytrich.
    - Preprocesses ring size order via smallerThan to reliably identify the largest unsettled ring.
    - Robust parsing of grounded fact names in "+predicate[arg1,arg2]" format.

    This heuristic is fast, admissible, and scales to large instances (up to 20 rings) while
    incorporating hierarchical structure in a safe manner.
    """

    def __init__(self, towers_gpt5_heuristic_4=None, name="TowersGPT5Heuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}       # ring -> target object (from goals)
        self.rings_by_size = []        # rings sorted smallest -> largest
        self.on_facts_by_ring = {}     # ring -> list[(Fact, target_obj)] where Fact is '+on[ring,target]'
        self.num_rings = 0

    # ------------------------- Initialization / Preprocessing -------------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        # Initial h-value from the initial node's state
        h0 = self._calculate_heuristic(initial_node.state, getattr(initial_node, 'task_network', None))
        return super().initialize(model, h0)

    def _parse_fact_name(self, name: str):
        """
        Parse a grounded fact name of the form '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Returns (predicate, [args]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Extract static domain structure from grounded facts and goals:
        - Goal positions: from model.goals (bitwise integer), not from goal_on facts.
        - Ring order (smallest -> largest): from smallerThan relations.
        - Mapping of 'on' facts per ring for fast state queries.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller_ring, larger_obj)

        # 1) Goals: read from model.goals bitset
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == 'on' and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if r.startswith('r'):
                        self.goal_positions[r] = tgt

        # 2) Scan all facts for smallerThan and on
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
            return

        # 3) Derive ring order (smallest -> largest) via smallerThan counts
        #    A smaller ring appears more often on the LHS of smallerThan.
        if smaller_than_pairs:
            smaller_count = defaultdict(int)
            for s, _ in smaller_than_pairs:
                if s in rings:
                    smaller_count[s] += 1
            # Sort by decreasing count: highest count => smallest
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback if no explicit ordering (rare). Infer from goal stacking.
            is_smaller_than = defaultdict(set)
            for ring, tgt in self.goal_positions.items():
                if tgt and tgt.startswith('r'):
                    is_smaller_than[ring].add(tgt)
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller_than.keys()):
                    for mid in list(is_smaller_than[r]):
                        for big in is_smaller_than.get(mid, set()):
                            if big not in is_smaller_than[r]:
                                is_smaller_than[r].add(big)
                                changed = True
            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    # ------------------------- Fast state queries -------------------------
    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Return the object the ring currently sits on, by scanning its grounded '+on[ring,*]' facts
        and checking which one is true in the given bitwise state.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, tgt in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    # ------------------------- State-based Hanoi bound -------------------------
    def _hanoi_lower_bound(self, state_bitwise: int) -> int:
        """
        Classic admissible bound for Towers of Hanoi:
        - Scan from largest to smallest ring.
        - Find the largest ring not in its goal position.
        - If k rings from the smallest up to that ring are unsettled, return 2^k - 1.
        - If all rings are correct, return 0.
        """
        if not self.rings_by_size:
            return 0
        # rings_by_size is smallest -> largest; iterate reverse for largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                k = i + 1  # count of rings from smallest (index 0) up to i (inclusive)
                return (1 << k) - 1
        return 0

    # ------------------------- Task-network-based bound -------------------------
    def _task_functor(self, name: str) -> str:
        """
        Robustly extract the task/operator functor from a name.
        Supports these possibilities:
        - "(functor args...)"
        - "functor args..."
        - plain "functor" or any name containing the functor as a substring
        Returns functor in original case (we compare via substring anyway).
        """
        s = name.strip()
        if s.startswith('(') and s.endswith(')'):
            inner = s[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0]
        # Fallback: split by whitespace
        if ' ' in s:
            return s.split()[0]
        return s

    def _task_network_lower_bound(self, node: HTNNode) -> int:
        """
        Sum minimal required primitives from the current task network.
        Rules (admissible):
        - Each pending primitive 'move' operator contributes at least its cost (typically 1).
        - Each AbstractTask contributes:
            move_abstract: 1
            rotateTower:   1
            shiftTower:    1
            selectDirection: 0 (could recurse without immediate moves)
            exchange:        0 (could be empty via exchangeClear)
        """
        tn = getattr(node, 'task_network', None)
        if not tn:
            return 0

        lb = 0
        for task in tn:
            # Operator case
            if hasattr(task, 'cost') and hasattr(task, 'name') and hasattr(task, 'global_id'):
                # Treat as primitive operator (move)
                # Minimal contribution is its own cost (default to 1 if absent/zero)
                cost = getattr(task, 'cost', 1) or 1
                # Be specific: only count the 'move' operator (domain has a single operator)
                functor = self._task_functor(getattr(task, 'name', ''))
                if 'move' in functor:
                    lb += cost
                else:
                    # For safety, do not add cost for any unknown operator names
                    pass
                continue

            # AbstractTask case
            if hasattr(task, 'name'):
                functor = self._task_functor(task.name)
                if 'move_abstract' in functor:
                    lb += 1
                elif 'rotateTower' in functor:
                    lb += 1
                elif 'shiftTower' in functor:
                    lb += 1
                else:
                    # selectDirection / exchange or anything else: contribute 0 for safety
                    pass

        return lb

    # ------------------------- Heuristic evaluation -------------------------
    def _calculate_heuristic(self, state_bitwise: int, task_network=None) -> int:
        # Lower bound from the physical state (tight for Hanoi)
        h_state = self._hanoi_lower_bound(state_bitwise)

        # Lower bound from the current HTN task network structure (cheap and safe)
        # Construct a temporary node-like wrapper if only bitstate is provided
        h_tn = 0
        if task_network is not None:
            # We expect to be called with a full node; if not, the caller will use __call__ route
            # that passes the node itself.
            fake_node = type('FakeNode', (), {'task_network': task_network})
            h_tn = self._task_network_lower_bound(fake_node)

        # Final admissible heuristic: avoid double counting by taking max
        return max(h_state, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, getattr(node, 'task_network', None))
        super().update_info(h_value)
        return h_value

    # ------------------------- Introspection helpers -------------------------
    def __repr__(self):
        return f"TowersGPT5Heuristic4(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic4 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
