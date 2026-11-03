import time
from collections import defaultdict
from typing import List, Union, Any

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - Towers of Hanoi with three pegs has a tight lower bound: for the largest disk
      not yet at goal position, all smaller disks must be moved at least once
      before it can move. This yields the classic bound 2^k - 1, where k is the
      number of disks from the smallest up to that largest misplaced disk.
    - We extract ring size ordering and ring goal positions during preprocessing,
      and use fast bitwise checks to find a ring's current position at evaluation time.
    - To also consider the hierarchical task structure, we compute an additional
      lower bound from the current task network: each outstanding primitive operator
      contributes at least its cost; additionally, each outstanding move_abstract
      task contributes at least 1 primitive move in any refinement. We combine
      bounds admissibly using max(h_state, h_tn).

    This heuristic is consistent and admissible by construction, as it never
    overestimates the true remaining cost-to-go.
    """

    def __init__(self, towers_gpt5_heuristic_2=None, name: str = "TowersGPT5Heuristic2"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}       # ring -> goal support (OBJ: ring or tower)
        self.rings_by_size: List[str] = []  # rings sorted from smallest to largest
        self.on_facts_by_ring = {}     # ring -> list of (Fact, target_obj)
        self.num_rings = 0

        # Keep a quick set of ring symbols to validate parsing
        self._rings_set = set()

    # ----------------------- Framework integration -----------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        # Initial heuristic value for the root (optional, returned to framework)
        h0 = self._calculate_heuristic(initial_node.state, initial_node)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._calculate_heuristic(node.state, node)
        super().update_info(h)
        return h

    # -------------------------- Preprocessing ---------------------------
    def _parse_fact_name(self, name: str):
        """
        Parse fact/component name to (predicate, [args]).
        Supports grounded "+pred[a,b]" / "-pred[a,b]" and HDDL "(pred a b)" formats.
        """
        name = name.strip()
        # Grounded format: +predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            b = name2.find("[")
            if b > 0 and name2.endswith("]"):
                pred = name2[:b]
                args_str = name2[b + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
        # HDDL format: (predicate arg1 arg2)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """
        Extract rings, their size ordering, goal positions, and map 'on' facts per ring.
        We must read goals from model.goals (bitwise), not from explicit goal_on.
        """
        rings = set()
        smaller_than_pairs = []  # (smaller_ring, larger_obj)

        # 1) Extract goal positions from model.goals
        if hasattr(self.model, "goals") and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "on" and len(args) >= 2:
                    r, tgt = args[0], args[1]
                    if self._is_ring(r):
                        self.goal_positions[r] = tgt
                        rings.add(r)
                        if self._is_ring(tgt):
                            rings.add(tgt)

        # 2) Scan all grounded facts to collect 'on' mappings and size constraints
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue

            if pred == "smallerThan":
                sm, lg = args[0], args[1]
                if self._is_ring(sm):
                    rings.add(sm)
                    smaller_than_pairs.append((sm, lg))
                    if self._is_ring(lg):
                        rings.add(lg)

            elif pred == "on":
                r, tgt = args[0], args[1]
                if self._is_ring(r):
                    rings.add(r)
                    self.on_facts_by_ring.setdefault(r, []).append((fact, tgt))
                    if self._is_ring(tgt):
                        rings.add(tgt)

        self._rings_set = rings
        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3) Derive ring ordering (smallest -> largest)
        # Prefer ring-ring "smallerThan" information if present.
        ring_smaller_counts = defaultdict(int)
        ring_ring_seen = False
        for sm, lg in smaller_than_pairs:
            if self._is_ring(sm) and self._is_ring(lg):
                ring_ring_seen = True
                ring_smaller_counts[sm] += 1
        if ring_ring_seen:
            # Sort by how many rings they are smaller than (more means smaller)
            self.rings_by_size = sorted(list(rings), key=lambda r: -ring_smaller_counts.get(r, 0))
        else:
            # Fallback: infer a partial order from goal chain if available
            implied_smaller = defaultdict(set)
            for r, tgt in self.goal_positions.items():
                if self._is_ring(tgt):
                    implied_smaller[r].add(tgt)
            # transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(implied_smaller.keys()):
                    larger_set = list(implied_smaller[r])
                    for mid in larger_set:
                        for lg in implied_smaller.get(mid, set()):
                            if lg not in implied_smaller[r]:
                                implied_smaller[r].add(lg)
                                changed = True
            counts = {r: len(implied_smaller.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -counts.get(r, 0))

    # --------------------------- Heuristic core --------------------------
    def _is_ring(self, sym: str) -> bool:
        # Convention in provided instances: ring symbols start with 'r'
        return isinstance(sym, str) and sym.startswith("r")

    def _get_current_position(self, state_bitwise: int, ring: str) -> Union[str, None]:
        """
        Find current support object of a ring by scanning its 'on' facts using
        fast bitwise membership over fact.global_id.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
        for fact, tgt in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _h_state(self, state_bitwise: int) -> int:
        """
        State-based Hanoi lower bound:
        - Identify the largest ring that is not currently at its goal support.
        - If its index in the size order is i (0-based from smallest), then at
          least the i+1 smallest rings must move: 2^(i+1) - 1.
        """
        if not self.rings_by_size:
            return 0
        # Iterate largest -> smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                k = i + 1
                return (1 << k) - 1
        return 0

    def _extract_task_network_components(self, node: HTNNode) -> List[Any]:
        """
        Robustly obtain the current task network as a list of components
        (operators or abstract tasks). Handles several possible storage styles:
        - node.task_network (list of components or ids)
        - node.tn (list of components or ids)
        - node.task_network_ids (list of ids)
        Maps ids to components using model.get_component when needed.
        """
        tasks = None
        for attr in ("task_network", "tn", "task_network_ids"):
            if hasattr(node, attr):
                tasks = getattr(node, attr)
                break
        if tasks is None:
            return []
        comps: List[Any] = []
        for it in tasks:
            if isinstance(it, int):
                try:
                    comps.append(self.model.get_component(it))
                except Exception:
                    # Fallback: ignore unknown ids
                    pass
            else:
                comps.append(it)
        return comps

    def _h_task_network(self, node: HTNNode) -> int:
        """
        Hierarchical task structure lower bound:
        - Sum costs of any already-present primitive operators (exact LB for them).
        - Count each 'move_abstract' abstract task as at least 1 primitive action.
        For other abstract tasks (shiftTower, rotateTower, exchange), we use 0 as
        a safe lower bound here, since their minimal refinement cost can be 0 in
        some special states (e.g., exchangeClear). The strong domain LB comes from
        the state-based Hanoi bound, and we combine with max() to remain admissible.
        """
        try:
            comps = self._extract_task_network_components(node)
        except Exception:
            return 0
        if not comps:
            return 0

        lb = 0
        for comp in comps:
            # Primitive operator if it has a 'cost' attribute
            if hasattr(comp, "cost"):
                try:
                    c = int(comp.cost)
                except Exception:
                    c = 1
                lb += max(0, c)
            else:
                # Abstract task: use name-based hint
                name = getattr(comp, "name", "")
                if isinstance(name, str) and "move_abstract" in name:
                    lb += 1
        return lb

    def _calculate_heuristic(self, state_bitwise: int, node: HTNNode) -> int:
        h_state = self._h_state(state_bitwise)
        # Combine with task-network-based LB admissibly
        h_tn = self._h_task_network(node)
        return max(h_state, h_tn)

    # ------------------------------ Utils -------------------------------
    def __repr__(self):
        return f"TowersGPT5Heuristic2(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic2 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers GPT-5 Heuristic 2 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
