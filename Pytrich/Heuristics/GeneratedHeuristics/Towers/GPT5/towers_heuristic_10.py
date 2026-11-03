import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set, Union

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.model import Operator, AbstractTask


class TowersGPT5Heuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) HDDL domain.

    Design rationale and domain insights:
    - The domain encodes a classic three-peg Towers of Hanoi problem using HTN methods
      (shiftTower, selectDirection, rotateTower, exchange) that eventually reduce to
      the primitive action 'move'. The number of moves required to relocate a stack of
      k rings is known to be 2^k - 1 in the optimal solution.
    - We compute a robust, admissible lower bound based on the largest ring that is
      not yet at its goal location. To move that ring, all smaller rings must be moved
      at least once, giving a tight lower bound of 2^k - 1 where k counts rings from
      the smallest up to that largest misplaced ring.
    - We also leverage the hierarchical task structure when available: every pending
      abstract task 'move_abstract' must eventually decompose into exactly one
      primitive move. Counting how many such tasks are present in the current task
      network provides another admissible lower bound. To remain admissible when
      combined with the state-based bound, we take the max of both.

    Implementation notes:
    - Uses efficient bitwise checks to test state membership (critical for performance).
    - Extracts ring ordering from 'smallerThan' facts to sort rings from smallest to largest.
    - Reads goal facts from model.goals (bitwise) using Model.state_explicit_repr, as required.
    - Robust parsing for fact/operator names that may appear either in grounded format
      '+predicate[arg1,arg2]' or HDDL format '(predicate arg1 arg2)'.

    Admissibility & consistency:
    - The 2^k - 1 bound is a classic admissible (and consistent) lower bound for Towers.
    - The additional bound from counting pending 'move_abstract' tasks is also a lower
      bound; we combine bounds with max(..) to stay admissible.
    - This heuristic scales well with the number of rings (e.g., up to 20+), as it
      performs only simple checks per ring.
    """

    def __init__(self, towers_gpt5_heuristic_10=None, name: str = "TowersGPT5Heuristic10"):
        super().__init__(name=name)
        self.preprocessing_time: float = 0.0

        # Preprocessed domain knowledge
        self.model: Optional[Model] = None
        self.goal_positions: Dict[str, str] = {}          # ring -> target object (goal)
        self.rings_by_size: List[str] = []                 # smallest -> largest
        self.on_facts_by_ring: Dict[str, List[Tuple[object, str]]] = {}  # ring -> list of (Fact, target)
        self.num_rings: int = 0

        # For fast lookup of ring/ring partial order if needed
        self._ring_set: Set[str] = set()

    # ------------------------------- Initialization -------------------------------
    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # Preprocess domain structure and cache everything needed for quick heuristic computation
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Initial h-value for root
        h_value = self._calculate_heuristic(initial_node.state, initial_node)
        return super().initialize(model, h_value)

    # ------------------------------- Preprocessing -------------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse a fact/operator name into (predicate, [args...]).
        Handles two formats:
        - Grounded: '+predicate[arg1,arg2]' / '-predicate[arg1,arg2]'
        - HDDL-like: '(predicate arg1 arg2)'
        Returns (None, []) if parsing fails.
        """
        if not name:
            return None, []
        s = name.strip()

        # Grounded format
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
            bracket_idx = s.find("[")
            if bracket_idx > 0 and s.endswith("]"):
                predicate = s[:bracket_idx]
                args_str = s[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # HDDL-like format
        if s.startswith("(") and s.endswith(")"):
            inner = s[1:-1].strip()
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        if self.model is None:
            return

        rings: Set[str] = set()
        ring_graph: Dict[str, Set[str]] = defaultdict(set)  # ring -> set of larger rings
        indeg: Dict[str, int] = defaultdict(int)            # indegree for topo sort (only ring->ring)

        # 1) Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, 'goals') and self.model.goals:
            for goal_name in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(goal_name)
                if pred == 'on' and len(args) >= 2:
                    r, target = args[0], args[1]
                    if r.startswith('r'):
                        self.goal_positions[r] = target

        # 2) Scan all facts to gather:
        #    - set of rings
        #    - 'on' fact handles per ring
        #    - ring->ring ordering via 'smallerThan'
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None or len(args) < 2:
                continue

            if pred == 'on':
                r, target = args[0], args[1]
                if r.startswith('r'):
                    rings.add(r)
                    if r not in self.on_facts_by_ring:
                        self.on_facts_by_ring[r] = []
                    self.on_facts_by_ring[r].append((fact, target))

            elif pred == 'smallerThan':
                small, big = args[0], args[1]
                if small.startswith('r'):
                    rings.add(small)
                    # Only consider ring->ring relations for size ordering
                    if big.startswith('r'):
                        rings.add(big)
                        if big not in ring_graph[small]:
                            ring_graph[small].add(big)
                            indeg[big] += 1
                            if small not in indeg:
                                indeg[small] = 0

        self._ring_set = set(rings)
        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3) Topologically sort rings small->large using the ring_graph (Kahn's algorithm)
        self.rings_by_size = self._topo_sort_rings(rings, ring_graph, indeg)
        if not self.rings_by_size:
            # Fallback: order by how many 'smallerThan ring X' entries each ring appears as LHS
            smaller_count = defaultdict(int)
            for small in ring_graph:
                for big in ring_graph[small]:
                    smaller_count[small] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _topo_sort_rings(self, rings: Set[str], graph: Dict[str, Set[str]], indeg: Dict[str, int]) -> List[str]:
        # Ensure all rings in indeg
        for r in rings:
            indeg.setdefault(r, 0)

        q = deque([r for r in rings if indeg[r] == 0])
        order: List[str] = []
        local_graph: Dict[str, Set[str]] = {k: set(v) for k, v in graph.items()}
        local_indeg: Dict[str, int] = dict(indeg)

        while q:
            u = q.popleft()
            order.append(u)
            for v in list(local_graph.get(u, [])):
                local_indeg[v] -= 1
                local_graph[u].remove(v)
                if local_indeg[v] == 0:
                    q.append(v)

        if len(order) == len(rings):
            # order is from smallest (no smaller predecessors) to largest
            return order
        # Cycle or missing info
        return []

    # ------------------------------- Helpers -------------------------------
    def _get_current_position(self, state_bitwise: int, ring: str) -> Optional[str]:
        """
        Return the object the ring is currently on by checking which 'on' fact holds.
        Uses bitwise checks for speed.
        """
        entries = self.on_facts_by_ring.get(ring)
        if not entries:
            return None
        for fact, target in entries:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _largest_misplaced_index(self, state_bitwise: int) -> int:
        """
        Return index (in rings_by_size) of the largest ring that is not at its goal
        position, or -1 if all rings are at their goals.
        rings_by_size is small->large, so we iterate from large->small.
        """
        if not self.rings_by_size:
            return -1

        for i in range(len(self.rings_by_size) - 1, -1, -1):
            r = self.rings_by_size[i]
            cur = self._get_current_position(state_bitwise, r)
            goal = self.goal_positions.get(r)
            if cur != goal:
                return i
        return -1

    def _lower_bound_from_tasks(self, node: HTNNode) -> int:
        """
        Hierarchical (task-network) lower bound component.
        - Count how many pending 'move_abstract' tasks exist in the current task network.
          Each such abstract task must decompose into exactly one primitive move.
        - If access to the task network is unavailable or unclear, return 0.
        This is combined with the state-based bound via max(..) to stay admissible.
        """
        if self.model is None:
            return 0

        # Try to obtain task network from node with different attribute names (be robust)
        tn = getattr(node, 'task_network', None)
        if tn is None:
            tn = getattr(node, 'tn', None)
        if tn is None or not isinstance(tn, list):
            return 0

        lb = 0
        for item in tn:
            comp = None
            # Items may be component IDs or the components themselves
            if isinstance(item, int):
                try:
                    comp = self.model.get_component(item)
                except Exception:
                    comp = None
            else:
                comp = item

            if comp is None:
                continue

            # Only count AbstractTask explicitly named 'move_abstract'
            if isinstance(comp, AbstractTask):
                name = getattr(comp, 'name', '') or ''
                # Normalize possible formats
                pred, _ = self._parse_fact_name(name) if name else (None, [])
                # Either the raw name is 'move_abstract' or parsed predicate is 'move_abstract'
                if 'move_abstract' in name or (pred == 'move_abstract'):
                    lb += 1
        return lb

    # ------------------------------- Heuristic core -------------------------------
    def _calculate_heuristic(self, state_bitwise: int, node: Optional[HTNNode] = None) -> int:
        if not self.rings_by_size:
            return 0

        # State-based Hanoi lower bound: 2^k - 1 for largest misplaced ring
        idx = self._largest_misplaced_index(state_bitwise)
        if idx < 0:
            h_state = 0
        else:
            k = idx + 1  # number of rings from smallest (index 0) up to this ring (inclusive)
            h_state = (1 << k) - 1

        # Task-network-based bound (only counts pending 'move_abstract' tasks)
        h_tasks = 0
        if node is not None:
            try:
                h_tasks = self._lower_bound_from_tasks(node)
            except Exception:
                h_tasks = 0

        # Combine admissibly
        return max(h_state, h_tasks)

    # ------------------------------- Heuristic API -------------------------------
    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state, node)
        super().update_info(h_value)
        return h_value

    # ------------------------------- Introspection -------------------------------
    def __repr__(self):
        return f"TowersGPT5Heuristic10(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic10 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"TowersGPT5Heuristic10 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
