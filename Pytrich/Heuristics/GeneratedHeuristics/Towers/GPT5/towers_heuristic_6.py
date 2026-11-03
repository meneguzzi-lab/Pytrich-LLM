import time
from collections import defaultdict, deque
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGPT5Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) HDDL domain.

    Design rationale and domain insights:
    - Primitive cost is induced solely by the action 'move'. All abstract tasks
      (shiftTower, selectDirection, rotateTower, exchange, move_abstract) decompose
      to sequences where only 'move' contributes non-zero cost. Therefore, the
      cost-to-go equals the number of remaining 'move' operators required.
    - The classic minimal number of moves to place the largest misplaced ring
      correctly (together with all smaller rings that must be moved out of the
      way) is 2^k - 1, where k is the count of rings from smallest up to that
      largest misplaced ring. This yields a tight admissible lower bound.
    - To partially account for hierarchical structure without risking
      inadmissibility, we compute a separate lower bound from the current
      task-network using minimal guaranteed costs of abstract tasks
      (e.g., rotateTower >= 1 due to mandatory move_abstract). We combine this
      with the state-based bound via max(), preserving admissibility while adding
      guidance when the state-bound is weak early in the search.

    Heuristic returned: max(state_lower_bound, tasknet_lower_bound)
      - state_lower_bound = 2^k - 1 for the largest ring not at its goal place
      - tasknet_lower_bound = sum of minimal costs implied by pending tasks
        (safe, tiny estimates; we never exceed the true remaining cost because
         these are independent, method-level minima and we take max with the
         state lower bound rather than summing them together)

    Implementation details:
    - We preprocess once:
        * Extract goal positions from model.goals using model.state_explicit_repr
          (critical: goals stored bitwise!)
        * Build ring ordering (smallest -> largest) from smallerThan ring-ring
          relations if available; otherwise fall back to relations implied by
          on-chains in goals/initial or a safe counting fallback.
        * Map each ring to its grounded 'on' facts for fast bitwise checking.
        * Cache IDs and minimal costs for abstract tasks by name to compute a
          cheap task-network-based bound.
    - During heuristic calls:
        * Compute the classic 2^k - 1 bound by finding the largest ring not yet
          at its goal position using fast bitwise membership checks.
        * Optionally read node.task_network (if provided by the planner) and
          compute a minimal guaranteed number of moves remaining induced by
          pending tasks: rotateTower/selectDirection/shiftTower/move_abstract
          each contribute at least 1; exchange contributes 0 (it can decompose
          to empty subtasks in the given domain). Primitive 'move' operators
          in the task network contribute exactly 1 each.
        * Return the max of both bounds (admissible and typically tighter than
          either bound alone). This keeps the search focused while remaining fast.

    Notes on grounded fact parsing:
    - Facts may appear as '+predicate[arg1,arg2]' (bitwise grounded form) or
      '(predicate arg1 arg2)' (HDDL-like pretty form). We accept both.

    Performance:
    - All expensive parsing is done once at initialize().
    - State checks rely on bitwise membership via fact.global_id.
    """

    def __init__(self, towers_gpt5_heuristic_6=None, name="TowersGPT5Heuristic6"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed domain knowledge
        self.goal_positions = {}          # ring -> target object from goals
        self.rings_by_size = []           # list of rings sorted from smallest to largest
        self.on_facts_by_ring = {}        # ring -> list of (Fact, target_obj)
        self.num_rings = 0

        # For hierarchical task-network lower bound
        self.min_cost_by_abstract_name = {}   # name -> minimal guaranteed cost
        self.operator_name_cost = {}          # name -> primitive operator cost (usually 1)

    # ---------------------------- Framework hooks ---------------------------- #
    def initialize(self, model: Model, initial_node: HTNNode):
        start = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start

        h0 = self._calculate_heuristic(initial_node)
        return super().initialize(model, h0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._calculate_heuristic(node)
        super().update_info(h_val)
        return h_val

    # ------------------------------- Preprocess ------------------------------ #
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            b = name.find("[")
            if b > 0 and name.endswith("]"):
                pred = name[:b]
                args_str = name[b+1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        # Fallback: (predicate arg1 arg2)
        if name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _collect_goal_positions(self):
        # Critical: goals are in bitwise form
        if hasattr(self.model, 'goals') and self.model.goals:
            for gname in self.model.state_explicit_repr(self.model.goals):
                pred, args = self._parse_fact_name(gname)
                if pred == 'on' and len(args) >= 2:
                    ring, tgt = args[0], args[1]
                    if self._is_ring(ring):
                        self.goal_positions[ring] = tgt

    def _is_ring(self, obj: str) -> bool:
        # In given instances rings are r1..rN (type RING). Use a robust check.
        # If needed, this can be enhanced by scanning smallerThan facts later.
        return obj and obj[0] == 'r'

    def _preprocess_rings_and_onfacts(self):
        rings = set()
        ring_graph = defaultdict(set)  # a -> set of b where smallerThan(a, b) and both are rings
        on_relations_goals = []        # optional fallback: (smaller ring, larger ring) from goal chain

        # Build from facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue

            if pred == 'smallerThan':
                a, b = args[0], args[1]
                if self._is_ring(a):
                    rings.add(a)
                if self._is_ring(b):
                    rings.add(b)
                if self._is_ring(a) and self._is_ring(b):
                    ring_graph[a].add(b)

            elif pred == 'on':
                r, o = args[0], args[1]
                if self._is_ring(r):
                    rings.add(r)
                    if r not in self.on_facts_by_ring:
                        self.on_facts_by_ring[r] = []
                    self.on_facts_by_ring[r].append((fact, o))

        # Optional fallback: infer 'a < b' when (on a b) in goal state
        # This helps if smallerThan ring-ring edges are missing.
        for ring, tgt in self.goal_positions.items():
            if self._is_ring(ring) and self._is_ring(tgt):
                on_relations_goals.append((ring, tgt))
                ring_graph[ring].add(tgt)
                rings.add(ring); rings.add(tgt)

        self.num_rings = len(rings)

        # Compute ordering smallest -> largest by topological sort if possible.
        # If graph has no edges (or incomplete), fall back to the counting method.
        if rings:
            indeg = {r: 0 for r in rings}
            for a in ring_graph:
                for b in ring_graph[a]:
                    indeg[b] = indeg.get(b, 0) + 1

            # Kahn's algorithm; nodes with lowest indegree are smallest
            q = deque([r for r in rings if indeg.get(r, 0) == 0])
            topo = []
            _graph_copy = {u: set(vs) for u, vs in ring_graph.items()}
            while q:
                u = q.popleft()
                topo.append(u)
                for v in list(_graph_copy.get(u, [])):
                    _graph_copy[u].remove(v)
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        q.append(v)

            # If topo includes all rings, we trust it; else fallback
            if len(topo) == len(rings):
                # 'topo' yields increasing order: u -> v means u is smaller than v
                # It might not guarantee that rings absent from edges are at correct
                # relative position, but they are isolated (no constraints), so this
                # is acceptable.
                self.rings_by_size = topo
            else:
                # Fallback: count-based ordering similar to the working example
                smaller_count = defaultdict(int)
                for a in ring_graph:
                    for _ in ring_graph[a]:
                        smaller_count[a] += 1
                self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            self.rings_by_size = []

    def _preprocess_task_min_costs(self):
        # Minimal guaranteed primitive costs per abstract task by domain structure
        # - move_abstract: always includes exactly one move => LB = 1
        # - rotateTower: decomposes to move_abstract + exchange => LB >= 1
        # - selectDirection: leads to rotateTower => LB >= 1
        # - shiftTower: leads to selectDirection => LB >= 1
        # - exchange: can be empty (exchangeClear) => LB = 0
        self.min_cost_by_abstract_name = {
            'move_abstract': 1,
            'rotateTower': 1,
            'selectDirection': 1,
            'shiftTower': 1,
            'exchange': 0,
        }
        # Primitive operator minimal cost: detected by name; in Towers domain, 'move' cost is 1
        self.operator_name_cost = {
            'move': 1,
        }

    def _preprocess(self):
        self._collect_goal_positions()
        self._preprocess_rings_and_onfacts()
        self._preprocess_task_min_costs()

    # ---------------------------- Heuristic parts ---------------------------- #
    def _get_current_position(self, state_bitwise: int, ring: str):
        """Return the object that 'ring' is currently on, using fast bitwise checks."""
        for fact, tgt in self.on_facts_by_ring.get(ring, []):
            if (state_bitwise >> fact.global_id) & 1:
                return tgt
        return None

    def _state_lower_bound(self, state_bitwise: int) -> int:
        """
        Classic Towers of Hanoi lower bound:
        - Find the largest ring that is not at its goal position.
        - Let k be the number of rings from smallest up to that ring.
        - Lower bound = 2^k - 1.
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

    def _tasknet_lower_bound(self, node: HTNNode) -> int:
        """
        Minimal guaranteed cost from the remaining task network (if accessible).
        We conservatively sum minimal costs of each pending component:
          - Primitive operator named 'move' contributes 1.
          - Abstract tasks contribute by the minima specified in _preprocess_task_min_costs.
        This is an admissible LB on its own. We combine with state LB via max().
        """
        lb = 0
        task_list = None
        # Try to fetch the node's pending task network in a robust way
        if hasattr(node, 'task_network') and node.task_network is not None:
            task_list = node.task_network
        elif hasattr(node, 'tn') and node.tn is not None:
            task_list = node.tn

        if not task_list:
            return 0

        for comp in task_list:
            # Component may be an ID or an object; normalize to object
            try:
                if isinstance(comp, int):
                    component = self.model.get_component(comp)
                else:
                    component = comp
            except Exception:
                # Be robust to unexpected forms
                component = None

            if component is None:
                continue

            # Detect by duck-typing: Operator has attributes like 'add_effects'
            # AbstractTask has 'decompositions', Decomposition has 'task_network'
            name = getattr(component, 'name', None)
            if name is None:
                continue

            # Primitive operator costs (e.g., 'move')
            if hasattr(component, 'apply') and hasattr(component, 'applicable'):
                lb += self.operator_name_cost.get(name, 0)
                continue

            # Abstract task minimal cost by name
            if hasattr(component, 'decompositions'):
                lb += self.min_cost_by_abstract_name.get(name, 0)
                continue

            # Decomposition nodes themselves don't add primitive cost directly
            # (we ignore them here)
        return lb

    def _calculate_heuristic(self, node: HTNNode) -> int:
        state_bitwise = node.state
        h_state = self._state_lower_bound(state_bitwise)
        h_tasks = self._tasknet_lower_bound(node)
        return max(h_state, h_tasks)

    # ------------------------------ Introspection ---------------------------- #
    def __repr__(self):
        return f"TowersGPT5Heuristic6(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGPT5Heuristic6 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers GPT5 Heuristic 6 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tRings (smallest->largest): {self.rings_by_size}\n"
            f"\tGoal Positions: {self.goal_positions}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
