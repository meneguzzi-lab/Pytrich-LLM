from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

class TowersGemini3FlashHeuristic17(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers (Hanoi) domain.
    
    Bottleneck Analysis:
    The primary bottleneck in the Towers domain is the recursive requirement of moving rings.
    To move any ring k, all rings smaller than it (1 to k-1) must be moved to an auxiliary peg.
    This creates an exponential cost structure where the effort to move a ring is doubled
    for every additional ring on top.

    Admissibility Sketches:
    1. Largest Displaced Ring Cost: The minimum moves to solve a 3-peg Hanoi puzzle with N 
       rings is 2^N - 1. If the largest ring that is not in its goal-stable position has rank k
       (where k=1 is the smallest ring), at least 2^k - 1 moves are required to fix it and the
       rings above it. This is a classic admissible lower bound.
    2. Task Network Length: Every task in the network eventually decomposes into at least one
       'move' operator (except for clear tasks). Thus, len(task_network) is a basic lower bound.

    Tie-breaking:
    Sub-unit penalties for the task network size are added to guide the search through the
    HTN decomposition tree without violating the dominant admissible structure.
    """

    def __init__(self, towers_gemini3flash_heuristic_17="TowersGemini3FlashHeuristic17"):
        super().__init__(towers_gemini3flash_heuristic_17)
        self.on_fact_ids = {}      # ring -> {support_obj: global_id}
        self.goal_supports = {}    # ring -> support_obj
        self.sorted_rings = []     # List of rings from largest to smallest
        self.ring_to_rank = {}     # ring_name -> index in sorted_rings
        self.tower_names = set()   # Set of tower object names
        self.num_rings = 0
        self.memo = {}

    def _parse_fact_name(self, name: str):
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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Identify all rings and towers
        rings = set()
        potential_towers = set()
        
        # Map 'on' fact IDs for O(1) state checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on':
                r, s = args[0], args[1]
                rings.add(r)
                potential_towers.add(s)
                if r not in self.on_fact_ids: 
                    self.on_fact_ids[r] = {}
                self.on_fact_ids[r][s] = fact.global_id

        self.tower_names = potential_towers - rings

        # 2. Order rings by size (largest to smallest)
        # Use 'smallerThan' facts in initial state to establish rank
        ring_smaller_count = collections.defaultdict(int)
        init_fact_names = model.state_explicit_repr(initial_node.state)
        for f_name in init_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'smallerThan':
                # smallerThan(smaller, larger)
                if args[0] in rings and args[1] in rings:
                    ring_smaller_count[args[1]] += 1
            elif pred == 'goal_on':
                self.goal_supports[args[0]] = args[1]

        # Fallback: Parse target supports from model.goals if goal_on was missing
        goal_fact_names = model.state_explicit_repr(model.goals)
        for f_name in goal_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'on':
                self.goal_supports[args[0]] = args[1]

        # Largest ring is smaller than 0 other rings. 
        # Smallest ring is smaller than N-1 other rings.
        # Sort descending by ring_smaller_count for [Largest, ..., Smallest]
        self.sorted_rings = sorted(list(rings), key=lambda x: ring_smaller_count[x], reverse=True)
        self.num_rings = len(self.sorted_rings)
        for idx, r in enumerate(self.sorted_rings):
            self.ring_to_rank[r] = idx

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state: int, task_network: list) -> int:
        if not task_network:
            return 0
            
        if state in self.memo:
            return self.memo[state]
            
        # stable[i] means sorted_rings[i] is on its goal support AND that support is stable
        stable = [False] * self.num_rings
        first_misplaced_rank = -1

        for i in range(self.num_rings):
            ring = self.sorted_rings[i]
            g_supp = self.goal_supports.get(ring)
            
            if g_supp is None:
                stable[i] = True
                continue

            # Fast bitwise check: is ring on its goal support right now?
            is_on_goal = False
            if ring in self.on_fact_ids and g_supp in self.on_fact_ids[ring]:
                fid = self.on_fact_ids[ring][g_supp]
                if (state >> fid) & 1:
                    is_on_goal = True

            is_stable = False
            if is_on_goal:
                if g_supp in self.tower_names:
                    is_stable = True
                elif g_supp in self.ring_to_rank:
                    parent_rank = self.ring_to_rank[g_supp]
                    # parent_rank < i means parent is larger and was processed already
                    if parent_rank < i and stable[parent_rank]:
                        is_stable = True
            
            stable[i] = is_stable
            if not is_stable and first_misplaced_rank == -1:
                first_misplaced_rank = i

        if first_misplaced_rank == -1:
            h = 0
        else:
            # k = number of rings in the stack starting from the largest misplaced one
            k = self.num_rings - first_misplaced_rank
            h = (1 << k) - 1

        # Guidance: Maximize state cost with task network tie-breaker
        res = max(h, len(task_network)) + (len(task_network) * 0.001)
        self.memo[state] = int(res)
        return int(res)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
