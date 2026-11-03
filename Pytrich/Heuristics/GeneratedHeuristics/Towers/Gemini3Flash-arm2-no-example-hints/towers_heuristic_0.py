from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic0(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers (Hanoi) domain.
    
    1. Bottleneck Analysis:
       The core constraint is the hierarchical movement of rings: to move a larger ring, 
       all smaller rings above it must be moved to a spare peg. This recursive 
       dependency is the serialization bottleneck.
    
    2. Admissible Lower Bounds:
       - Hanoi Distance: For the largest ring 'k' (rank 1..N) that is not in its 
         target position, at least 2^k - 1 moves are required to reach the goal. 
         This is a standard optimal lower bound for 3-peg Hanoi and is admissible 
         because every move is an action in the plan.
       - Task Network Penalty: A small tie-breaker proportional to the number of 
         tasks in the network helps guide the search through the decomposition tree.
    """

    def __init__(self, towers_gemini3flash_heuristic_0="TowersGemini3FlashHeuristic0"):
        super().__init__(towers_gemini3flash_heuristic_0)
        self.on_fact_ids = {}  # {ring_name: {obj_name: global_id}}
        self.ring_goal_obj = {} # {ring_name: target_obj_name}
        self.rank_to_ring = {}  # {rank: ring_name}
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
        # 1. Identify all rings and calculate their ranks (size)
        rings = set()
        smaller_than_pairs = []
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                rings.add(args[0])
            elif pred == "smallerThan":
                smaller_than_pairs.append((args[0], args[1]))

        self.num_rings = len(rings)
        
        # Transitive closure for robust rank calculation (in case only immediate pairs are provided)
        is_smaller = {r: set() for r in rings}
        for r_small, r_big in smaller_than_pairs:
            if r_small in rings and r_big in rings:
                is_smaller[r_small].add(r_big)
        
        # Floyd-Warshall style closure
        for k in rings:
            for i in rings:
                for j in rings:
                    if k in is_smaller[i] and j in is_smaller[k]:
                        is_smaller[i].add(j)
        
        # Rank: smallest ring (rank 1) is smaller than (N-1) other rings.
        # Rank: largest ring (rank N) is smaller than 0 other rings.
        ring_rank = {r: self.num_rings - len(is_smaller[r]) for r in rings}
        self.rank_to_ring = {rank: r for r, rank in ring_rank.items()}

        # 2. Map 'on' fact IDs for fast lookup in __call__
        for ring in rings:
            self.on_fact_ids[ring] = {}
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                ring, obj = args[0], args[1]
                if ring in self.on_fact_ids:
                    self.on_fact_ids[ring][obj] = f.global_id

        # 3. Parse goal states to determine target positions
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_fact in goal_fact_names:
            pred, args = self._parse_fact_name(g_fact)
            if pred == "on":
                self.ring_goal_obj[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute_state_h(self, state_bitwise):
        # Check rings from largest (N) to smallest (1)
        for rank in range(self.num_rings, 0, -1):
            ring = self.rank_to_ring.get(rank)
            if not ring:
                continue
            
            target_obj = self.ring_goal_obj.get(ring)
            if not target_obj:
                continue
            
            # If ring is not on its goal object, moving it (and its dependencies) costs at least 2^rank - 1
            fid = self.on_fact_ids[ring].get(target_obj)
            if fid is None or not ((state_bitwise >> fid) & 1):
                return (1 << rank) - 1
        return 0

    def _compute(self, state_bitwise, task_network):
        if state_bitwise not in self.memo:
            self.memo[state_bitwise] = self._compute_state_h(state_bitwise)
        
        # Return state-based Hanoi distance + tie-breaker for task network depth
        return self.memo[state_bitwise] + 0.001 * len(task_network)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value) if h_value > 1000000 else h_value
