from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The Tower of Hanoi domain is governed by the recursive dependency of ring placement.
# To move any ring 'k', all rings smaller than 'k' must first be moved to an auxiliary peg.
# This creates an exponential lower bound (2^k - 1) based on the largest misplaced ring.

# Admissibility Sketches:
# 1. Hanoi Distance: The standard recursive state-to-goal move count for the 3-peg Hanoi problem.
#    Since the move action is the only way to change the state, and standard Hanoi constraints apply,
#    the 3-peg optimal cost (2^n - 1) is a valid lower bound for any problem with >= 3 pegs.
# 2. Task Network progress: A small penalty proportional to the number of tasks remaining in the network
#    helps distinguish between nodes at the same state but different levels of decomposition progress.

class TowersGemini3FlashHeuristic2(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_2="towers_gemini3flash_heuristic_2"):
        super().__init__(towers_gemini3flash_heuristic_2)
        self.num_rings = 0
        self.ring_to_size = {} # name -> 1..N
        self.size_to_ring = {} # 1..N -> name
        self.tower_to_idx = {} # name -> 0,1,2
        self.ring_on_candidates = {} # ring -> [(target_name, fact_id), ...]
        self.goal_pegs = [] # size-indexed list of target peg indices
        self.all_peg_indices = set()

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
        # 1. Identify Rings and Towers
        towers = set()
        rings = set()
        smaller_than_rels = []

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "towerTop":
                towers.add(args[1])
            elif pred == "smallerThan":
                rings.add(args[0])
                smaller_than_rels.append((args[0], args[1]))

        # 2. Determine Ring Sizes (Smallest = 1, Largest = N)
        # A ring's size is defined by how many rings are smaller than it.
        num_rings = len(rings)
        self.num_rings = num_rings
        for r in rings:
            # count x such that smallerThan(x, r)
            count = sum(1 for sub, obj in smaller_than_rels if obj == r and sub in rings)
            size = count + 1
            self.ring_to_size[r] = size
            self.size_to_ring[size] = r

        # 3. Map Towers to Indices
        sorted_towers = sorted(list(towers))
        self.tower_to_idx = {t: i for i, t in enumerate(sorted_towers)}
        self.all_peg_indices = set(range(len(sorted_towers)))

        # 4. Pre-process 'on' candidates for fast bitwise checking
        self.ring_on_candidates = {r: [] for r in rings}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and args[0] in rings:
                self.ring_on_candidates[args[0]].append((args[1], f.global_id))

        # 5. Determine Goal Pegs from Goal State
        goal_on_map = {}
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_fact_names:
            pred, args = self._parse_fact_name(fact_name)
            if pred == "on" and len(args) == 2:
                goal_on_map[args[0]] = args[1]

        self.goal_pegs = [0] * (num_rings + 1)
        for i in range(1, num_rings + 1):
            ring = self.size_to_ring[i]
            curr = ring
            path = {curr}
            while curr in goal_on_map:
                curr = goal_on_map[curr]
                if curr in self.tower_to_idx:
                    self.goal_pegs[i] = self.tower_to_idx[curr]
                    break
                if curr in path: break # Cycle
                path.add(curr)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        if self.num_rings == 0:
            return 0

        # 1. Find Current Peg for each ring (Largest to Smallest to avoid O(N^2))
        current_pegs = [0] * (self.num_rings + 1)
        for i in range(self.num_rings, 0, -1):
            ring = self.size_to_ring[i]
            for target, fact_id in self.ring_on_candidates[ring]:
                if (state >> fact_id) & 1:
                    if target in self.tower_to_idx:
                        current_pegs[i] = self.tower_to_idx[target]
                    elif target in self.ring_to_size:
                        # Larger rings are already processed, smallerThan constraint holds
                        target_size = self.ring_to_size[target]
                        current_pegs[i] = current_pegs[target_size]
                    break

        # 2. Hanoi Distance calculation (admissible lower bound)
        h_dist = 0
        targets = list(self.goal_pegs)
        for i in range(self.num_rings, 0, -1):
            if current_pegs[i] != targets[i]:
                h_dist += (1 << (i - 1))
                # Auxiliary peg is any peg not current and not target
                aux_options = self.all_peg_indices - {current_pegs[i], targets[i]}
                aux = min(aux_options) if aux_options else current_pegs[i]
                for j in range(1, i):
                    targets[j] = aux

        # 3. Tie-breaking based on Task Network length
        # Preserves admissibility if h_dist is accurate in units of 'move' actions
        h_tn = 0.001 * len(task_network)

        return int(h_dist + h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
