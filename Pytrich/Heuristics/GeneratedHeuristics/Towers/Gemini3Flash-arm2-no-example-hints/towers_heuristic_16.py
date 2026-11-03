from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic16(Heuristic):
    """
    Bottleneck Analysis: The Tower of Hanoi domain is fundamentally constrained by the 
    recursive requirement to clear all smaller rings off a larger ring and onto an 
    auxiliary tower before the larger ring can be moved. This results in the 
    exponential 2^n cost structure.

    Admissible Lower Bounds:
    1. Hanoi Distance: The optimal number of moves to move N disks from their current 
       peg to their goal peg (Hanoi distance) is a known quantity and is admissible.
    2. Task Network Length: Every task in the network eventually results in at least 
       one action or is a placeholder for future work.

    Design: We precompute the ring sizes and goal positions. In each call, we trace 
    the current peg of each ring and calculate the Hanoi distance using the 
    optimal bitwise recursive formula for 3 pegs.
    """

    def __init__(self, towers_gemini3flash_heuristic_16="towers_heuristic_16"):
        super().__init__(towers_gemini3flash_heuristic_16)
        self.on_facts_by_ring = {}
        self.smaller_than_facts = {}
        self.tower_name_to_idx = {}
        self.ordered_rings = []
        self.goal_pegs = {}
        self.num_rings = 0
        self.hanoi_memo = {}

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
        rings_set = set()
        towers_set = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "smallerThan":
                rings_set.add(args[0])
                self.smaller_than_facts[(args[0], args[1])] = f.global_id
            if pred == "towerTop":
                towers_set.add(args[1])
            if pred == "on":
                if args[0] not in self.on_facts_by_ring: self.on_facts_by_ring[args[0]] = {}
                self.on_facts_by_ring[args[0]][args[1]] = f.global_id

        self.towers = sorted(list(towers_set))
        self.tower_name_to_idx = {name: i for i, name in enumerate(self.towers)}

        # 2. Sort Rings by size (largest to smallest)
        ring_counts = {}
        for r in rings_set:
            count = 0
            for obj in list(rings_set) + self.towers:
                fid = self.smaller_than_facts.get((r, obj))
                if fid is not None and (initial_node.state >> fid) & 1:
                    count += 1
            ring_counts[r] = count
        # More objects a ring is smallerThan -> ring is smaller. Sort counts ASC for largest first.
        self.ordered_rings = sorted(list(rings_set), key=lambda r: ring_counts[r])
        self.num_rings = len(self.ordered_rings)

        # 3. Precompute Goal Pegs
        goal_facts = model.state_explicit_repr(model.goals)
        goal_on_map = {}
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                goal_on_map[args[0]] = args[1]
        
        for ring in self.ordered_rings:
            curr = ring
            visited = set()
            while curr in rings_set:
                visited.add(curr)
                curr = goal_on_map.get(curr)
                if curr is None or curr in visited: break
            self.goal_pegs[ring] = self.tower_name_to_idx.get(curr, 0)

        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

    def _get_current_peg(self, ring, state):
        curr = ring
        for _ in range(self.num_rings + 1):
            if curr in self.tower_name_to_idx:
                return self.tower_name_to_idx[curr]
            found = False
            if curr in self.on_facts_by_ring:
                for obj_name, fid in self.on_facts_by_ring[curr].items():
                    if (state >> fid) & 1:
                        curr = obj_name
                        found = True
                        break
            if not found: break
        return -1

    def _compute_hanoi(self, state):
        if state in self.hanoi_memo: return self.hanoi_memo[state]
        
        current_pegs = [self._get_current_peg(r, state) for r in self.ordered_rings]
        temp_targets = [self.goal_pegs[r] for r in self.ordered_rings]
        
        cost = 0
        for i in range(self.num_rings): # largest to smallest
            rank = self.num_rings - 1 - i
            if current_pegs[i] != temp_targets[i] and current_pegs[i] != -1:
                cost += (1 << rank)
                # Update targets for all smaller disks
                aux_peg = 3 - current_pegs[i] - temp_targets[i]
                for j in range(i + 1, self.num_rings):
                    temp_targets[j] = aux_peg
        
        self.hanoi_memo[state] = cost
        return cost

    def _compute(self, state, task_network):
        h_hanoi = self._compute_hanoi(state)
        # h_hanoi is the exact move count. Add small tie-breaker for task network length.
        return h_hanoi + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
