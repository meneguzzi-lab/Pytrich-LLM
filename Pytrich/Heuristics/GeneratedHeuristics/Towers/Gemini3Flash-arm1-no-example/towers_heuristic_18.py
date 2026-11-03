from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic18(Heuristic):
    """
    Bottleneck Analysis: The Tower of Hanoi problem is the central bottleneck. 
    The state space grows exponentially with the number of rings (2^n states to find the optimal path).
    Any valid plan must move each ring r that is not currently part of the correct 
    stack at its goal destination, and moving such a ring requires recursively clearing 
    all rings smaller than it onto an auxiliary peg.

    Admissible Lower Bounds:
    1. Misplaced Hanoi Count (h_hanoi): The standard Hanoi heuristic. For the largest misplaced ring k,
       the minimum number of moves to place it on its target peg is 2^k - 1. We refine this by identifying
       the correct auxiliary peg destination for each smaller ring during the calculation.
       This is a well-known admissible lower bound for the number of 'move' actions.
    2. Task Network Complexity: Every non-trivial task (AbstractTask or Operator) remaining in 
       the task network represents at least one future action (move). 
       Cost = len(task_network).

    The final heuristic is max(h_hanoi, h_tasks).
    """

    def __init__(self, towers_gemini3flash_heuristic_18="towers_gemini3flash_heuristic_18"):
        super().__init__(towers_gemini3flash_heuristic_18)
        self.on_goal = {}
        self.ring_rank = {}
        self.rank_to_ring = {}
        self.on_fact_list = {}
        self.all_rings = set()
        self.all_towers = []
        self.ring_goal_tower = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse all facts to identify rings and towers
        smaller_than_rels = []
        tower_candidates = set()
        
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "smallerThan":
                self.all_rings.add(args[0])
                smaller_than_rels.append((args[0], args[1]))
            elif pred == "towerTop" and args[0] == args[1]:
                tower_candidates.add(args[0])

        self.all_towers = list(tower_candidates)

        # 2. Rank rings by size: Rank 1 (smallest) to Rank N (largest)
        # A ring is smaller if it appears as the first argument in more 'smallerThan' relations with other rings.
        ring_larger_counts = {r: 0 for r in self.all_rings}
        for r_small, r_large in smaller_than_rels:
            if r_large in self.all_rings:
                ring_larger_counts[r_small] += 1
        
        sorted_rings = sorted(self.all_rings, key=lambda r: ring_larger_counts[r], reverse=True)
        for i, r in enumerate(sorted_rings):
            rank = i + 1
            self.ring_rank[r] = rank
            self.rank_to_ring[rank] = r

        # 3. Parse goal state for 'on' facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                ring, target = args[0], args[1]
                if ring in self.all_rings:
                    self.on_goal[ring] = target

        # Precompute static goal towers
        for ring in self.all_rings:
            curr = ring
            visited = set()
            while curr in self.all_rings and curr not in visited:
                visited.add(curr)
                curr = self.on_goal.get(curr, curr)
            self.ring_goal_tower[ring] = curr

        # 4. Map 'on' facts for fast lookup in __call__
        self.on_fact_list = {ring: [] for ring in self.all_rings}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and args[0] in self.all_rings:
                self.on_fact_list[args[0]].append((args[1], f.global_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network: list) -> int:
        if not self.all_rings:
            return len(task_network)

        # Refined Hanoi Heuristic
        h_hanoi = 0
        n_rings = len(self.all_rings)
        # Start with the goal tower for the largest ring
        current_target_peg = self.ring_goal_tower.get(self.rank_to_ring[n_rings])
        
        for i in range(n_rings, 0, -1):
            ring = self.rank_to_ring[i]
            curr_peg = self._get_current_tower(ring, state)
            
            if curr_peg != current_target_peg:
                # Admissible cost to move ring i and all its dependencies
                h_hanoi += (1 << (i - 1))
                # Smaller rings must clear to the auxiliary peg
                current_target_peg = self._get_aux_peg(curr_peg, current_target_peg)
            # else: current_target_peg remains the same for the next smaller ring

        h_tasks = len(task_network)
        # We return the h_hanoi value, ensuring it's at least the task count if hanoi is solved
        return int(max(h_hanoi, h_tasks))

    def _get_current_tower(self, ring: str, state: int) -> str:
        curr = ring
        # Follow 'on' chain to tower. Chain length is at most N.
        for _ in range(len(self.all_rings) + 1):
            if curr not in self.all_rings:
                return curr
            found = False
            for obj, fid in self.on_fact_list.get(curr, []):
                if (state >> fid) & 1:
                    curr = obj
                    found = True
                    break
            if not found:
                return curr # Should not happen in consistent states
        return curr

    def _get_aux_peg(self, p1: str, p2: str) -> str:
        for t in self.all_towers:
            if t != p1 and t != p2:
                return t
        return self.all_towers[0] if self.all_towers else p1

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
