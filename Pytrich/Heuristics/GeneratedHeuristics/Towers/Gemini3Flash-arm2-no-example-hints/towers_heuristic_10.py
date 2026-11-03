from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import collections

# Bottleneck Analysis:
# The primary bottleneck in the Towers (Tower of Hanoi) domain is the sequential movement of rings.
# Moving the largest misplaced ring requires all smaller rings to be moved to a helper peg first.
# This leads to an exponential optimal move sequence (2^n - 1) for a tower of n rings.

# Admissibility Sketch:
# Term 1: Optimal Hanoi Cost. The minimum number of moves to move a ring of size k from its 
# current peg to its target peg is 2^k, provided all smaller rings are moved out of the way.
# This recursive calculation provides the exact optimal distance for any valid 3-peg state,
# and remains a valid lower bound for any state reachable via hierarchical decomposition.

class TowersGemini3FlashHeuristic10(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_10="towers_gemini3flash_heuristic_10"):
        super().__init__(towers_gemini3flash_heuristic_10)
        self.on_fact_info = {} # fid -> (ring, obj)
        self.on_fact_ids = set()
        self.ring_to_on_fact_ids = {}
        self.smaller_than_info = {} # fid -> (small, large)
        self.smaller_than_ids = set()
        self.rings = [] # sorted by size (smallest first)
        self.towers = []
        self.goal_tower_for_ring = {}
        self.tower_triplets = {} # (t1, t2) -> other

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
        self.model = model
        
        # 1. Map facts
        first_args = set()
        all_objs = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_info[fact.global_id] = (args[0], args[1])
                self.on_fact_ids.add(fact.global_id)
                first_args.add(args[0])
                all_objs.add(args[0]); all_objs.add(args[1])
            elif pred == "smallerThan":
                self.smaller_than_info[fact.global_id] = (args[0], args[1])
                self.smaller_than_ids.add(fact.global_id)
                first_args.add(args[0])
                all_objs.add(args[0]); all_objs.add(args[1])

        # 2. Identify rings and towers
        rings = [o for o in all_objs if o in first_args]
        self.towers = [o for o in all_objs if o not in first_args]
        
        # 3. Sort rings by size (larger index = larger ring)
        larger_than_count = {r: 0 for r in rings}
        for fid in self.smaller_than_ids:
            r_small, obj_large = self.smaller_than_info[fid]
            if r_small in larger_than_count and obj_large in larger_than_count:
                larger_than_count[r_small] += 1
        self.rings = sorted(rings, key=lambda r: larger_than_count[r], reverse=True)

        # 4. Precompute grouping for fast lookup in __call__
        self.ring_to_on_fact_ids = {r: [] for r in self.rings}
        for fid in self.on_fact_ids:
            r, obj = self.on_fact_info[fid]
            if r in self.ring_to_on_fact_ids:
                self.ring_to_on_fact_ids[r].append((obj, fid))

        # 5. Precompute peg triplets (for 3-peg logic)
        if len(self.towers) >= 3:
            for t1 in self.towers:
                for t2 in self.towers:
                    if t1 != t2:
                        others = [t for t in self.towers if t != t1 and t != t2]
                        self.tower_triplets[(t1, t2)] = others[0]

        # 6. Precompute Goal Towers
        goal_parents = {}
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                goal_parents[args[0]] = args[1]
        
        start_parents = {}
        for fid in self.on_fact_ids:
            if (initial_node.state >> fid) & 1:
                r, obj = self.on_fact_info[fid]
                start_parents[r] = obj

        for r in self.rings:
            # Default to current if not in goal
            curr_search = r
            while curr_search in start_parents:
                curr_search = start_parents[curr_search]
                if curr_search in self.towers:
                    self.goal_tower_for_ring[r] = curr_search
                    break
            # Override with goal
            curr_search = r
            while curr_search in goal_parents:
                curr_search = goal_parents[curr_search]
                if curr_search in self.towers:
                    self.goal_tower_for_ring[r] = curr_search
                    break

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def compute_hanoi(self, current_towers, target_tower, ring_idx):
        if ring_idx < 0:
            return 0
        r_name = self.rings[ring_idx]
        curr = current_towers.get(r_name)
        if curr == target_tower:
            return self.compute_hanoi(current_towers, target_tower, ring_idx - 1)
        else:
            other = self.tower_triplets.get((curr, target_tower))
            if not other:
                return (1 << ring_idx) # Admissible fall-back
            # cost = H(current_tower_of_smaller, other, idx-1) + 1 + (2^idx - 1)
            return self.compute_hanoi(current_towers, other, ring_idx - 1) + (1 << ring_idx)

    def _compute(self, state, task_network):
        if not self.rings or len(self.towers) < 3:
            return len(task_network)

        # Find current towers for all rings
        current_parents = {}
        for r in self.rings:
            for obj, fid in self.ring_to_on_fact_ids[r]:
                if (state >> fid) & 1:
                    current_parents[r] = obj
                    break
        
        current_towers = {}
        for r in self.rings:
            curr_ptr = r
            limit = 0
            while curr_ptr in current_parents and limit < len(self.rings) + 1:
                curr_ptr = current_parents[curr_ptr]
                if curr_ptr in self.towers:
                    current_towers[r] = curr_ptr
                    break
                limit += 1
        
        # Hanoi heuristic starting from largest ring
        h = self.compute_hanoi(current_towers, self.goal_tower_for_ring.get(self.rings[-1], self.towers[0]), len(self.rings) - 1)
        
        # Sub-unit tie breaker for remaining HTN complexity
        return h + (0.01 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
