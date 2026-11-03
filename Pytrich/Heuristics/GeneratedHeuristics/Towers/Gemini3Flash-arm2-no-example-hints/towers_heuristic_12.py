from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

# Bottleneck Analysis:
# The bottleneck in the Towers (Tower of Hanoi) domain is the sequential dependency of rings. 
# To move a ring of size k, all rings smaller than it (1 to k-1) must be stacked on the auxiliary peg.
# This leads to the classic 2^k - 1 minimum moves requirement for the largest displaced ring.

# Admissibility Sketches:
# 1. State-based Lower Bound (h_hanoi): For the largest ring k that is not on its goal peg, 
#    at least 2^k - 1 primitive move operations are required to move it and its sub-tower to the goal.
#    This is the standard optimal solution for the 3-peg Tower of Hanoi.
# 2. Task-based Lower Bound (h_tasks): Each 'move_abstract' task in the HTN network represents 
#    at least one primitive move operation. Max(h_hanoi, num_tasks) is admissible.

class TowersGemini3FlashHeuristic12(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_12="TowersGemini3FlashHeuristic12"):
        super().__init__(towers_gemini3flash_heuristic_12)
        self.rings = set()
        self.towers = set()
        self.ring_sizes = {} # ring -> int (1 is smallest)
        self.rings_sorted_desc = [] # List of (ring, size) largest to smallest
        self.goal_pegs = {} # ring -> tower_name
        self.on_fact_info = [] # List of (global_id, ring_name, target_name)
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
        # 1. Identify Rings and Towers
        smaller_than_pairs = []
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'towerTop':
                self.towers.add(args[1])
            elif pred == 'smallerThan':
                # smallerThan(ring, obj)
                self.rings.add(args[0])
                smaller_than_pairs.append((args[0], args[1]))
            elif pred == 'on':
                self.on_fact_info.append((f.global_id, args[0], args[1]))

        # 2. Compute Ring Sizes (rank by how many rings are smaller than it)
        for r in self.rings:
            # count how many other rings are smaller than this one
            rank = 0
            for other in self.rings:
                if (other, r) in smaller_than_pairs:
                    rank += 1
            self.ring_sizes[r] = rank + 1
        
        self.rings_sorted_desc = sorted(self.ring_sizes.items(), key=lambda x: x[1], reverse=True)

        # 3. Precompute Goal Pegs
        goal_on_map = {}
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on':
                goal_on_map[args[0]] = args[1]
        
        # Trace pegs from largest to smallest for goals
        for r, size in self.rings_sorted_desc:
            target = goal_on_map.get(r)
            if target in self.towers:
                self.goal_pegs[r] = target
            elif target in self.goal_pegs:
                self.goal_pegs[r] = self.goal_pegs[target]

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state: int, task_network) -> int:
        # Memoization by state and task network length
        memo_key = (state, len(task_network))
        if memo_key in self.memo:
            return self.memo[memo_key]

        # 1. Extract current 'on' relations efficiently
        current_on_map = {}
        for fid, r_name, target_name in self.on_fact_info:
            if (state >> fid) & 1:
                current_on_map[r_name] = target_name

        # 2. Determine current peg for each ring (O(n) trace largest-to-smallest)
        current_pegs = {}
        for r, size in self.rings_sorted_desc:
            target = current_on_map.get(r)
            if target in self.towers:
                current_pegs[r] = target
            elif target in current_pegs:
                current_pegs[r] = current_pegs[target]

        # 3. Find largest displaced ring k
        max_k = 0
        for r, size in self.rings_sorted_desc:
            curr_peg = current_pegs.get(r)
            goal_peg = self.goal_pegs.get(r)
            if goal_peg and curr_peg != goal_peg:
                max_k = size
                break

        # Hanoi cost for size k is 2^k - 1
        h_hanoi = (1 << max_k) - 1 if max_k > 0 else 0
        
        # Task-based term: every move_abstract is at least one step
        # We sum them as they must eventually occur.
        move_tasks = 0
        for t in task_network:
            if hasattr(t, 'name') and 'move_abstract' in t.name:
                move_tasks += 1
        
        # Final h is max of independent bounds + tiny tie-breaker for search guidance
        # len(task_network) helps break ties in decomposition paths
        res = max(h_hanoi, move_tasks)
        final_h = int(res)
        
        # Add tiny tie-breaker (sub-unit to preserve admissibility if res is optimal)
        tie_breaker = 0.001 * len(task_network)
        
        self.memo[memo_key] = final_h
        return final_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
