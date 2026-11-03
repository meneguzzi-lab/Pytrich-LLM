from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

# Bottleneck: The Towers of Hanoi constraint (larger rings cannot be placed on smaller ones) 
# creates a recursive dependency where moving a single large ring requires an exponential 
# number of moves (2^n - 1) for the stack above it.
# 
# Admissible Lower Bounds:
# 1. Hanoi Distance: The exact optimal number of moves to reach a goal configuration from a given 
#    state in a 3-peg Tower of Hanoi is computed using the standard recursive definition. 
#    This is admissible because it represents the shortest possible physical path for the operators.
# 2. Task Network Tie-breaker: A sub-unit penalty for the number of tasks in the network helps 
#    guide the search toward nodes closer to total decomposition without violating admissibility.

class TowersGemini3FlashHeuristic18(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_18="TowersGemini3FlashHeuristic18"):
        super().__init__(towers_gemini3flash_heuristic_18)
        self.rings = []
        self.towers = []
        self.ring_on_fact_ids = {} # ring -> {obj_on: global_id}
        self.goal_pegs = {} # ring -> tower_name
        self.other_peg_map = {} # (peg1, peg2) -> peg3

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

    def _get_num(self, s):
        nums = re.findall(r'\d+', s)
        return int(nums[0]) if nums else 0

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Classify objects and collect fact IDs
        temp_rings = set()
        temp_towers = set()
        
        # Parse all facts once
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                temp_rings.add(args[0])
                if args[0] not in self.ring_on_fact_ids:
                    self.ring_on_fact_ids[args[0]] = {}
                self.ring_on_fact_ids[args[0]][args[1]] = f.global_id
            elif pred == 'towerTop':
                temp_towers.add(args[1])

        self.rings = sorted(list(temp_rings), key=self._get_num)
        self.towers = sorted(list(temp_towers))

        # 2. Precompute the 'other' peg for 3-peg Hanoi logic
        if len(self.towers) >= 3:
            for p1 in self.towers:
                for p2 in self.towers:
                    if p1 != p2:
                        for p3 in self.towers:
                            if p3 != p1 and p3 != p2:
                                self.other_peg_map[(p1, p2)] = p3
                                break

        # 3. Identify goal peg for each ring
        goal_on_map = {}
        goal_fact_names = model.state_explicit_repr(model.goals)
        for f_name in goal_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'on':
                goal_on_map[args[0]] = args[1]

        # Trace goal pegs from largest to smallest
        for r in reversed(self.rings):
            target = goal_on_map.get(r)
            if target:
                if target in self.towers:
                    self.goal_pegs[r] = target
                elif target in self.goal_pegs:
                    self.goal_pegs[r] = self.goal_pegs[target]
        
        # Default goal for largest ring if unspecified (usually onto t3 in these pfiles)
        if self.rings and self.rings[-1] not in self.goal_pegs:
            self.goal_pegs[self.rings[-1]] = self.towers[-1] if self.towers else None

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _h_hanoi(self, ring_idx, target_peg, current_pegs):
        if ring_idx < 0:
            return 0
        ring = self.rings[ring_idx]
        curr = current_pegs.get(ring)
        if curr == target_peg:
            return self._h_hanoi(ring_idx - 1, target_peg, current_pegs)
        else:
            other = self.other_peg_map.get((curr, target_peg), self.towers[0] if self.towers else None)
            # Hanoi cost: 2^i moves to shift stack of size i
            return (1 << ring_idx) + self._h_hanoi(ring_idx - 1, other, current_pegs)

    def _compute(self, state, task_network):
        if not self.rings:
            return 0
            
        # 1. Determine current peg for each ring in O(N) by tracing largest to smallest
        current_pegs = {}
        for ring in reversed(self.rings):
            facts = self.ring_on_fact_ids.get(ring, {})
            for obj, fid in facts.items():
                if (state >> fid) & 1:
                    if obj in self.towers:
                        current_pegs[ring] = obj
                    else:
                        current_pegs[ring] = current_pegs.get(obj)
                    break
        
        # 2. Compute Hanoi cost from largest displaced ring
        # All rings should ideally end up on their goal peg defined by the base
        base_ring = self.rings[-1]
        target = self.goal_pegs.get(base_ring, self.towers[-1] if self.towers else None)
        
        h_hanoi = self._h_hanoi(len(self.rings) - 1, target, current_pegs)
        
        # 3. Add tie-breaker for task network progress
        h_total = h_hanoi + (0.001 * len(task_network))
        return int(h_total) if h_total > 1000000000 else h_total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
