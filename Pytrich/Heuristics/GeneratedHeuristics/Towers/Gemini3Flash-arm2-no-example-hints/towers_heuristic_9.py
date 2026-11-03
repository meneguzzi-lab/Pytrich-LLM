from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import sys

class TowersGemini3FlashHeuristic9(Heuristic):
    """
    Bottleneck Analysis: The Tower of Hanoi domain is constrained by the sequential dependency of ring moves. 
    The largest misplaced ring dictates the minimum remaining cost, following the exponential 2^k - 1 complexity. 

    Admissibility Sketch:
    The heuristic uses the recursive state-to-state Hanoi distance. This distance represents the exact 
    minimum number of moves required to move a stack of i rings from their current configuration to a target tower 
    in a 3-peg configuration. Since any valid plan must at least perform these moves, it is admissible. 
    """

    def __init__(self, towers_gemini3flash_heuristic_9="TowersGemini3FlashHeuristic9"):
        super().__init__(towers_gemini3flash_heuristic_9)
        self.memo = {}
        self.rings_ordered = []
        self.tower_names = []
        self.on_facts_by_ring = {}
        self.common_goal_tower = None

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Parse facts to identify objects, rings, and towers
        all_objs = set()
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p: 
                for a in args: all_objs.add(a)

        rings = set()
        smaller_than_rels = []
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p == "smallerThan":
                rings.add(args[0])
                smaller_than_rels.append(args)

        towers = all_objs - rings
        self.tower_names = sorted(list(towers))

        # 2. Order rings by size (smallest has index 0)
        ring_sizes = {r: 0 for r in rings}
        for r_smaller, r_larger in smaller_than_rels:
            if r_smaller in rings and r_larger in rings:
                ring_sizes[r_smaller] += 1
        self.rings_ordered = sorted(list(rings), key=lambda r: ring_sizes[r], reverse=True)

        # 3. Index 'on' facts for fast state checking
        self.on_facts_by_ring = {r: {} for r in rings}
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p == "on":
                r, obj = args[0], args[1]
                if r in self.on_facts_by_ring:
                    self.on_facts_by_ring[r][obj] = f.global_id

        # 4. Determine common goal tower
        goal_fact_names = model.state_explicit_repr(model.goals)
        goal_on = {}
        for fact_name in goal_fact_names:
            p, args = self._parse_fact_name(fact_name)
            if p == "on":
                goal_on[args[0]] = args[1]
        
        for r in reversed(self.rings_ordered):
            if r in goal_on:
                curr = goal_on[r]
                while curr in goal_on:
                    curr = goal_on[curr]
                self.common_goal_tower = curr
                break

        h_initial = self._compute(initial_node.state)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        state_bitwise = node.state
        if state_bitwise in self.memo:
            h = self.memo[state_bitwise]
        else:
            h = self._compute(state_bitwise)
            self.memo[state_bitwise] = h
        super().update_info(h)
        return h

    def _compute(self, state_bitwise: int) -> int:
        if self.common_goal_tower is None:
            return 0

        # Find current tower for every ring (O(N) using tracing)
        curr_on = {}
        for r, facts in self.on_facts_by_ring.items():
            for obj, f_id in facts.items():
                if (state_bitwise >> f_id) & 1:
                    curr_on[r] = obj
                    break

        current_towers = {}
        # Trace from largest to smallest to resolve chains efficiently
        for r in reversed(self.rings_ordered):
            target = curr_on.get(r)
            if target in self.tower_names:
                current_towers[r] = target
            elif target in current_towers:
                current_towers[r] = current_towers[target]
            else:
                current_towers[r] = None

        return self._hanoi_dist(len(self.rings_ordered) - 1, self.common_goal_tower, current_towers)

    def _hanoi_dist(self, ring_idx: int, target_tower: str, current_towers: dict) -> int:
        if ring_idx < 0:
            return 0
        ring = self.rings_ordered[ring_idx]
        curr_tower = current_towers.get(ring)

        if curr_tower == target_tower or curr_tower is None:
            return self._hanoi_dist(ring_idx - 1, target_tower, current_towers)
        else:
            # Find aux tower (Hanoi has 3 pegs)
            aux_tower = None
            for t in self.tower_names:
                if t != target_tower and t != curr_tower:
                    aux_tower = t
                    break
            # Cost = moves to move smaller stack to aux + move ring_idx to target + moves to move smaller back
            # This recursive step captures 2^(ring_idx) moves in total state-to-state logic
            return (1 << ring_idx) + self._hanoi_dist(ring_idx - 1, aux_tower, current_towers)

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
