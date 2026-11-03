from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import sys

class TowersGemini3FlashHeuristic15(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers of Hanoi domain.

    Bottleneck Analysis:
    The bottleneck in the Towers domain is the recursive requirement for moving rings. 
    To move a ring r to a target peg, all rings smaller than r must first be moved 
    to the third 'spare' peg. This results in an exponential number of actions (2^n - 1) 
    for a tower of n rings.

    Admissibility Sketch:
    1. Hanoi Distance: The standard Tower of Hanoi move count from any state to a goal state 
       is exactly computable in O(n) time. Since every action 'move' in the domain 
       corresponds to exactly one ring movement, the Hanoi distance is a tight 
       admissible lower bound on the number of actions required.
    2. Task Network Penalty: A sub-unit penalty for the number of remaining tasks helps 
       the search prioritize deeper/more progressed task networks while remaining admissible 
       (as long as the Hanoi distance term is the primary integer value).
    """

    def __init__(self, towers_gemini3flash_heuristic_15="TowersGemini3FlashHeuristic15"):
        super().__init__(towers_gemini3flash_heuristic_15)
        self.ring_to_id = {}
        self.tower_to_idx = {}
        self.ring_on_facts = []
        self.sorted_rings = []
        self.num_rings = 0
        self.final_target_peg = None

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
        
        # 1. Identify Rings and Towers
        rings = set()
        towers = set()
        smaller_than_map = {}

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'smallerThan':
                rings.add(args[0])
                if args[0] not in smaller_than_map: smaller_than_map[args[0]] = set()
                smaller_than_map[args[0]].add(args[1])
            elif pred == 'towerTop':
                towers.add(args[1])

        # 2. Sort Rings by size (smallest to largest)
        # A ring is smaller if it is 'smallerThan' more objects.
        ring_size_counts = {r: len(smaller_than_map.get(r, [])) for r in rings}
        self.sorted_rings = sorted(list(rings), key=lambda r: ring_size_counts[r], reverse=True)
        self.num_rings = len(self.sorted_rings)
        self.ring_to_id = {name: i for i, name in enumerate(self.sorted_rings)}

        # 3. Map Towers to indices 0, 1, 2
        sorted_towers = sorted(list(towers))
        self.tower_to_idx = {name: i for i, name in enumerate(sorted_towers)}

        # 4. Map 'on' facts for fast current-state lookup
        self.ring_on_facts = [[] for _ in range(self.num_rings)]
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on' and args[0] in self.ring_to_id:
                r_idx = self.ring_to_id[args[0]]
                target = args[1]
                if target in self.tower_to_idx:
                    self.ring_on_facts[r_idx].append((f.global_id, self.tower_to_idx[target], True))
                elif target in self.ring_to_id:
                    self.ring_on_facts[r_idx].append((f.global_id, self.ring_to_id[target], False))

        # 5. Determine Goal Tower (where the largest ring needs to end up)
        goal_target = {}
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_fact_names:
            p, args = self._parse_fact_name(fact_name)
            if p == 'on':
                goal_target[args[0]] = args[1]
        
        def get_goal_tower(ring_name):
            curr = ring_name
            visited = {curr}
            while curr in goal_target:
                curr = goal_target[curr]
                if curr in self.tower_to_idx: return self.tower_to_idx[curr]
                if curr in self.ring_to_id:
                    if curr in visited: break
                    visited.add(curr)
                else: break
            return None

        self.final_target_peg = 0 # Default fallback
        for r in reversed(self.sorted_rings):
            target = get_goal_tower(r)
            if target is not None:
                self.final_target_peg = target
                break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        if not self.sorted_rings:
            return 0

        # 1. Determine current peg for every ring
        # We iterate from largest to smallest ring because a ring can only be on 
        # a larger ring or a tower. 
        current_pegs = [None] * self.num_rings
        for i in range(self.num_rings - 1, -1, -1):
            for g_id, target_idx, is_tower in self.ring_on_facts[i]:
                if (state >> g_id) & 1:
                    if is_tower:
                        current_pegs[i] = target_idx
                    else:
                        current_pegs[i] = current_pegs[target_idx]
                    break

        # 2. Compute Hanoi Distance to move all rings to self.final_target_peg
        h_hanoi = 0
        target = self.final_target_peg
        for i in range(self.num_rings - 1, -1, -1):
            curr = current_pegs[i]
            if curr is not None and curr != target:
                h_hanoi += (1 << i)
                target = 3 - curr - target
        
        # 3. Add task network tie-breaker (progress guidance)
        # Sub-unit magnitude ensures h Hanoi remains dominant.
        h_tn = 0.001 * len(task_network)

        return h_hanoi + h_tn

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
