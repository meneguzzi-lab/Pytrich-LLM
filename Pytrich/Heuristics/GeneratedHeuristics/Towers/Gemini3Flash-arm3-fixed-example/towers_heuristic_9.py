import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Towers Domain Bottleneck Analysis:
# The bottleneck in the Towers of Hanoi is the sequential dependency of ring movements. 
# Larger rings must be positioned correctly before smaller rings can be permanently placed on top of them.
# Conversely, all rings currently on top of a ring, and all rings at its target destination, 
# must be moved to an auxiliary peg before that ring can be shifted.

# Admissible Lower Bounds:
# 1. Misplaced Goal Rings: Every ring not in its correct goal position must move at least once. 
# 2. Sequential Goal Stability: A ring is only "well-placed" if it is on its goal target AND its target is also well-placed (or is the correct tower peg).
#    The number of rings that are not "well-placed" is an admissible lower bound on the number of moves needed.
# 3. Task Network: Every 'move' or 'move_abstract' task currently in the task network represents at least one future action.

class TowersGemini3FlashHeuristic9(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_9=0.001, name="towers_gemini3flash_heuristic_9"):
        super().__init__(name=name)
        self.towers_gemini3flash_heuristic_9 = towers_gemini3flash_heuristic_9
        self.goal_on_map = {}
        self.on_ids = {}
        self.goal_chain = []
        self.all_rings = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse all 'on' facts to build a map of IDs
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                r, o = args[0], args[1]
                if r not in self.on_ids: self.on_ids[r] = {}
                self.on_ids[r][o] = f.global_id

        # 2. Parse goal state 'on' facts
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on':
                self.goal_on_map[args[0]] = args[1]
        
        # 3. Identify all rings involved in the goal and construct the goal chain
        self.all_rings = set(self.goal_on_map.keys())
        
        # Find the bottom ring (the one whose target is not another ring)
        bottom_ring = None
        for ring, target in self.goal_on_map.items():
            if target not in self.all_rings:
                bottom_ring = ring
                break
        
        # Build goal chain from bottom to top
        current = bottom_ring
        visited = set()
        while current and current not in visited:
            self.goal_chain.append(current)
            visited.add(current)
            # Find the ring that sits on 'current' in the goal state
            next_ring = None
            for r, t in self.goal_on_map.items():
                if t == current:
                    next_ring = r
                    break
            current = next_ring

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    def _compute(self, state_bitwise, task_network):
        # Count how many rings are well-placed starting from the bottom of the goal tower
        well_placed_count = 0
        for r in self.goal_chain:
            target = self.goal_on_map.get(r)
            oid = self.on_ids.get(r, {}).get(target)
            if oid is not None and (state_bitwise >> oid) & 1:
                well_placed_count += 1
            else:
                # If this ring is not correctly placed, everything above it is also not well-placed
                break
        
        # Primary term: number of rings not yet stable in goal position
        h = float(len(self.all_rings) - well_placed_count)
        
        # Secondary term (Tie-breaker): length of the task network to guide HTN decomposition
        if task_network:
            h += len(task_network) * self.towers_gemini3flash_heuristic_9
            
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
