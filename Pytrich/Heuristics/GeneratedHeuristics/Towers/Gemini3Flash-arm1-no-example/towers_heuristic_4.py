from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The Towers (Tower of Hanoi) domain bottleneck is the strict stack-based serialization of tasks. 
# To move any ring, the entire stack above it must be cleared, and it can only be placed on an 
# object (tower or larger ring) that is currently clear and valid. This creates a recursive 
# dependency where the largest misplaced ring dictates the minimum amount of work remaining.

# Admissibility Sketches:
# 1. Unstable Ring Count: A ring is 'stable' only if it is on its goal-destination and that destination 
#    is also stable (a tower is always stable). Any ring that is not stable MUST be moved at least once. 
#    Summing these misplaced rings provides a strict lower bound on move actions.
# 2. Task Network Depth: Each task in the network (shiftTower, rotateTower, move_abstract) represents 
#    at least one future move or decomposition step. In a serial stack domain, these tasks 
#    represent independent units of work.

class TowersGemini3FlashHeuristic4(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_4="towers_gemini3flash_heuristic_4"):
        super().__init__(towers_gemini3flash_heuristic_4)
        self.on_ids = {} # (ring, obj) -> global_id
        self.goal_on = {} # ring -> target_obj
        self.towers = set()
        self.rings = set()
        self.goal_stack = [] # Bottom-to-top ring sequence

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
        on_facts = []
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                self.rings.add(args[0])
                on_facts.append(args)
        
        all_targets = set(a[1] for a in on_facts)
        self.towers = all_targets - self.rings

        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g in goal_names:
            p, args = self._parse_fact_name(g)
            if p == "on":
                self.goal_on[args[0]] = args[1]

        remaining = set(self.goal_on.keys())
        # Build stack from bottom (on tower) to top (on ring)
        current_layer = [r for r in remaining if self.goal_on[r] in self.towers]
        for r in current_layer:
            self.goal_stack.append(r)
            remaining.remove(r)
        
        while remaining:
            added = False
            for r in list(remaining):
                target = self.goal_on[r]
                if target in self.goal_stack:
                    idx = self.goal_stack.index(target)
                    self.goal_stack.insert(idx + 1, r)
                    remaining.remove(r)
                    added = True
                    break
            if not added: break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network) -> int:
        # Term 1: Stability count (Admissible)
        # A ring is stable if it's on its goal target and that target is stable.
        stable_objs = set(self.towers)
        unstable_count = 0
        for r in self.goal_stack:
            target = self.goal_on.get(r)
            fid = self.on_ids.get((r, target))
            if fid is not None and (state >> fid) & 1 and target in stable_objs:
                stable_objs.add(r)
            else:
                unstable_count += 1
        
        # Term 2: Task network tie-breaker
        # Favor states with fewer remaining high-level tasks
        h_tasks = len(task_network)

        return unstable_count + 0.001 * h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
