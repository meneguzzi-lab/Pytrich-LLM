from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class TowersGemini3FlashHeuristic6(Heuristic):
    """
    Towers of Hanoi Domain Heuristic
    Bottleneck: The stack property and sequential nature of ring movement. To move a large ring,
    all smaller rings above it must be moved out of the way. Rings must be moved one at a time.

    Admissible Bounds:
    1. Misplaced Rings: A ring is misplaced if it is not on its goal target, or if the target it sits on is misplaced.
       Each misplaced ring requires at least one 'move' action. Since each ring move is a unique action and one move
       cannot solve two 'misplaced' rings simultaneously (except by moving them sequentially), this is a lower bound.
    2. Task Network Size: Every task in the HTN network must eventually be decomposed into at least one action
       (or terminates in a base case). The number of remaining tasks provides a lower bound on steps remaining.
    """
    def __init__(self, towers_gemini3flash_heuristic_6="towers_gemini3flash_heuristic_6"):
        super().__init__(name=towers_gemini3flash_heuristic_6)
        self.rings = set()
        self.on_fact_ids = {} # ring -> {target_name -> fact_id}
        self.goal_on_target_ids = {} # ring -> fact_id of desired 'on' state
        self.goal_ring_target_names = {} # ring -> name of object it should be on

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
        # Precompute fact mappings
        for fid in range(self.model.ifacts_init, self.model.ifacts_end + 1):
            f = self.model.get_component(fid)
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                ring, target = args[0], args[1]
                self.rings.add(ring)
                if ring not in self.on_fact_ids:
                    self.on_fact_ids[ring] = {}
                self.on_fact_ids[ring][target] = fid

        # Precompute goal information
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                ring, target = args[0], args[1]
                if ring in self.on_fact_ids and target in self.on_fact_ids[ring]:
                    self.goal_on_target_ids[ring] = self.on_fact_ids[ring][target]
                    self.goal_ring_target_names[ring] = target

        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

    def _is_ring_misplaced(self, state, ring, memo):
        if ring in memo:
            return memo[ring]
        
        goal_fid = self.goal_on_target_ids.get(ring)
        if goal_fid is None:
            memo[ring] = False
            return False

        # 1. Is it currently on its goal target?
        is_on_goal = (state >> goal_fid) & 1
        if not is_on_goal:
            memo[ring] = True
            return True

        # 2. It is on the correct target. But if that target is a ring, is the target itself misplaced?
        target_name = self.goal_ring_target_names.get(ring)
        if target_name in self.rings:
            if self._is_ring_misplaced(state, target_name, memo):
                memo[ring] = True
                return True

        memo[ring] = False
        return False

    def _compute(self, state, task_network):
        # Term 1: Misplaced rings count
        misplaced_count = 0
        memo = {}
        for ring in self.rings:
            if self._is_ring_misplaced(state, ring, memo):
                misplaced_count += 1
        
        # Term 2: Task network complexity
        # Every task remaining must be processed. 
        # Abstract tasks decompose into moves. Operators are moves themselves.
        task_count = len(task_network)

        # Both are admissible; use max to combine safely.
        return max(misplaced_count, task_count)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
