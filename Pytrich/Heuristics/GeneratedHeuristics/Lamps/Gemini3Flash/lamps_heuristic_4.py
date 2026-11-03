"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
The primary bottleneck in the Lamps domain is the sequential nature of lamp flips and their chain-reaction propagations. 
Each flip costs 1 unit. To reach a goal configuration, each lamp currently in the wrong status must be flipped at least 
once, either through a primary 'turn' or a secondary 'propagate' action.

Admissible Lower Bounds:
1. Mismatched Lamps: The number of cells whose current status does not match the goal status is a lower bound because 
   each mismatch requires at least one 'flip' primitive action, and every 'flip' costs 1.
2. Task Network Weight: Each non-recursive task (like 'turn' or 'propagate') in the current task network must 
   eventually decompose into at least one primitive action (either a 'flip' or a '__method_precondition' check), 
   each costing 1.

This heuristic returns max(mismatches, tn_weight), which is admissible as both components are independent 
lower bounds on the remaining cost.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic4(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_4=None, name="LampsGemini3FlashHeuristic4"):
        super().__init__(name=name)
        self.goal_fact_ids = []
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # Build lookup for fact IDs
        fact_name_to_id = {f.name: f.global_id for f in self.model.facts}
        
        # Identify goal fact IDs from the bitwise goals representation
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                # Pytrich state_explicit_repr returns grounded names like +on[l0,l0,N]
                if g_name in fact_name_to_id:
                    self.goal_fact_ids.append(fact_name_to_id[g_name])

        self.preprocessing_time = time.time() - start_time
        
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
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        # Term 1: Count of goal facts that are currently false
        # Every unsatisfied goal 'on[x,y,status]' requires at least one flip to be fixed.
        mismatches = 0
        for fact_id in self.goal_fact_ids:
            if not ((state_bitwise >> fact_id) & 1):
                mismatches += 1
        
        # Term 2: Task network cost estimation
        # Every Operator has an explicit cost. 
        # Every AbstractTask (except play) will eventually produce at least one precondition check (cost 1).
        tn_cost = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                # Operator
                tn_cost += task.cost
            else:
                # AbstractTask
                # 'play' is the recursive controller and can decompose to empty (cost 0).
                # Others (turn, propagate) must result in at least one primitive action.
                if "play" not in task.name.lower():
                    tn_cost += 1
                    
        # Both are admissible lower bounds. Max is admissible and usually tighter.
        return float(max(mismatches, tn_cost))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic4(goals={len(self.goal_fact_ids)})"

    def __str__(self):
        return f"LampsGemini3FlashHeuristic4 with {len(self.goal_fact_ids)} goals"

    def __output__(self):
        return (
            f"Lamps Gemini 3 Flash Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Goal Predicates: {len(self.goal_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
