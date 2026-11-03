from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

1. Bottleneck Analysis:
   The Lamps domain is a grid-based puzzle where every cell status change requires a 'flip' action costing 1.
   Additionally, propagation of changes through methods incurs costs for method precondition checks.
   The serialization bottleneck is the required number of flips to reach the goal grid configuration.

2. Admissible Lower Bounds:
   - Term 1 (Mismatch Count): Count of goal status bits not yet satisfied in the current state.
     Admissibility: Each mismatch requires at least one 'flip' action (cost 1). Actions only change one cell at a time.
   - Term 2 (Task Network Cost): Count of primitive actions and non-trivial tasks (turn, propagate) in the current network.
     Admissibility: Every Operator costs 1. Every 'turn' or 'propagate' task decomposes into at least one flip or precondition check.

This heuristic returns max(Term 1, Term 2) with a small tie-breaker for task completion.
"""

class LampsGemini3FlashHeuristic2(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_2=None, name="LampsGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.goal_mask = 0
        self.non_trivial_tasks = set()
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # Preprocess goal requirements
        if self.model.goals:
            self.goal_mask = self.model.goals
            
        # Preprocess abstract task IDs that definitely incur cost
        for abt in self.model.abstract_tasks:
            name_lower = abt.name.lower()
            if "turn" in name_lower or "propagate" in name_lower:
                self.non_trivial_tasks.add(abt.global_id)

        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 1: Mismatch count based on bitwise goals
        # Count bits that are 1 in goals but 0 in current state
        missing_goals = (self.goal_mask & ~state)
        h_mismatch = bin(missing_goals).count('1')

        # Term 2: Task network estimated cost
        h_tn = 0
        for task in task_network:
            # If it's an operator global ID
            if self.model.iop_init <= task.global_id <= self.model.iop_end:
                h_tn += 1
            # If it's a cost-bearing complex task
            elif task.global_id in self.non_trivial_tasks:
                h_tn += 1

        # Admissible base heuristic
        lamps_gemini3flash_heuristic_2_val = float(max(h_mismatch, h_tn))

        # Tie-breaker to prefer smaller task networks (nearing completion)
        return lamps_gemini3flash_heuristic_2_val + 0.001 * len(task_network)

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

    def __output__(self):
        return (
            f"LampsGemini3FlashHeuristic2 Info:\n"
            f"\tGoal Mask: {bin(self.goal_mask)}\n"
            f"\tNon-Trivial Tasks: {len(self.non_trivial_tasks)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
