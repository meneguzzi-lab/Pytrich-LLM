"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
In the Lamps domain, reaching a specific configuration of lamp statuses is the bottleneck. 
Every status change (from 'Y' to 'N' or vice-versa) requires at least one 'flip' primitive action. 
Since the Pytrich cost model charges 1 for each primitive action (whether user-initiated 
or propagated), the total number of 'flip' actions executed is the total plan cost.

Admissible Lower Bound:
1. Goal Hamming Distance: For each coordinate (x, y) where the goal specifies a status, 
   if the current status does not match the goal, at least one 'flip' must be applied 
   to that lamp. Since each 'flip' only affects one lamp and costs 1, the number of 
   goal lamps currently in the wrong status is an admissible lower bound on the remaining cost.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic19(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_19=None, name="LampsGemini3FlashHeuristic19"):
        super().__init__(name=name)
        self.goal_on_fact_ids = []
        self.preprocessing_time = 0
        self.lamps_gemini3flash_heuristic_19 = lamps_gemini3flash_heuristic_19

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocessing to identify goal-relevant facts and their global IDs.
        """
        start_time = time.time()
        self.model = model
        
        self.goal_on_fact_ids = []
        # Identify bits in model.goals corresponding to 'on' facts
        if hasattr(self.model, 'goals') and self.model.goals:
            for fact in self.model.facts:
                if (self.model.goals >> fact.global_id) & 1:
                    predicate, args = self._parse_fact_name(fact.name)
                    if predicate == "on":
                        self.goal_on_fact_ids.append(fact.global_id)
        
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse grounded fact names into predicate and arguments.
        Handles both +predicate[arg1,arg2] and (predicate arg1 arg2) formats.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _calculate_heuristic(self, state_bitwise: int, task_network) -> float:
        """
        Admissible Hamming distance for lamps.
        Each mismatch between current state and goal lamp status costs at least 1 flip.
        """
        h = 0
        # Term 1: Hamming distance on goal bits
        for fact_id in self.goal_on_fact_ids:
            # If goal bit is not set in current state
            if not ((state_bitwise >> fact_id) & 1):
                h += 1
        
        # Term 2: Task network tie-breaking
        # Small penalty for network size to guide decomposition toward the end_game method
        # when the goal is already satisfied.
        h += 0.001 * len(task_network)
        
        return float(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._calculate_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic19(goals={len(self.goal_on_fact_ids)})"

    def __str__(self):
        return f"LampsGemini3FlashHeuristic19 [Goals: {len(self.goal_on_fact_ids)}]"

    def __output__(self):
        return (
            f"Lamps Gemini3 Flash Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Lamps Tracked: {len(self.goal_on_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
