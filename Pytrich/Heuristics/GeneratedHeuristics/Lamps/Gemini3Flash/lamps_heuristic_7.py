import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

1. Bottleneck Analysis:
   The bottleneck in the Lamps domain is the total number of 'flip' actions required to synchronize 
   the current lamp statuses with the goal configuration. Since every 'flip' costs 1, and each lamp
   at a specific (x,y) can only be in one status (Y or N), the number of unsatisfied goal status 
   requirements provides a direct and admissible lower bound on the cost to reach the goal.

2. Admissible Lower Bound:
   - Term 1 (Hamming Distance): Every goal status on(x, y, s) that is not currently true in the state
     must be achieved by at least one 'flip' action. Since each 'flip' costs 1, the number of 
     mismatched goal lamp statuses is an admissible lower bound on the total cost. 
     Admissibility Sketch: Cost = flips + precondition_checks. My term = num_missing_goal_on_facts. 
     Since 1 flip resolves at most 1 missing on-fact at a coordinate (x,y), and cost is 1 per flip, 
     num_missing_goal_on_facts <= total flip actions <= total cost.

3. Tie-breaking:
   A small penalty (0.001) is added for every task remaining in the task network. This encourages 
   the planner to prefer nodes that are closer to completing their decomposition hierarchy when 
   the Hamming distance is identical.
"""

class LampsGemini3FlashHeuristic7(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_7=None, name="LampsGemini3FlashHeuristic7"):
        super().__init__(name=name)
        self.goal_on_fact_ids = []
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Pre-calculates the IDs of all goal on(...) facts.
        """
        start_time = time.time()
        self.model = model
        
        # Get explicit goal representation from the bitwise model.goals
        goal_fact_names = set(self.model.state_explicit_repr(self.model.goals))
        
        # Identify the global_ids of all 'on' facts that are specified in the goal
        self.goal_on_fact_ids = []
        for fact in self.model.facts:
            if fact.name in goal_fact_names:
                # Check if it's an 'on' predicate via name parsing
                predicate, args = self._parse_fact_name(fact.name)
                if predicate == "on":
                    self.goal_on_fact_ids.append(fact.global_id)
                    
        self.preprocessing_time = time.time() - start_time
        
        # Initial compute
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """Helper to extract predicate and arguments from a grounded fact string."""
        name = name.strip()
        # Format 1: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        # Format 2: (predicate arg1 arg2)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        """
        Calculates the Hamming distance (mismatched lamps) + a small tie-breaker for task network length.
        """
        # Count how many goal statuses are NOT currently satisfied
        # Each mismatch represents at least one flip action (cost 1)
        mismatches = 0
        for fact_id in self.goal_on_fact_ids:
            # If the bit is 0, the fact is not in the state
            if not ((state_bitwise >> fact_id) & 1):
                mismatches += 1
        
        # Guidance: Small penalty per task in the network to break ties towards completion.
        network_penalty = len(task_network) * 0.001
        
        return float(mismatches + network_penalty)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every node expansion. Efficiently evaluates the current state.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}()"

    def __str__(self):
        return f"{self.name}"

    def __output__(self):
        return (
            f"Lamps Gemini3 Flash Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal On-Facts Monitored: {len(self.goal_on_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Search Calls: {self.calls}\n"
            f"\tMin H-Value Encountered: {self.min_hvalue}\n"
        )
