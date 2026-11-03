import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

1. Bottleneck Analysis:
   The bottleneck of the Lamps domain is the total number of coordinates where the current status
   does not match the goal status. Every lamp that is incorrectly set must be changed via a 'flip' 
   action. A 'turn' initiates one manual 'flip' and triggers a cascade of 'flip' actions via propagation.
   However, each 'flip' is a distinct primitive action with a cost of 1.

2. Admissible Lower Bounds:
   - Term 1: Lamp Mismatch Count. Every coordinate (x, y) that is currently set to status S1 but
     required to be S2 must undergo at least one 'flip' action. Since each 'flip' primitive
     costs 1 and only affects one lamp, the number of mismatches is a strict lower bound on the 
     number of 'flip' actions required to reach the goal configuration.

Admissibility Sketch:
   H(n) = count(incorrect_lamps). Since every 'flip' costs 1 and flips exactly one lamp at (x, y),
   changing N mismatched lamps requires at least N flip actions. Thus, H(n) <= cost_to_goal.
"""

class LampsGemini3FlashHeuristic8(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_8=None, name="LampsGemini3FlashHeuristic8"):
        super().__init__(name=name)
        self.goal_on_fact_ids = {}  # (x, y) -> global_id of the required status fact
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once at the start of search. Precomputes goal requirements and fact IDs.
        """
        start_time = time.time()
        self.model = model
        
        # 1. Parse goals from model.goals (bitwise representation)
        goal_status = {} # (x, y) -> status_str
        if self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_fact_names:
                predicate, args = self._parse_fact_name(name)
                if predicate == "on" and len(args) == 3:
                    # Format: on[x, y, status]
                    x, y, s = args[0], args[1], args[2]
                    goal_status[(x, y)] = s

        # 2. Map coordinates to the global_id of the 'on' fact for the target status
        # This allows O(1) checking per goal lamp in the search loop.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                if (x, y) in goal_status and goal_status[(x, y)] == s:
                    self.goal_on_fact_ids[(x, y)] = fact.global_id

        self.preprocessing_time = time.time() - start_time
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses Pytrich grounded fact format: +predicate[arg1,arg2]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # strip sign
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _compute(self, state: int, task_network) -> float:
        """
        Calculates the mismatch count: number of goal lamps not in their target status.
        """
        mismatches = 0
        for fact_id in self.goal_on_fact_ids.values():
            # If the bit for the target 'on' fact is NOT set, the lamp is in the wrong status
            if not ((state >> fact_id) & 1):
                mismatches += 1
        
        # Tie-breaker: prefer smaller task networks to encourage finishing decompositions
        tie_breaker = 0.001 * len(task_network)
        
        return float(mismatches + tie_breaker)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every node expansion.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic8(goals={len(self.goal_on_fact_ids)})"

    def __str__(self):
        return f"Lamps mismatch heuristic ({len(self.goal_on_fact_ids)} targets)"

    def __output__(self):
        return (
            f"Lamps Gemini3 Flash Heuristic 8 Info:\n"
            f"\tTarget Lamps: {len(self.goal_on_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
