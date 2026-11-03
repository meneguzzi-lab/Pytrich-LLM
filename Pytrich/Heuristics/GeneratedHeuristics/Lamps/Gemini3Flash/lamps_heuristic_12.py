from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.
    
    1. Bottleneck Analysis:
       The bottleneck in the Lamps domain is the requirement that every lamp (coord x, coord y) 
       must match its specific goal status (Y or N). Each change in status for a lamp 
       necessitates at least one 'flip' primitive action, which costs 1 unit. 
       While flips can propagate, each lamp flipped in a propagation chain still 
       executes a 'flip' action.

    2. Admissible Lower Bounds:
       - Term 1 (Mismatch Count): The number of lamps (x, y) whose current status in the state 
         does not match their target status specified in the goal. Since every lamp 
         that needs to change status must be flipped at least once, and each 'flip' 
         operator has a cost of 1, the mismatch count is an admissible lower bound 
         on the total action cost (excluding method precondition actions).
       - Term 2 (Task Network Tie-breaker): A small penalty based on the task network 
         size helps guide the search to prefer shorter decomposition paths and reach 
         the 'end_game' task sooner.
    """

    def __init__(self, lamps_gemini3flash_heuristic_12=None, name="LampsGemini3FlashHeuristic12"):
        super().__init__(name=name)
        self.goal_on_fact_ids = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocessing to identify which grounded facts correspond to goal status requirements.
        """
        self.model = model
        self.goal_on_fact_ids = []

        # 1. Identify goal requirements from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            goal_requirements = {} # (x, y) -> target_status
            
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                # Expecting 'on[x, y, status]'
                if pred == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    goal_requirements[(x, y)] = status

            # 2. Map target statuses to their global fact IDs
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if pred == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    if goal_requirements.get((x, y)) == status:
                        self.goal_on_fact_ids.append(fact.global_id)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses Pytrich grounded fact format: +predicate[arg1,arg2]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Calculate the number of lamps not yet in their goal status.
        """
        mismatches = 0
        # Every cell whose goal 'on' fact is NOT true is a mismatch
        for fact_id in self.goal_on_fact_ids:
            is_true = (state_bitwise >> fact_id) & 1
            if not is_true:
                mismatches += 1
        
        # Admissible mismatch count + small tie-breaker for task network size
        h = float(mismatches)
        if task_network:
            h += 0.001 * len(task_network)
            
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Heuristic calculation called during search expansion.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic12(goals={len(self.goal_on_fact_ids)})"
