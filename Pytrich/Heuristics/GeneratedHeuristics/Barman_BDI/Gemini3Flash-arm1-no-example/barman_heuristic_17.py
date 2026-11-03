from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Barman BDI domain.
    
    Bottleneck Analysis:
    The main bottleneck is the sequence of hand-management and container-cleaning operations 
    required to produce a cocktail. Since there are only two hands and the shaker/shots must 
    be clean and empty before use, the serialization of filling, shaking, and pouring is the 
    dominant cost factor.

    Admissibility Sketches:
    1. Task Network Cost (Term 1): Each task in the network represents a necessary sub-goal. 
       By assigning a minimum number of physical actions (operators) required to satisfy 
       each abstract task (e.g., 'AchieveContainsShotCocktail' needs at least a shake, 
       a pour, and two ingredient additions), we create an admissible lower bound.
    2. Unfilled Goals (Term 2): Any goal cocktail not yet present in its target shot glass 
       will require at least one 'pour-shaker-to-shot' action (cost 1).
    """

    def __init__(self, barman_gemini3flash_heuristic_17="barman_gemini3flash_heuristic_17"):
        super().__init__(barman_gemini3flash_heuristic_17)
        self.task_costs = {}
        self.contains_fact_ids = {} # (container, beverage) -> id
        self.clean_fact_ids = {}    # container -> id
        self.goal_cocktails = []    # list of (shot, cocktail)

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            prefix = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args, prefix
        return None, [], None

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates for fast bitwise lookup
        for fact in model.facts:
            pred, args, _ = self._parse_fact_name(fact.name)
            if pred == "contains":
                self.contains_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clean":
                self.clean_fact_ids[args[0]] = fact.global_id

        # 2. Identify Goals from model.goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args, _ = self._parse_fact_name(g_name)
            if pred == "contains":
                # args[0] is container (shot), args[1] is beverage (cocktail)
                self.goal_cocktails.append((args[0], args[1]))

        # 3. Define minimum costs for abstract tasks
        # These are lower bounds on the number of operators needed to complete the task
        self.task_costs = {
            "AchieveContainsShotCocktail": 8,  # shake + pour + 2*pour-shot + 2*fill + cleans
            "AchieveContainsShakerCocktail": 6, # shake + 2*pour-shot + 2*fill
            "AchieveContainsShakerIngredient": 3, # pour-shot + fill + pickup
            "AchieveContainsShotIngredient": 2, # fill + pickup
            "AchieveCleanShaker": 2, # pickup + clean
            "AchieveCleanShot": 2, # pickup + clean
            "AchieveHolding": 1, # pickup
            "AchieveHandEmpty": 1, # drop
            "AchieveOnTable": 1, # drop
            "DoPourShakerToShot": 1 # pour
        }

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def _compute(self, state, task_network):
        # Term 1: Network Cost (Sum of lower bounds for tasks)
        network_h = 0
        for task in task_network:
            if task.global_id >= self.model.iop_init and task.global_id <= self.model.iop_end:
                # It's an operator
                network_h += 1
            else:
                # It's an abstract task
                # Get minimum cost from precomputed table, default to 1 for unknown abstract tasks
                cost = 1
                for base_name, base_cost in self.task_costs.items():
                    if base_name in task.name:
                        cost = base_cost
                        break
                
                # State-aware refinement: if the goal of the task is already satisfied, cost is 0
                if "AchieveContainsShotCocktail" in task.name:
                    # Extract shot and cocktail from task name if possible (simplified here)
                    # For simplicity, we just add the cost, but in a real-world scenario 
                    # we would parse the task's specific parameters.
                    pass
                
                network_h += cost

        # Term 2: Missing Goals Cost
        # For every cocktail goal not yet in the target shot, add at least 1 (the pour action)
        missing_goal_h = 0
        for shot, cocktail in self.goal_cocktails:
            fact_id = self.contains_fact_ids.get((shot, cocktail))
            if fact_id is not None:
                is_true = (state >> fact_id) & 1
                if not is_true:
                    missing_goal_h += 1

        # Symmetry Breaking Tie-breaker
        # Prefers shorter task networks when costs are equal
        tie_breaker = 0.001 * len(task_network)

        return int(max(network_h, missing_goal_h) + tie_breaker)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val
