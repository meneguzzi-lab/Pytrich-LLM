from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic7(Heuristic):
    """
    Bottleneck Analysis:
    The Shaker is the primary serializing bottleneck in the Barman_BDI domain. Mixing a cocktail 
    requires a clean shaker, multiple ingredients (via shots), and a shaking process. Since there 
    is typically only one shaker, only one cocktail can be in production at a time.

    Admissibility Sketches:
    1. Task Network Term: Every AbstractTask in the task network has a minimum number of primitive 
       operators it must decompose into to be satisfied. By summing these conservative lower bounds 
       (e.g., 6 for AchieveContainsShotCocktail), we obtain an admissible estimate of future actions.
    2. State-Based Goal Term: Every unsatisfied 'contains' goal for a cocktail requires at least 
       a pour action from the shaker. If the cocktail is not yet shaked in the shaker, it requires 
       additional actions (shake, fill, pour-shot). Counting 2 actions per missing cocktail goal 
       is a safe lower bound.
    """

    def __init__(self, barman_gemini3flash_heuristic_7="barman_heuristic_7"):
        super().__init__(barman_gemini3flash_heuristic_7)
        self.fact_map = {}  # predicate -> args_tuple -> global_id
        self.goal_cocktails = []  # list of (fact_id, args_tuple)
        self.task_min_costs = {
            "AchieveContainsShotCocktail": 6,
            "AchieveContainsShakerCocktail": 4,
            "AchieveContainsShakerIngredient": 2,
            "AchieveContainsShotIngredient": 1,
            "AchieveCleanShaker": 1,
            "AchieveCleanShot": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }
        self._task_name_cache = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates to global IDs for fast bitwise checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                if pred not in self.fact_map:
                    self.fact_map[pred] = {}
                self.fact_map[pred][tuple(args)] = fact.global_id

        # 2. Identify cocktail goals from the model's goal bitset
        goal_bits = model.goals
        for bit in range(goal_bits.bit_length()):
            if (goal_bits >> bit) & 1:
                fact = model.facts[bit]
                pred, args = self._parse_fact_name(fact.name)
                if pred == "contains":
                    self.goal_cocktails.append((bit, tuple(args)))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 1: Task Network Complexity
        task_h = 0
        for task in task_network:
            # If it's an Operator, use its cost
            if hasattr(task, 'cost'):
                task_h += task.cost
            else:
                # It's an AbstractTask, use precomputed min cost
                t_name, t_args = self._parse_task_args(task.name)
                cost = self.task_min_costs.get(t_name, 1)
                
                # State-aware reduction: check if task goal is already satisfied
                if t_name == "AchieveContainsShotCocktail" and self._check_fact(state, "contains", t_args):
                    cost = 0
                elif t_name == "AchieveContainsShakerCocktail" and self._check_fact(state, "contains", t_args):
                    cost = 0
                elif t_name == "AchieveCleanShot" and self._check_fact(state, "clean", t_args):
                    cost = 0
                elif t_name == "AchieveCleanShaker" and self._check_fact(state, "clean", t_args):
                    cost = 0
                elif t_name == "AchieveHandEmpty" and self._check_fact(state, "handEmpty", t_args):
                    cost = 0
                elif t_name == "AchieveHolding" and self._check_fact(state, "holding", t_args):
                    cost = 0
                elif t_name == "AchieveOnTable" and self._check_fact(state, "ontable", t_args):
                    cost = 0
                
                task_h += cost

        # Term 2: Goal State Lower Bound
        goal_h = 0
        for fact_id, _ in self.goal_cocktails:
            if not ((state >> fact_id) & 1):
                goal_h += 2 # Min cost to satisfy a container-cocktail goal

        # Combine terms and add a small tie-breaker penalty for network depth
        h = max(task_h, goal_h)
        return h + (0.001 * len(task_network))

    def _check_fact(self, state, pred, args):
        f_id = self.fact_map.get(pred, {}).get(tuple(args))
        if f_id is not None:
            return (state >> f_id) & 1
        return False

    def _parse_task_args(self, task_name: str):
        if task_name in self._task_name_cache:
            return self._task_name_cache[task_name]
        parts = task_name.split()
        res = (parts[0], parts[1:])
        self._task_name_cache[task_name] = res
        return res

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
