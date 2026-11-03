from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from Pytrich.Operators.operator import Operator

class BarmanBDIGemini3FlashHeuristic10(Heuristic):
    """
    Barman_BDI Bottleneck Analysis:
    The shaker and hands are the primary serializing resources. Making a cocktail requires a clean shaker, 
    preparing two ingredients in shots, transferring them to the shaker, shaking, and pouring into a clean target shot.

    Admissible Lower Bounds:
    1. Task Network Remaining Work (State-Aware): Each abstract task has a minimum number of decomposition steps 
       required to reach primitive operators. If the task's objective is already met in the current state, 
       its cost is 0. Otherwise, we use a conservative estimate of the actions needed (e.g., 10 for a cocktail).
    2. Goal-Based Progress: Each unsatisfied goal cocktail requires a sequence of preparation steps. 
       We estimate the cost based on where the cocktail or its ingredients currently reside (dispenser, shot, or shaker).
    3. Cleaning and Hand Costs: Any container used in a task that is not clean or not empty requires cleaning actions.
    """

    def __init__(self, barman_gemini3flash_heuristic_10="barman_gemini3flash_heuristic_10"):
        super().__init__(barman_gemini3flash_heuristic_10)
        self.fact_id_map = {}
        self.cocktail_parts = {}
        self.goal_cocktails = []
        self.task_min_cost = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 7,
            "AchieveContainsShakerIngredient": 3,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHandEmpty": 1,
            "AchieveHolding": 1,
            "DoPourShakerToShot": 1,
            "AchieveOnTable": 1
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_id_map[(pred, tuple(args))] = fact.global_id
                if pred == "cocktailPart1":
                    if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = [None, None]
                    self.cocktail_parts[args[0]][0] = args[1]
                elif pred == "cocktailPart2":
                    if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = [None, None]
                    self.cocktail_parts[args[0]][1] = args[1]

        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains":
                fid = self.fact_id_map.get((pred, tuple(args)))
                if fid is not None:
                    self.goal_cocktails.append((args[0], args[1], fid))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        tn_cost = 0
        for task in task_network:
            if isinstance(task, Operator):
                tn_cost += task.cost
            else:
                prefix, args = self._parse_task_name(task.name)
                tn_cost += self._get_task_cost(prefix, args, state)

        goal_cost = 0
        for shot, cocktail, fid in self.goal_cocktails:
            if not ((state >> fid) & 1):
                goal_cost += self._get_goal_remaining_cost(shot, cocktail, state)

        cleaning_cost = 0
        for fact_key, fid in self.fact_id_map.items():
            pred, args = fact_key
            if pred == "clean":
                if not ((state >> fid) & 1):
                    cleaning_cost += 0.5
            elif pred == "handEmpty":
                if not ((state >> fid) & 1):
                    cleaning_cost += 0.1

        return int(max(tn_cost, goal_cost, cleaning_cost))

    def _get_task_cost(self, prefix, args, state):
        if prefix == "AchieveContainsShotCocktail":
            return 0 if self._is_true(state, "contains", (args[0], args[1])) else 10
        if prefix == "AchieveContainsShakerCocktail":
            return 0 if self._is_true(state, "contains", (args[0], args[1])) else 7
        if prefix == "AchieveContainsShakerIngredient":
            return 0 if self._is_true(state, "contains", (args[0], args[1])) else 3
        if prefix == "AchieveCleanShot":
            return 0 if self._is_true(state, "clean", (args[0],)) else 1
        if prefix == "AchieveCleanShaker":
            return 0 if self._is_true(state, "clean", (args[0],)) else 1
        return self.task_min_cost.get(prefix, 1)

    def _get_goal_remaining_cost(self, shot, cocktail, state):
        shaker = "shaker1"
        if self._is_true(state, "contains", (shaker, cocktail)):
            return 1
        parts = self.cocktail_parts.get(cocktail)
        if parts:
            i1, i2 = parts
            c1 = self._is_true(state, "contains", (shaker, i1))
            c2 = self._is_true(state, "contains", (shaker, i2))
            if c1 and c2:
                return 2
            return 5
        return 8

    def _is_true(self, state, pred, args):
        fid = self.fact_id_map.get((pred, tuple(args)))
        return fid is not None and (state >> fid) & 1

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

    def _parse_task_name(self, name):
        parts = name.split(" ")
        return parts[0], parts[1:]
