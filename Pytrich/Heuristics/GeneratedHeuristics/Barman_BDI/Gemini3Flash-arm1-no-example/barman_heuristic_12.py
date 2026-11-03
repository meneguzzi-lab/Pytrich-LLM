from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

# Bottleneck Analysis:
# The primary bottleneck in the Barman_BDI domain is the single Shaker (or limited number of shakers).
# To produce any cocktail, the shaker must be cleaned, filled with two distinct ingredients, 
# shaken, and then poured. Each of these steps involves hand usage and potentially cleaning shots.
# This serialization around the shaker and the hands dominates plan length.

# Admissible Lower Bounds:
# 1. Task Network Progress (h_tasks): Sum of minimal actions required to complete each ground task in the network.
#    - Each cocktail goal task in the network (not yet satisfied) requires at least 5 actions (clean, 2x fill, shake, pour).
#    - Each clean/holding task requires at least 1 action if not satisfied.
# 2. State-Goal Gap (h_goals): Number of unsatisfied goal predicates from model.goals.
#    - Each missing cocktail goal is a lower bound on the remaining plan length.

class BarmanBDIGemini3FlashHeuristic12(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_12="BarmanBDIGemini3FlashHeuristic12"):
        super().__init__(barman_gemini3flash_heuristic_12)
        self.fact_ids = {}
        self.shakers = []
        self.shots = []
        self.cocktail_recipes = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        # Parse all facts and map them to global IDs
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self.fact_ids[(pred, tuple(args))] = f.global_id
                if pred == "clean":
                    # Identify shakers and shots based on objects found in clean predicates
                    obj = args[0]
                    if "shaker" in obj.lower():
                        if obj not in self.shakers: self.shakers.append(obj)
                    elif "shot" in obj.lower():
                        if obj not in self.shots: self.shots.append(obj)

        # Identify cocktail recipes from initial state
        for f_id in range(model.ifacts_end + 1):
            if (initial_node.state >> f_id) & 1:
                f = model.facts[f_id]
                pred, args = self._parse_fact_name(f.name)
                if pred in ["cocktailPart1", "cocktailPart2"]:
                    cocktail, ingredient = args[0], args[1]
                    if cocktail not in self.cocktail_recipes: 
                        self.cocktail_recipes[cocktail] = []
                    self.cocktail_recipes[cocktail].append(ingredient)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network: list) -> int:
        h_tasks = 0
        for task in task_network:
            # If it's an operator, it's 1 action
            if hasattr(task, 'cost'):
                h_tasks += task.cost
                continue
            
            # If it's an abstract task, estimate based on name and state
            t_type, t_args = self._parse_task_name(task.name)
            h_tasks += self._estimate_task_cost(t_type, t_args, state)

        # Check model.goals if task network is empty or near completion
        h_goals = 0
        if self.model.goals != 0:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if not self._is_true(state, pred, args):
                    if pred == "contains": # Usually contains shot cocktail
                        h_goals += 5
                    else:
                        h_goals += 1
        
        return max(h_tasks, h_goals)

    def _estimate_task_cost(self, t_type, t_args, state) -> int:
        if t_type == "AchieveContainsShotCocktail":
            shot, cocktail = t_args[0], t_args[1]
            if self._is_true(state, "contains", (shot, cocktail)): return 0
            # Check if shaker already has it
            for shaker in self.shakers:
                if self._is_true(state, "contains", (shaker, cocktail)): return 2
            return 8
        
        if t_type == "AchieveContainsShakerCocktail":
            shaker, cocktail = t_args[0], t_args[1]
            if self._is_true(state, "contains", (shaker, cocktail)): return 0
            # Check if shaker has parts
            parts = self.cocktail_recipes.get(cocktail, [])
            parts_in = sum(1 for p in parts if self._is_true(state, "contains", (shaker, p)))
            if parts_in == 2: return 1 # Just shake
            return 5
            
        if t_type == "AchieveCleanShot":
            if self._is_true(state, "clean", (t_args[0],)): return 0
            return 1
            
        if t_type == "AchieveCleanShaker":
            if self._is_true(state, "clean", (t_args[0],)): return 0
            return 1
            
        if t_type == "AchieveHolding":
            # arg0 is hand, arg1 is container
            if self._is_true(state, "holding", (t_args[0], t_args[1])): return 0
            return 1

        if t_type == "AchieveHandEmpty":
            if self._is_true(state, "handEmpty", (t_args[0],)): return 0
            return 1
            
        if t_type == "AchieveContainsShakerIngredient":
            if self._is_true(state, "contains", (t_args[0], t_args[1])): return 0
            return 2
            
        if t_type == "AchieveContainsShotIngredient":
            if self._is_true(state, "contains", (t_args[0], t_args[1])): return 0
            return 1

        return 1

    def _is_true(self, state: int, pred: str, args: tuple) -> bool:
        fid = self.fact_ids.get((pred, tuple(args)))
        if fid is None: return False
        return (state >> fid) & 1

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

    def _parse_task_name(self, name: str):
        name = name.strip()
        bracket_idx = name.find("(")
        if bracket_idx > 0 and name.endswith(")"):
            task_type = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = [a.strip() for a in args_str.split(",")]
            return task_type, args
        parts = name.split()
        return parts[0], parts[1:]
