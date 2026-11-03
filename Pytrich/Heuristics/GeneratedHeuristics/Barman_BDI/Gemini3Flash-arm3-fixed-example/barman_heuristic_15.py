"""
Domain-dependent heuristic for the Barman_BDI HTN domain.

Bottleneck Analysis:
The bottleneck in Barman_BDI is the coordinated use of hands and containers (shakers/shots) to sequence ingredients. 
Preparing a single cocktail requires a specific chain of cleaning, filling, and pouring actions that are 
heavily constrained by the two-hand limit and container cleanliness requirements.

Admissibility Sketches:
1. Task Decomposition Cost: Each abstract task in the HTN network represents a known minimum sequence of actions 
   (e.g., AchieveContainsShotCocktail requires at least 10 actions including cleaning, grasping, filling, 
   pouring twice, shaking, and pouring again). Summing these minimum costs for ordered subtasks remains admissible.
2. Goal Predicate Cost: Each unsatisfied goal predicate (contains) requires at least one action to fulfill.
   By taking the max of task-network costs and goal-state costs, we maintain admissibility while improving guidance.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic15(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_15=None, name="BarmanBDIGemini3FlashHeuristic15"):
        super().__init__(name=name)
        self.barman_gemini3flash_heuristic_15 = barman_gemini3flash_heuristic_15
        self.fact_ids = {} # (predicate, tuple(args)) -> global_id
        self.task_costs = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 7,
            "AchieveContainsShakerIngredient": 4,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShot": 2,
            "AchieveCleanShaker": 2,
            "AchieveHandEmpty": 1,
            "AchieveHolding": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Preprocesses the fact map for O(1) state checks.
        """
        self.model = model
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_ids[(pred, tuple(args))] = fact.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Called for every node expansion. Returns the estimated cost to goal.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state_bitwise: int, task_network) -> int:
        # Term 1: HTN Task Network progression cost
        task_h = 0
        for task in task_network:
            if hasattr(task, 'add_effects'): # Primitive Operator
                task_h += 1
            else: # Abstract Task
                task_h += self._estimate_task_cost(task.name, state_bitwise)

        # Term 2: Goal predicate dissatisfaction (if model.goals is populated)
        goal_h = 0
        if hasattr(self.model, 'goals') and self.model.goals:
            # Count bits in model.goals that are NOT in state_bitwise
            diff = self.model.goals & (self.model.goals ^ (self.model.goals & state_bitwise))
            goal_h = bin(diff).count('1')

        return max(task_h, goal_h)

    def _estimate_task_cost(self, task_full_name: str, state: int) -> int:
        parts = task_full_name.split()
        if not parts: 
            return 1
        
        pred = parts[0]
        args = parts[1:]
        
        # Check if the primary goal of the task is already satisfied
        if pred == "AchieveContainsShotCocktail" and self.is_true(state, "contains", args):
            return 0
        if pred == "AchieveContainsShakerCocktail" and self.is_true(state, "contains", args):
            return 0
        if pred == "AchieveContainsShakerIngredient" and self.is_true(state, "contains", args):
            return 0
        if pred == "AchieveContainsShotIngredient" and self.is_true(state, "contains", args):
            return 0
        if pred == "AchieveCleanShot" and self.is_true(state, "clean", args):
            return 0
        if pred == "AchieveCleanShaker" and self.is_true(state, "clean", args):
            return 0
        if pred == "AchieveHolding" and self.is_true(state, "holding", args):
            return 0
        if pred == "AchieveHandEmpty" and self.is_true(state, "handEmpty", args):
            return 0
        if pred == "AchieveOnTable" and self.is_true(state, "ontable", args):
            return 0
        
        return self.task_costs.get(pred, 1)

    def is_true(self, state: int, predicate: str, args: list) -> bool:
        fid = self.fact_ids.get((predicate, tuple(args)))
        if fid is not None:
            return (state >> fid) & 1
        return False

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
