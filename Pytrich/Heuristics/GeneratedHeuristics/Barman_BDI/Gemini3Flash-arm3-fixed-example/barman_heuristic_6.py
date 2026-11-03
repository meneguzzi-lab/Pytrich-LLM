"""
Domain-dependent heuristic for the Barman_BDI domain.

Bottleneck Analysis:
The primary bottleneck in this domain is the Shaker. Only one cocktail can be mixed at a time, 
requiring a strict sequence of cleaning, filling with specific ingredients, shaking, and pouring.
Hands and shots also act as secondary serializing resources.

Admissibility Sketches:
1. Task Network Cost: The heuristic sums the minimum remaining actions for each task in the 
   current task network. This is a lower bound because every non-satisfied abstract task must 
   eventually decompose into at least its minimum number of constituent actions, and each 
   operator costs at least 1.
2. State-Aware Task Filtering: If a task's objective (e.g., AchieveCleanShot) is already 
   satisfied in the current state, its cost is 0. This remains admissible as the 
   task will decompose into a 'Null' method.
3. Goal Distance: If the task network is empty but goals are not met, the cost is effectively 
   infinite (represented by a large constant), as the HTN planner cannot progress.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic6(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_6=None, name="BarmanBDIGemini3FlashHeuristic6"):
        super().__init__(name=name)
        self.fact_map = {} # (predicate, args_tuple) -> global_id
        self.task_cost_map = {
            "AchieveContainsShotCocktail": 8,
            "AchieveContainsShakerCocktail": 6,
            "AchieveContainsShakerIngredient": 3,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }
        self.task_to_pred = {
            "AchieveContainsShotCocktail": "contains",
            "AchieveContainsShakerCocktail": "contains",
            "AchieveContainsShakerIngredient": "contains",
            "AchieveContainsShotIngredient": "contains",
            "AchieveCleanShot": "clean",
            "AchieveCleanShaker": "clean",
            "AchieveHolding": "holding",
            "AchieveHandEmpty": "handEmpty",
            "AchieveOnTable": "ontable"
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        # Preprocess fact mapping
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = fact.global_id
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _parse_task(self, task):
        # task.name is usually "(Predicate Arg1 Arg2)"
        inner = task.name.strip("()")
        parts = inner.split()
        if not parts:
            return None, []
        return parts[0], parts[1:]

    def _is_satisfied(self, pred, args, state_bitwise):
        fid = self.fact_map.get((pred, tuple(args)))
        if fid is not None:
            return (state_bitwise >> fid) & 1
        return False

    def _compute(self, state, task_network):
        if not task_network:
            # If network is empty, check if global goals are met
            if (state & self.model.goals) == self.model.goals:
                return 0
            else:
                return 1000  # Dead end for HTN if TN empty but goals not reached

        h = 0
        for task in task_network:
            # Check if it is an Operator (Action) or AbstractTask
            # Operators usually don't have decompositions, AbstractTasks do.
            if hasattr(task, 'decompositions'): # AbstractTask
                t_name, t_args = self._parse_task(task)
                
                # State-aware cost: if task goal already true, cost is 0
                pred = self.task_to_pred.get(t_name)
                if pred and self._is_satisfied(pred, t_args, state):
                    h += 0
                else:
                    h += self.task_cost_map.get(t_name, 1)
            else: # Operator
                h += 1
        
        # Add a tiny penalty for holding containers to encourage clearing hands
        # This helps break symmetries between interchangeable shots/hands
        for fact in self.model.facts:
            if "holding" in fact.name:
                if (state >> fact.global_id) & 1:
                    h += 0.01

        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
