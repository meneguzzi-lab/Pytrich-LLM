"""
Domain-dependent heuristic for the Barman_BDI HDDL domain.

1. Bottleneck Analysis:
   The central bottleneck in the Barman domain is the Shaker and the sequence of actions 
   required to produce a single cocktail (cleaning, filling with two ingredients, 
   shaking, and pouring). Since there is usually only one shaker, these cycles 
   are largely serial.

2. Admissible Lower Bounds:
   - T1 (Task Decomposition Cost): Each task in the task network has a minimum number 
     of primitive actions required to satisfy it. For example, 'AchieveContainsShotCocktail' 
     requires at least 7-10 actions if nothing is prepared. We check the current state 
     to see if 'Null' methods (no-op) are applicable to reduce this cost to 0.
   - T2 (Unsatisfied Goals): Each goal 'contains(shot, cocktail)' in the goal state 
     that is not yet satisfied in the current state requires at least 1 action (pour) 
     plus the costs of shaking and cleaning if not already done.

This heuristic returns the sum of estimated costs for the current task network, 
refining each task's cost by checking the current state for 'Null' conditions.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic13(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_13=None, name="BarmanBDIGemini3FlashHeuristic13"):
        super().__init__(name=name)
        self.fact_map = {}
        self.task_costs = {
            "AchieveContainsShotCocktail": 10,
            "AchieveContainsShakerCocktail": 8,
            "AchieveContainsShakerIngredient": 3,
            "AchieveContainsShotIngredient": 2,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1,
            "DoPourShakerToShot": 1
        }

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

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        # Map predicates to global IDs for fast bitwise checks
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred not in self.fact_map:
                self.fact_map[pred] = {}
            self.fact_map[pred][tuple(args)] = fact.global_id
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_true(self, state_bitwise: int, predicate: str, args: list) -> bool:
        if predicate in self.fact_map:
            fact_id = self.fact_map[predicate].get(tuple(args))
            if fact_id is not None:
                return (state_bitwise >> fact_id) & 1
        return False

    def _compute(self, state: int, task_network) -> int:
        total_h = 0
        for task in task_network:
            task_name = task.name
            # Basic task name check (tasks might be grounded strings in some Pytrich versions)
            base_name = task_name
            args = []
            if " " in task_name:
                parts = task_name.replace("(", "").replace(")", "").split()
                base_name = parts[0]
                args = parts[1:]
            elif "[" in task_name:
                bracket_idx = task_name.find("[")
                base_name = task_name[:bracket_idx]
                args = task_name[bracket_idx+1:-1].split(",")

            # Apply state-based reductions (Null method logic)
            if base_name == "AchieveContainsShotCocktail":
                if self._is_true(state, "contains", args): continue
                total_h += 10
            elif base_name == "AchieveContainsShakerCocktail":
                if self._is_true(state, "contains", args): continue
                total_h += 8
            elif base_name == "AchieveContainsShakerIngredient":
                if self._is_true(state, "contains", args): continue
                total_h += 3
            elif base_name == "AchieveContainsShotIngredient":
                if self._is_true(state, "contains", args): continue
                total_h += 2
            elif base_name == "AchieveCleanShot":
                if self._is_true(state, "clean", args): continue
                total_h += 1
            elif base_name == "AchieveCleanShaker":
                if self._is_true(state, "clean", args): continue
                total_h += 1
            elif base_name == "AchieveHolding":
                if self._is_true(state, "holding", args): continue
                total_h += 1
            elif base_name == "AchieveHandEmpty":
                if self._is_true(state, "handEmpty", args): continue
                total_h += 1
            elif base_name == "AchieveOnTable":
                if self._is_true(state, "ontable", args): continue
                total_h += 1
            else:
                total_h += self.task_costs.get(base_name, 1)

        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
