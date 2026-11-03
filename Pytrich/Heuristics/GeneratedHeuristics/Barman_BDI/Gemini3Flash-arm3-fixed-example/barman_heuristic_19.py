import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Barman_BDI domain.

1. Bottleneck Analysis:
   The Shaker is the central bottleneck. Only one cocktail can be prepared at a time,
   requiring a strict sequence: cleaning, adding ingredients, shaking, and pouring.
   The hands and shots are also limited resources, but the shaker's serial nature
   dictates the overall plan length.

2. Admissible Lower Bounds:
   - h_goals: A state-based estimate. For each cocktail goal (contains shot cocktail) not met,
     we compute the minimum actions required based on the current location of the cocktail
     or its ingredients (e.g., in shot, in shaker shaked, in shaker unshaked, or not yet in shaker).
     Admissibility: Each state transition (pour, shake, fill) is counted as 1 action,
     which is the true cost in the domain.
   - h_tasks: Each task currently in the task network represents at least one future action
     (either an operator or a task that decomposes into operators).
     Admissibility: Every task in this domain requires at least one action to satisfy.

3. Guidance:
   - Tie-breaking: Small penalties for dirty containers (shaker/shots) that are needed for goals.
   - Task Network Weight: len(task_network) is used as a base cost to favor nodes closer to completion.
"""

class BarmanBDIGemini3FlashHeuristic19(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_19=None, name="BarmanBDIGemini3FlashHeuristic19"):
        super().__init__(name=name)
        self.goal_cocktails = {} # shot -> cocktail
        self.cocktail_parts = {} # cocktail -> [ing1, ing2]
        self.contains_ids = {}   # (container, beverage) -> global_id
        self.shaked_ids = {}     # shaker -> global_id
        self.clean_ids = {}      # container -> global_id
        self.empty_ids = {}      # container -> global_id
        self.shakers = []
        self.shots = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map predicates to IDs
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            if pred == "contains" and len(args) == 2:
                self.contains_ids[(args[0], args[1])] = fact.global_id
            elif pred == "shaked" and len(args) == 1:
                self.shaked_ids[args[0]] = fact.global_id
                if args[0] not in self.shakers: self.shakers.append(args[0])
            elif pred == "clean" and len(args) == 1:
                self.clean_ids[args[0]] = fact.global_id
            elif pred == "empty" and len(args) == 1:
                self.empty_ids[args[0]] = fact.global_id
            elif pred == "cocktailPart1" or pred == "cocktailPart2":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.cocktail_parts:
                    self.cocktail_parts[cocktail] = []
                self.cocktail_parts[cocktail].append(ing)

        # 2. Extract Goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                pred, args = self._parse_fact_name(goal_name)
                if pred == "contains" and len(args) == 2:
                    self.goal_cocktails[args[0]] = args[1]
        
        # If no explicit goals (common in some HTN formats), look for tasks in the initial network
        if not self.goal_cocktails:
            for task in initial_node.task_network:
                tn_pred, tn_args = self._parse_fact_name(task.name)
                if tn_pred == "AchieveContainsShotCocktail" and len(tn_args) == 2:
                    self.goal_cocktails[tn_args[0]] = tn_args[1]

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

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
            parts = name[1:-1].replace(",", " ").split()
            if parts: return parts[0], parts[1:]
        # Handle task names like TaskName[arg1,arg2]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return predicate, args
        return None, []

    def _is_true(self, state, fact_id):
        if fact_id is None: return False
        return (state >> fact_id) & 1

    def _compute(self, state, task_network):
        h_goals = 0
        shaker = self.shakers[0] if self.shakers else None

        for shot, cocktail in self.goal_cocktails.items():
            # If goal satisfied
            if self._is_true(state, self.contains_ids.get((shot, cocktail))):
                continue
            
            # If cocktail is in shaker and shaked
            if shaker and self._is_true(state, self.contains_ids.get((shaker, cocktail))) and \
               self._is_true(state, self.shaked_ids.get(shaker)):
                h_goals += 1 # Just needs pouring
            elif shaker and self._is_true(state, self.contains_ids.get((shaker, cocktail))):
                h_goals += 2 # Needs shake + pour
            else:
                # Check ingredients in shaker
                parts = self.cocktail_parts.get(cocktail, [])
                ings_in_shaker = 0
                for ing in parts:
                    if shaker and self._is_true(state, self.contains_ids.get((shaker, ing))):
                        ings_in_shaker += 1
                
                if ings_in_shaker == 2:
                    h_goals += 2 # Needs shake + pour
                elif ings_in_shaker == 1:
                    h_goals += 4 # Needs 1 more ing (fill + pour) + shake + pour
                else:
                    h_goals += 6 # Needs both ings + shake + pour

            # Penalty for dirty shot
            if not self._is_true(state, self.clean_ids.get(shot)):
                h_goals += 1

        # Shaker cleaning penalty
        if shaker and not self._is_true(state, self.clean_ids.get(shaker)) and not self._is_true(state, self.empty_ids.get(shaker)):
            h_goals += 1

        h_tasks = len(task_network)
        return h_goals + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
