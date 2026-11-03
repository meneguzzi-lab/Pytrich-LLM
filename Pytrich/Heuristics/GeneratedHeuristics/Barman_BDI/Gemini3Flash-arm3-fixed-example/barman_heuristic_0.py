import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Barman_BDI domain.

Bottleneck Analysis:
The Barman domain is constrained by the limited number of hands (2) and the shared use of a single shaker. 
To make a cocktail, a strict sequence of cleaning, filling ingredients into a shot, pouring into a shaker, 
shaking, and then pouring the result into a clean shot must be followed. The bottleneck is the shaker, 
which must be cleaned between different cocktail productions.

Admissibility Sketches:
1. Cocktail Cost Term: Making a cocktail requires a minimum of 7-10 actions (clean shot, fill shot1, pour, 
   clean shot, fill shot2, pour, shake, pour to shot). This heuristic assigns lower bounds based on 
   the current state (e.g., if ingredients are already in the shaker).
2. Task Cost Term: Each abstract task in the HTN network represents a set of unavoidable future actions. 
   By estimating the minimum number of operators required to decompose and fulfill these tasks from the 
   current state, we get an admissible lower bound.
"""

class BarmanBDIGemini3FlashHeuristic0(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_0=None, name="BarmanBDIGemini3FlashHeuristic0"):
        super().__init__(name=name)
        self.fact_ids = {} 
        self.cocktail_parts = {} # cocktail -> (ing1, ing2)
        self.shakers = set()
        self.shots = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Pre-process facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_ids[(pred, tuple(args))] = fact.global_id
                if pred == "cocktailPart1":
                    if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = [None, None]
                    self.cocktail_parts[args[0]][0] = args[1]
                elif pred == "cocktailPart2":
                    if args[0] not in self.cocktail_parts: self.cocktail_parts[args[0]] = [None, None]
                    self.cocktail_parts[args[0]][1] = args[1]
                elif pred == "shakerLevel":
                    self.shakers.add(args[0])
                elif pred == "clean" and "shot" in args[0]:
                    self.shots.add(args[0])

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith(("+", "-")):
            name = name[1:]
            idx = name.find("[")
            if idx > 0 and name.endswith("]"):
                pred = name[:idx]
                args = name[idx+1:-1].split(",")
                return pred, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _parse_task_name(self, name: str):
        name = name.strip("() ")
        parts = name.split()
        if not parts: return None, []
        return parts[0], parts[1:]

    def _is_true(self, state, pred, *args):
        fid = self.fact_ids.get((pred, args))
        if fid is not None:
            return (state >> fid) & 1
        return 0

    def _compute(self, state, task_network):
        h = 0
        for task in task_network:
            pred, args = self._parse_task_name(task.name)
            if not pred: continue

            if pred == "AchieveContainsShotCocktail":
                if len(args) >= 2:
                    h += self._cost_shot_cocktail(state, args[0], args[1])
            elif pred == "AchieveContainsShakerCocktail":
                if len(args) >= 2:
                    h += self._cost_shaker_cocktail(state, args[0], args[1])
            elif pred == "AchieveContainsShakerIngredient":
                if len(args) >= 2:
                    h += self._cost_shaker_ing(state, args[0], args[1])
            elif pred == "AchieveContainsShotIngredient":
                if len(args) >= 2:
                    h += self._cost_shot_ing(state, args[0], args[1])
            elif pred == "AchieveHolding":
                if len(args) >= 2:
                    if not self._is_true(state, "holding", args[0], args[1]):
                        h += 1
            elif pred == "AchieveCleanShot":
                if args and not self._is_true(state, "clean", args[0]):
                    h += 1
            elif pred == "AchieveCleanShaker":
                if args and not self._is_true(state, "clean", args[0]):
                    h += 1
            elif pred == "DoPourShakerToShot":
                h += 1
            elif pred == "AchieveHandEmpty":
                if args and not self._is_true(state, "handEmpty", args[0]):
                    h += 1
            else:
                # Generic cost for actions or other tasks
                h += 1
        return int(h)

    def _cost_shot_cocktail(self, state, shot, cocktail):
        if self._is_true(state, "contains", shot, cocktail): return 0
        
        # Find best shaker to estimate cost
        min_shaker_cost = 10
        for shaker in self.shakers:
            shaker_cost = self._cost_shaker_cocktail(state, shaker, cocktail)
            min_shaker_cost = min(min_shaker_cost, shaker_cost)
        
        # Add costs for pour shaker to shot (1) and cleaning the shot (if dirty)
        total = min_shaker_cost + 1
        if not self._is_true(state, "clean", shot): total += 1
        return total

    def _cost_shaker_cocktail(self, state, shaker, cocktail):
        if self._is_true(state, "contains", shaker, cocktail): return 0
        if self._is_true(state, "shaked", shaker): return 0
        
        ing_pair = self.cocktail_parts.get(cocktail)
        if not ing_pair: return 7 # Default estimate
        
        cost = 1 # shake action
        if not self._is_true(state, "clean", shaker) and not self._is_true(state, "unshaked", shaker):
            cost += 1 # clean shaker
        
        cost += self._cost_shaker_ing(state, shaker, ing_pair[0])
        cost += self._cost_shaker_ing(state, shaker, ing_pair[1])
        return cost

    def _cost_shaker_ing(self, state, shaker, ing):
        if self._is_true(state, "contains", shaker, ing): return 0
        # Cost to fill shot (1) + pour to shaker (1)
        return 2

    def _cost_shot_ing(self, state, shot, ing):
        if self._is_true(state, "contains", shot, ing): return 0
        # Fill shot action (1) + clean shot if needed (1)
        cost = 1
        if not self._is_true(state, "clean", shot): cost += 1
        return cost

    def __repr__(self):
        return f"{self.name}()"
