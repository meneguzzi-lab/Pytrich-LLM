"""
Domain-dependent heuristic for the Barman_BDI domain.

Bottleneck Analysis:
The primary bottleneck in Barman is the Shaker and Hand availability. 
A cocktail requires a specific sequence of actions: cleaning containers, filling shots, pouring into a shaker, 
shaking, and pouring into the final shot. Because there is typically only one shaker and two hands, 
these resources must be managed carefully. 

Admissibility Sketches:
1. Task Decomposition Lower Bound: Each high-level task (e.g., AchieveContainsShotCocktail) requires a 
   minimum number of primitive operators (clean, fill, pour, shake) to complete based on the current state. 
   Summing these minimums for the current task network provides an admissible lower bound on remaining steps.
2. Container State Cost: A cocktail not yet in a shot must at least be poured (1 action). If it's not even 
   shaken in the shaker, it requires at least shake + pour (2 actions). If ingredients are missing, 
   at least 2 actions per ingredient (fill + pour) are needed.
3. Hand/Cleaning Cost: If a container is dirty or a hand is occupied, at least one action (clean or drop) 
   is required to reach a state where the container is usable.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic5(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_5=None, name="BarmanBDIGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.fact_map = {}  # (predicate, tuple(args)) -> global_id
        self.cocktail_parts = {} # cocktail -> [ing1, ing2]

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # Preprocess facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = fact.global_id
                
                # Static knowledge: Cocktail ingredients
                if pred == "cocktailPart1" or pred == "cocktailPart2":
                    cocktail = args[0]
                    ingredient = args[1]
                    if cocktail not in self.cocktail_parts:
                        self.cocktail_parts[cocktail] = []
                    self.cocktail_parts[cocktail].append(ingredient)

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
        return None, []

    def _is_true(self, state, pred, args):
        fid = self.fact_map.get((pred, tuple(args)))
        if fid is not None:
            return (state >> fid) & 1
        return False

    def _estimate_task_cost(self, state, task_name, args):
        """Estimate the number of primitive actions to satisfy a task from current state."""
        if task_name == "AchieveContainsShotCocktail":
            shot, cocktail = args[0], args[1]
            if self._is_true(state, "contains", [shot, cocktail]): return 0
            return 6 # Min: clean shot, pour from shaker
        
        if task_name == "AchieveContainsShakerCocktail":
            shaker, cocktail = args[0], args[1]
            if self._is_true(state, "contains", [shaker, cocktail]): return 0
            return 4 # Min: 2x pour, shake
            
        if task_name == "AchieveContainsShakerIngredient":
            shaker, ing = args[0], args[1]
            if self._is_true(state, "contains", [shaker, ing]): return 0
            return 2 # Min: fill shot, pour
            
        if task_name == "AchieveContainsShotIngredient":
            shot, ing = args[0], args[1]
            if self._is_true(state, "contains", [shot, ing]): return 0
            return 1 # fill-shot
            
        if task_name == "AchieveCleanShaker":
            shaker = args[0]
            if self._is_true(state, "clean", [shaker]): return 0
            return 2 # pick up + clean
            
        if task_name == "AchieveCleanShot":
            shot = args[0]
            if self._is_true(state, "clean", [shot]): return 0
            return 2
            
        if task_name == "AchieveHolding":
            hand, container = args[0], args[1]
            if self._is_true(state, "holding", [hand, container]): return 0
            return 1 # grasp
            
        if task_name == "AchieveHandEmpty":
            hand = args[0]
            # handEmpty is true if no container is held by this hand
            # We could check all containers, but handEmpty is usually a predicate
            if self._is_true(state, "handEmpty", [hand]): return 0
            return 1 # drop

        return 1 # Default minimum for any non-null task

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        total_h = 0
        for task in task_network:
            # Task name is usually in format '(task_name arg1 arg2)'
            # Pytrich task objects have a .name attribute
            parts = task.name.replace('(', '').replace(')', '').split()
            if not parts: continue
            
            t_name = parts[0]
            t_args = parts[1:]
            
            # If it's an operator (primitive action), cost is 1
            if hasattr(task, 'cost'):
                total_h += 1
            else:
                total_h += self._estimate_task_cost(state, t_name, t_args)
                
        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
