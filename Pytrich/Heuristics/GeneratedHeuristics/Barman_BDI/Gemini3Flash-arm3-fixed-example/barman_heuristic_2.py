"""
Domain-dependent heuristic for the Barman_BDI domain.

Bottleneck Analysis:
The primary bottleneck in the Barman domain is the Shaker and the Hands. 
To produce a cocktail, a shaker must go through a serial process: 
Clean -> Fill Ingredient 1 -> Fill Ingredient 2 -> Shake -> Pour. 
Since there is usually only one shaker and only two hands, cocktail production 
is strictly serial. 

Admissibility Sketches:
1. Cocktail Progress Cost: For each cocktail goal (AchieveContainsShotCocktail), we estimate 
   the minimum number of actions based on the current state of the shaker and shot.
   - If shot contains cocktail: 0
   - If cocktail is in shaker and shaked: 1 (pour)
   - If cocktail is in shaker but unshaked: 2 (shake + pour)
   - For each missing ingredient in shaker: +2 (fill shot + pour)
   - If shaker needs cleaning: +1
   This is admissible because these actions are strictly required and serial.
2. Task Network Summation: We sum the state-aware costs of all tasks in the current 
   HTN task network. Since the task network represents the remaining work to be done, 
   and Barman tasks are mostly serial, this sum provides a strong lower bound.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic2(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_2=None, name="BarmanBDIGemini3FlashHeuristic2"):
        super().__init__(name=name)
        self.fact_map = {}  # (predicate, args_tuple) -> global_id
        self.task_info = {} # global_id -> (task_type, args_list)
        self.cocktail_parts = defaultdict(list) # cocktail -> [ing1, ing2]
        self.all_shakers = []
        self.abstract_task_ids = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Map Facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self.fact_map[(pred, tuple(args))] = f.global_id
                # Identify shakers from facts
                if pred in ['clean', 'empty', 'shakerLevel'] and 'shaker' in args[0]:
                    if args[0] not in self.all_shakers:
                        self.all_shakers.append(args[0])
                # Map cocktail ingredients
                if pred in ['cocktailPart1', 'cocktailPart2']:
                    self.cocktail_parts[args[0]].append(args[1])

        # 2. Map Tasks
        for ab_task in model.abstract_tasks:
            self.abstract_task_ids.add(ab_task.global_id)
            self.task_info[ab_task.global_id] = self._parse_task_name(ab_task.name)
            
        for op in model.operators:
            self.task_info[op.global_id] = self._parse_task_name(op.name)

        h_initial = self._compute_heuristic(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

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
        idx = name.find("[")
        if idx > 0 and name.endswith("]"):
            return name[:idx], name[idx+1:-1].split(",")
        return name, []

    def is_true(self, state, pred, *args):
        f_id = self.fact_map.get((pred, tuple(args)))
        return (state >> f_id) & 1 if f_id is not None else 0

    def _shaker_cocktail_cost(self, cocktail, state):
        """Min actions to get a cocktail into ANY shaker and shaked."""
        best_cost = 999
        for s in self.all_shakers:
            if self.is_true(state, 'contains', s, cocktail):
                cost = 0 if self.is_true(state, 'shaked', s) else 1
            else:
                # Base cost: 1 (shake)
                cost = 1
                # Add ingredients costs
                for ing in self.cocktail_parts.get(cocktail, []):
                    if not self.is_true(state, 'contains', s, ing):
                        cost += 2 # fill-shot + pour-shot
                # Shaker maintenance
                if not self.is_true(state, 'unshaked', s) and not self.is_true(state, 'empty', s):
                    cost += 1 # empty-shaker
                if not self.is_true(state, 'clean', s) and self.is_true(state, 'empty', s):
                    cost += 1 # clean-shaker
            if cost < best_cost: best_cost = cost
        return best_cost if best_cost != 999 else 5

    def _estimate_task_cost(self, t_id, state):
        info = self.task_info.get(t_id)
        if not info: return 1
        t_type, args = info
        
        if t_type == 'AchieveContainsShotCocktail':
            shot, cocktail = args
            if self.is_true(state, 'contains', shot, cocktail): return 0
            cost = 1 # pour
            if not self.is_true(state, 'clean', shot): cost += 1
            cost += self._shaker_cocktail_cost(cocktail, state)
            return cost
            
        if t_type == 'AchieveContainsShakerCocktail':
            _, cocktail = args
            return self._shaker_cocktail_cost(cocktail, state)
            
        if t_type == 'AchieveContainsShakerIngredient':
            shaker, ing = args
            if self.is_true(state, 'contains', shaker, ing): return 0
            return 2 # fill + pour

        if t_type == 'AchieveCleanShot':
            shot = args[0]
            if self.is_true(state, 'clean', shot): return 0
            return 2 if not self.is_true(state, 'empty', shot) else 1

        if t_type == 'AchieveCleanShaker':
            shaker = args[0]
            if self.is_true(state, 'clean', shaker): return 0
            return 2 if not self.is_true(state, 'empty', shaker) else 1
            
        if t_type == 'AchieveHolding':
            hand, container = args
            return 0 if self.is_true(state, 'holding', hand, container) else 1
            
        if t_type == 'AchieveHandEmpty':
            hand = args[0]
            return 0 if self.is_true(state, 'handEmpty', hand) else 1

        return 1

    def _compute_heuristic(self, state, task_network):
        h = 0
        for task in task_network:
            if task.global_id in self.abstract_task_ids:
                h += self._estimate_task_cost(task.global_id, state)
            else:
                h += getattr(task, 'cost', 1)
        # Tie-breaker: prefer shorter task networks and empty hands
        h += len(task_network) * 0.001
        return int(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute_heuristic(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
