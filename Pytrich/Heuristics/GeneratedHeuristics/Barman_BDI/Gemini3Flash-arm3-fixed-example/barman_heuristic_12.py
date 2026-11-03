import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Barman_BDI domain.

1. Bottleneck Analysis:
   The primary bottleneck is the Shaker and Hand availability. Each cocktail requires a strict 
   linear sequence of actions: cleaning a shaker, fetching two ingredients (requiring shot cleaning/grasping), 
   shaking (requiring two hands or specific states), and pouring. Since there is usually only one shaker, 
   all cocktail goals are effectively serialized through this resource.

2. Admissible Lower Bounds:
   - Cocktail Cost: For every cocktail goal not in its target shot, we must perform at least:
     1 pour (1) + 1 shake (1) + 2 ingredient fills (2) + 2 ingredient pours (2) = 6 actions.
     If parts are already in the shaker, these costs are subtracted.
   - Task-Based Summation: In HTN, the task network represents the remaining work. 
     Assigning a state-sensitive lower bound to each task in the network provides high guidance.

3. Symmetry Breaking:
   - Added sub-unit penalties based on the index of containers to favor canonical ordering.
"""

class BarmanBDIGemini3FlashHeuristic12(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_12=None, name="BarmanBDIGemini3FlashHeuristic12"):
        super().__init__(name=name)
        self.fact_ids = defaultdict(dict)
        self.cocktail_parts = defaultdict(list)
        self.shakers = []
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        # Preprocess Fact IDs and Static Recipes
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            self.fact_ids[pred][tuple(args)] = f.global_id
            
            # Identify shakers
            if pred == "shakerLevel" and args[0] not in self.shakers:
                self.shakers.append(args[0])
                
            # Extract Recipe (cocktailPart1/2 are usually true in init state)
            if pred in ["cocktailPart1", "cocktailPart2"]:
                if (initial_node.state >> f.global_id) & 1:
                    if args[1] not in self.cocktail_parts[args[0]]:
                        self.cocktail_parts[args[0]].append(args[1])

        self.preprocessing_time = time.time() - start_time
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

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
        name = name.strip().replace("(", "").replace(")", "")
        parts = name.split()
        if not parts:
            return "", []
        return parts[0], parts[1:]

    def _is_true(self, state: int, predicate: str, args: tuple):
        fact_id = self.fact_ids.get(predicate, {}).get(tuple(args))
        if fact_id is None:
            return False
        return (state >> fact_id) & 1

    def _estimate_cocktail_cost(self, shot, cocktail, state):
        if self._is_true(state, "contains", (shot, cocktail)):
            return 0
        
        # Minimum: Pour from shaker to shot
        cost = 1
        
        # Shaker Prep cost
        best_shaker_cost = 10
        for shaker in self.shakers:
            if self._is_true(state, "contains", (shaker, cocktail)):
                best_shaker_cost = 0
                break
            
            s_prep = 1 # shake
            parts = self.cocktail_parts.get(cocktail, [])
            for p in parts:
                if not self._is_true(state, "contains", (shaker, p)):
                    s_prep += 2 # fill-shot + pour-to-shaker
            
            if not self._is_true(state, "clean", (shaker)) and s_prep > 1:
                s_prep += 1
            
            if s_prep < best_shaker_cost:
                best_shaker_cost = s_prep
        
        cost += best_shaker_cost

        # Container state costs
        if not self._is_true(state, "clean", (shot)):
            cost += 1
            
        # Symmetry breaking tie-breaker
        if shot.startswith("shot"):
            try:
                cost += int(shot[4:]) * 0.0001
            except:
                pass
                
        return cost

    def _compute(self, state: int, task_network):
        if not task_network:
            # Handle potential goal state checks (though rare in HTN execution)
            h = 0
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                pred, args = self._parse_fact_name(goal_name)
                if pred == "contains" and len(args) == 2:
                    h += self._estimate_cocktail_cost(args[0], args[1], state)
            return h

        total_h = 0
        for task in task_network:
            t_name, t_args = self._parse_task_name(task.name)
            if t_name == "AchieveContainsShotCocktail":
                if len(t_args) >= 2:
                    total_h += self._estimate_cocktail_cost(t_args[0], t_args[1], state)
            elif t_name == "AchieveContainsShakerCocktail":
                if len(t_args) >= 2:
                    # Heuristic for subtask: cost to get cocktail in shaker
                    s_prep = 1
                    parts = self.cocktail_parts.get(t_args[1], [])
                    for p in parts:
                        if not self._is_true(state, "contains", (t_args[0], p)):
                            total_h += 2
                    total_h += s_prep
            elif t_name == "AchieveContainsShakerIngredient":
                if len(t_args) >= 2:
                    if not self._is_true(state, "contains", (t_args[0], t_args[1])):
                        total_h += 2
            elif t_name in ["AchieveCleanShot", "AchieveCleanShaker"]:
                if len(t_args) >= 1:
                    if not self._is_true(state, "clean", (t_args[0])):
                        total_h += 1
            elif t_name == "AchieveHolding":
                if len(t_args) >= 2:
                    if not self._is_true(state, "holding", (t_args[0], t_args[1])):
                        total_h += 1
            else:
                # Generic cost for operators and other tasks
                total_h += 1

        return int(total_h)
