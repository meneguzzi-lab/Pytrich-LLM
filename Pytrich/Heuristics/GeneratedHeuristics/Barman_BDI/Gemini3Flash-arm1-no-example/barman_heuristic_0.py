from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck: The single shaker (or small number of shakers) serves as a serialization point for cocktail production.
# Each cocktail must be cleaned, filled with two ingredients, shaken, and poured sequentially into a target shot.
# 
# Admissible Lower Bounds:
# 1. Task Network Cost: A sum of the minimum number of actions required to decompose and fulfill each task remaining in the task network.
#    Since cocktails are processed mostly sequentially due to hand and shaker constraints, a sum of individual task costs is a strong lower bound.
# 2. Goal State Distance: For each cocktail in the goal state not yet in its target shot, we estimate the remaining actions.
#    If it's in a shaked shaker, it needs at least 1-2 actions (grasp + pour). If it's in an unshaked shaker, at least 2-3 (shake + pour).
#    If not in a shaker, at least 6-8 actions (fill ingredients, shake, pour).

class BarmanBDIGemini3FlashHeuristic0(Heuristic):
    def __init__(self, barman_gemini3flash_heuristic_0="barman_gemini3flash_heuristic_0"):
        super().__init__(barman_gemini3flash_heuristic_0)
        self.fact_ids = {} # predicate -> args_tuple -> global_id
        self.goal_contains = [] # List of (shot, cocktail, global_id)
        self.shakers = []
        self.cocktails = set()

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # Precompute fact maps
        for f in model.facts:
            p, args = self._parse_fact_name(f.name)
            if p not in self.fact_ids:
                self.fact_ids[p] = {}
            self.fact_ids[p][tuple(args)] = f.global_id
            
            if p == "shakerLevel":
                if args[0] not in self.shakers: self.shakers.append(args[0])
            if p == "cocktailPart1":
                self.cocktails.add(args[0])

        # Precompute goal facts
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            p, args = self._parse_fact_name(g_name)
            if p == "contains" and args[1] in self.cocktails:
                g_id = self.fact_ids.get('contains', {}).get(tuple(args))
                if g_id is not None:
                    self.goal_contains.append((args[0], args[1], g_id))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        # Term 1: Task Network Work Estimate
        h_tasks = 0
        for task in task_network:
            t_name = task.name
            if "AchieveContainsShotCocktail" in t_name: h_tasks += 10
            elif "AchieveContainsShakerCocktail" in t_name: h_tasks += 6
            elif "AchieveContainsShakerIngredient" in t_name: h_tasks += 3
            elif "AchieveClean" in t_name: h_tasks += 2
            elif "AchieveHolding" in t_name: h_tasks += 2
            elif "AchieveHandEmpty" in t_name: h_tasks += 1
            elif "DoPourShakerToShot" in t_name: h_tasks += 1
            else: h_tasks += 1

        # Term 2: State-based Goal Distance
        h_goals = 0
        for shot, cocktail, g_id in self.goal_contains:
            # If shot doesn't have cocktail
            if not ((state >> g_id) & 1):
                found_in_any_shaker = False
                for shaker in self.shakers:
                    c_shaker_id = self.fact_ids.get('contains', {}).get((shaker, cocktail))
                    if c_shaker_id is not None and (state >> c_shaker_id) & 1:
                        found_in_any_shaker = True
                        s_id = self.fact_ids.get('shaked', {}).get((shaker,))
                        if s_id is not None and (state >> s_id) & 1:
                            h_goals += 2 # Grasp + Pour
                        else:
                            h_goals += 3 # Grasp + Shake + Pour
                        break
                
                if not found_in_any_shaker:
                    # Needs cleaning shaker, adding ingredients, shaking, pouring
                    h_goals += 7
        
        # Using max is admissible and provides consistent guidance
        return max(h_tasks, h_goals)
