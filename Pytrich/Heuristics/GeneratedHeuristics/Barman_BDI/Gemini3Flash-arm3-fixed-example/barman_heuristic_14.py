from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class BarmanBDIGemini3FlashHeuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.

    1. Bottleneck: The Shaker. Making a cocktail is a sequential process: 
       Clean Shaker -> Add Ingredient 1 (via Shot) -> Add Ingredient 2 (via Shot) -> Shake -> Pour to Shot.
    
    2. Admissible Lower Bounds:
       - Cocktail Progress Cost: For each goal cocktail not in its target shot, we estimate the 
         minimum actions remaining (pour=1, shake+pour=2, fill+pour+shake+pour=4, etc.).
       - Task Network Depth: High-level tasks in the network represent a commitment to a 
         sequence of primitive actions (e.g., AchieveContainsShotCocktail is at least 6 actions).

    3. Implementation:
       - Precomputes fact IDs and object relationships in initialize().
       - Uses bitwise operations for state checks in __call__.
       - Returns the max of state-based cost and task-based cost.
    """

    def __init__(self, barman_gemini3flash_heuristic_14=None, name="BarmanBDIGemini3FlashHeuristic14"):
        super().__init__(name=name)
        self.goal_cocktails = {} # shot -> cocktail
        self.cocktail_ingredients = {} # cocktail -> [ing1, ing2]
        self.fact_ids = {} # (pred, tuple(args)) -> id
        self.shakers = []
        self.shots = []
        self.task_costs = {
            "AchieveContainsShotCocktail": 6,
            "AchieveContainsShakerCocktail": 4,
            "AchieveContainsShakerIngredient": 2,
            "AchieveContainsShotIngredient": 1,
            "AchieveCleanShot": 1,
            "AchieveCleanShaker": 1,
            "AchieveHolding": 1,
            "AchieveHandEmpty": 1,
            "AchieveOnTable": 1
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                args_tuple = tuple(args)
                self.fact_ids[(pred, args_tuple)] = f.global_id
                if pred == "cocktailPart1" or pred == "cocktailPart2":
                    c, i = args[0], args[1]
                    if c not in self.cocktail_ingredients: self.cocktail_ingredients[c] = []
                    self.cocktail_ingredients[c].append(i)
                elif pred == "shakerLevel":
                    if args[0] not in self.shakers: self.shakers.append(args[0])
                elif pred == "clean" and args[0].startswith("shot"):
                    if args[0] not in self.shots: self.shots.append(args[0])

        if hasattr(model, 'goals') and model.goals:
            goal_names = model.state_explicit_repr(model.goals)
            for gname in goal_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "contains" and len(args) == 2:
                    if args[0].startswith("shot"):
                        self.goal_cocktails[args[0]] = args[1]

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
        return None, []

    def _is_true(self, state, pred, args):
        fid = self.fact_ids.get((pred, tuple(args)))
        if fid is None: return False
        return (state >> fid) & 1

    def _compute(self, state, task_network):
        # 1. State-based goal check
        state_h = 0
        for shot, cocktail in self.goal_cocktails.items():
            if self._is_true(state, "contains", [shot, cocktail]):
                continue
            
            found_in_shaker = False
            for shaker in self.shakers:
                if self._is_true(state, "contains", [shaker, cocktail]):
                    if self._is_true(state, "shaked", [shaker]):
                        state_h += 1
                    else:
                        state_h += 2
                    found_in_shaker = True
                    break
            
            if not found_in_shaker:
                sub_h = 2 # shake + pour
                ings = self.cocktail_ingredients.get(cocktail, [])
                for ing in ings:
                    in_shaker = any(self._is_true(state, "contains", [sh, ing]) for sh in self.shakers)
                    if in_shaker: continue
                    in_shot = any(self._is_true(state, "contains", [sh, ing]) for sh in self.shots)
                    sub_h += 1 if in_shot else 2
                state_h += sub_h

        # 2. Task network check
        tn_h = 0
        for task in task_network:
            tname = task.name
            base_name = tname.split("[")[0] if "[" in tname else tname
            if base_name in self.task_costs:
                tn_h += self.task_costs[base_name]
            elif hasattr(task, 'cost'):
                tn_h += task.cost
            else:
                tn_h += 1

        return max(state_h, tn_h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
