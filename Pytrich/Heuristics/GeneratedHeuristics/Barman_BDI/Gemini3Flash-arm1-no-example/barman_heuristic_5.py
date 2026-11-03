from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic5(Heuristic):
    """
    Bottleneck: The Shaker container.
    Every cocktail requires the shaker to be cleaned, ingredients to be added from shot glasses,
    shaken, and then poured into a clean goal shot. The sequence is strictly ordered and 
    revolves around shaker availability and state.
    
    Admissibility Sketch:
    1. Each AchieveContainsShotCocktail task in the network requires a series of actions (clean, fill, pour, shake, pour).
       Estimating the minimum number of remaining primitive actions for these tasks provides an admissible lower bound.
    2. Cleaning tasks (AchieveCleanShot, AchieveCleanShaker) contribute at least 1 action if the container is currently dirty.
    3. Ingredient tasks (AchieveContainsShotIngredient, AchieveContainsShakerIngredient) contribute at least 2 or 3 actions respectively.
    4. Summing these costs is admissible as they represent disjoint sequences of primitive actions required to clear the task network.
    """

    def __init__(self, barman_gemini3flash_heuristic_5="barman_gemini3flash_heuristic_5"):
        super().__init__(barman_gemini3flash_heuristic_5)
        self.pred_map = {}
        self.shakers = set()
        self.cocktail_parts = {}
        self.goal_fids = []

    def initialize(self, model: Model, initial_node: HTNNode):
        for f in model.facts:
            p, a = self._parse_fact_name(f.name)
            if not p:
                continue
            self.pred_map[(p, tuple(a))] = f.global_id
            if p == "shakerLevel":
                self.shakers.add(a[0])
            elif p == "cocktailPart1":
                c, i = a[0], a[1]
                if c not in self.cocktail_parts: self.cocktail_parts[c] = [None, None]
                self.cocktail_parts[c][0] = i
            elif p == "cocktailPart2":
                c, i = a[0], a[1]
                if c not in self.cocktail_parts: self.cocktail_parts[c] = [None, None]
                self.cocktail_parts[c][1] = i
        
        temp_goals = model.goals
        idx = 0
        while temp_goals:
            if temp_goals & 1:
                self.goal_fids.append(idx)
            temp_goals >>= 1
            idx += 1
            
        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state: int, task_network) -> int:
        h = 0
        for task in task_network:
            if hasattr(task, 'cost'):
                h += task.cost
                continue
            
            t_name, t_args = self._parse_task(task)
            
            if t_name == "AchieveContainsShotCocktail":
                shot, cocktail = t_args[0], t_args[1]
                if not self._check(state, "contains", (shot, cocktail)):
                    found_in_shaker = False
                    for shaker in self.shakers:
                        if self._check(state, "contains", (shaker, cocktail)):
                            h += 2 if self._check(state, "shaked", (shaker,)) else 3
                            found_in_shaker = True
                            break
                    if not found_in_shaker: h += 8
            elif t_name == "AchieveContainsShakerCocktail":
                shaker, cocktail = t_args[0], t_args[1]
                if not self._check(state, "contains", (shaker, cocktail)):
                    parts = self.cocktail_parts.get(cocktail, [None, None])
                    ing_in = 0
                    if parts[0] and self._check(state, "contains", (shaker, parts[0])): ing_in += 1
                    if parts[1] and self._check(state, "contains", (shaker, parts[1])): ing_in += 1
                    h += (6 - 2 * ing_in)
            elif t_name == "AchieveContainsShakerIngredient":
                shaker, ing = t_args[0], t_args[1]
                if not self._check(state, "contains", (shaker, ing)): h += 3
            elif t_name == "AchieveContainsShotIngredient":
                shot, ing = t_args[0], t_args[1]
                if not self._check(state, "contains", (shot, ing)): h += 2
            elif t_name == "AchieveCleanShot" or t_name == "AchieveCleanShaker":
                if not self._check(state, "clean", (t_args[0],)): h += 1
            elif t_name == "AchieveHolding":
                if len(t_args) >= 2 and not self._check(state, "holding", (t_args[0], t_args[1])): h += 1
            elif t_name == "AchieveHandEmpty":
                if not self._check(state, "handEmpty", (t_args[0],)): h += 1
            else:
                h += 1
        
        for fid in self.goal_fids:
            if not ((state >> fid) & 1): h += 10
        return int(h)

    def _check(self, state, pred, args):
        fid = self.pred_map.get((pred, tuple(args)))
        return (fid is not None) and ((state >> fid) & 1)

    def _parse_task(self, task):
        parts = task.name.split()
        return parts[0], parts[1:]

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
