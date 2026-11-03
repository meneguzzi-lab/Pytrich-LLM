from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.core.operator import Operator
from Pytrich.core.abstract_task import AbstractTask

class BarmanGemini3ProHeuristic8(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_3, model):
        super().__init__(barman_gemini3pro_heuristic_3, model)
        self.model = model
        self.fact_lookup = {}
        self.cocktail_recipes = {}  # cocktail -> {ing1, ing2}
        self.shakers = []
        
        self._initialize_mappings()

    def _initialize_mappings(self):
        # Parse all facts to build lookup table and extract domain objects/recipes
        for fact in self.model.facts:
            name = fact.name
            predicate, args = self._parse_fact_name(name)
            if predicate:
                self.fact_lookup[(predicate, tuple(args))] = fact.global_id
                
                # Extract cocktail recipes
                if predicate == "cocktailPart1":
                    c, i = args[0], args[1]
                    if c not in self.cocktail_recipes: self.cocktail_recipes[c] = set()
                    self.cocktail_recipes[c].add(i)
                elif predicate == "cocktailPart2":
                    c, i = args[0], args[1]
                    if c not in self.cocktail_recipes: self.cocktail_recipes[c] = set()
                    self.cocktail_recipes[c].add(i)
                
                # Identify shakers (only shakers have levels)
                if predicate == "shakerLevel":
                    if args[0] not in self.shakers: self.shakers.append(args[0])

    def _parse_fact_name(self, name: str):
        # Parses "+pred[arg1,arg2]" into ("pred", ["arg1", "arg2"])
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

    def get_fact_id(self, predicate, args):
        return self.fact_lookup.get((predicate, tuple(args)), -1)

    def is_true(self, state, predicate, args):
        fid = self.get_fact_id(predicate, args)
        if fid == -1: return False
        return (state >> fid) & 1

    def h(self, task_network, state):
        cost = 0
        
        # 1. State-based Goal Estimation (Admissibility Anchor)
        # Calculate unsatisfied state goals using bitwise operations
        # This ensures we account for global state requirements not fully covered by the current TN frontier
        goals_unsatisfied = (self.model.goals & ~state)
        if goals_unsatisfied:
            cost += bin(goals_unsatisfied).count('1') * 2

        # 2. Hierarchical Task Network Estimation
        for task in task_network:
            if isinstance(task, Operator):
                cost += 1
            elif isinstance(task, AbstractTask):
                # Parse task name format "TaskName arg1 arg2 ..."
                parts = task.name.split()
                task_name = parts[0]
                args = parts[1:]
                cost += self._estimate_abstract_task(task_name, args, state)
                
        return cost

    def _estimate_abstract_task(self, task_name, args, state):
        # Estimates the remaining effort for specific abstract tasks based on domain structure
        cost = 0
        
        if task_name == "AchieveContainsShotCocktail":
            # args: ?shot ?cocktail
            shot, cocktail = args[0], args[1]
            
            if self.is_true(state, "contains", [shot, cocktail]):
                return 0
            
            # Base cost: Clean Shaker(3) + Add Ing1(4) + Add Ing2(4) + Shake(1) + Clean Shot(3) + Pour(1) ~ 16
            # We use a conservative estimate and subtract based on progress
            cost = 14
            
            # Penalty if shot is dirty
            if not self.is_true(state, "clean", [shot]) and not self.is_true(state, "empty", [shot]):
                cost += 2 # Extra cost to empty first
                
            # Reward progress in shaker
            # We check if *any* shaker contains the cocktail or ingredients
            # Since the specific shaker isn't bound in the high-level task, we scan available shakers
            shaker_progress = 0
            ingredients = self.cocktail_recipes.get(cocktail, set())
            
            for shaker in self.shakers:
                if self.is_true(state, "contains", [shaker, cocktail]):
                    # Cocktail is ready in a shaker
                    # Remaining: Clean Shot (accounted above) + Pour(1)
                    # We subtract most of the preparation cost
                    shaker_progress = max(shaker_progress, 11)
                else:
                    # Check ingredient progress
                    ing_count = 0
                    for ing in ingredients:
                        if self.is_true(state, "contains", [shaker, ing]):
                            ing_count += 1
                    
                    if ing_count > 0:
                        # ~3 ops saved per ingredient (grasp, fill, pour)
                        shaker_progress = max(shaker_progress, ing_count * 3)
            
            cost -= shaker_progress

        elif task_name == "AchieveContainsShakerCocktail":
            # args: ?shaker ?cocktail
            shaker, cocktail = args[0], args[1]
            if self.is_true(state, "contains", [shaker, cocktail]):
                return 0
            
            # Base cost for mixing: Clean(3) + Ing1(4) + Ing2(4) + Shake(1) = 12
            cost = 12
            if not self.is_true(state, "clean", [shaker]):
                if not self.is_true(state, "empty", [shaker]):
                    cost += 2 # empty cost
            
            ingredients = self.cocktail_recipes.get(cocktail, set())
            ing_count = 0
            for ing in ingredients:
                if self.is_true(state, "contains", [shaker, ing]):
                    ing_count += 1
            cost -= (ing_count * 3)

        elif task_name == "AchieveCleanShot":
            # args: ?shot
            shot = args[0]
            if not self.is_true(state, "clean", [shot]):
                cost = 3
                if not self.is_true(state, "empty", [shot]):
                    cost += 1

        elif task_name == "AchieveCleanShaker":
            # args: ?shaker
            shaker = args[0]
            if not self.is_true(state, "clean", [shaker]):
                cost = 3
                if not self.is_true(state, "empty", [shaker]):
                    cost += 1

        elif task_name == "AchieveContainsShakerIngredient":
            # args: ?shaker ?ingredient
            shaker, ing = args[0], args[1]
            if not self.is_true(state, "contains", [shaker, ing]):
                cost = 4 # Grasp shot, fill, pour, drop

        elif task_name == "AchieveHandEmpty":
            hand = args[0]
            if not self.is_true(state, "handEmpty", [hand]):
                cost = 1

        elif task_name == "AchieveHolding":
            hand, obj = args[0], args[1]
            if not self.is_true(state, "holding", [hand, obj]):
                cost = 1

        elif task_name == "AchieveOnTable":
            obj = args[0]
            if not self.is_true(state, "ontable", [obj]):
                cost = 1
        
        elif task_name == "DoPourShakerToShot":
             # Wraps a single action
             cost = 1

        else:
            # Default fallback for minor helper tasks
            cost = 1

        return max(0, cost)
