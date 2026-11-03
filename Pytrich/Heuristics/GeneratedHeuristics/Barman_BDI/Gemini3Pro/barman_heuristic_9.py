import sys

from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGemini3ProHeuristic9(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_4):
        super().__init__(barman_gemini3pro_heuristic_4)
        self.model = barman_gemini3pro_heuristic_4
        self.facts_by_predicate = {}
        self.cocktail_recipes = {}
        self.shakers = set()
        self.shots = set()
        self.ingredients = set()
        
        self._initialize_indices()
        
    def _initialize_indices(self):
        """
        Parses facts to build lookup tables and discover static domain information
        like cocktail recipes and object types.
        """
        for fact in self.model.facts:
            name = fact.name
            # Parse grounded fact name: +predicate[arg1,arg2]
            if name.startswith("+"):
                core = name[1:]
                if "[" in core and core.endswith("]"):
                    pred, args_part = core.split("[", 1)
                    args = tuple(args_part[:-1].split(",")) if args_part[:-1] else ()
                    
                    # Store global_id for fast state checking
                    if pred not in self.facts_by_predicate:
                        self.facts_by_predicate[pred] = {}
                    self.facts_by_predicate[pred][args] = fact.global_id
                    
                    # Domain specific object discovery
                    if pred == "shakerLevel":
                        self.shakers.add(args[0])
                    elif pred == "dispenses":
                        self.ingredients.add(args[1])
                    elif pred == "cocktailPart1":
                        cocktail, ing = args[0], args[1]
                        if cocktail not in self.cocktail_recipes:
                            self.cocktail_recipes[cocktail] = set()
                        self.cocktail_recipes[cocktail].add(ing)
                    elif pred == "cocktailPart2":
                        cocktail, ing = args[0], args[1]
                        if cocktail not in self.cocktail_recipes:
                            self.cocktail_recipes[cocktail] = set()
                        self.cocktail_recipes[cocktail].add(ing)

        # Infer shots: containers that are not shakers
        if "clean" in self.facts_by_predicate:
            for args in self.facts_by_predicate["clean"]:
                obj = args[0]
                if obj not in self.shakers:
                    self.shots.add(obj)

    def _is_true(self, state, pred, args):
        """Fast bitwise check if a grounded predicate is true in the state."""
        if pred in self.facts_by_predicate and args in self.facts_by_predicate[pred]:
            fid = self.facts_by_predicate[pred][args]
            return (state >> fid) & 1
        return 0

    def compute(self, state, task_network):
        """
        Estimates the cost to complete the task network from the current state.
        Focuses on the high-level barman tasks and approximates the cost of 
        cleaning, filling, shaking, and pouring.
        """
        h_val = 0
        
        for task in task_network:
            fullname = task.name
            tokens = fullname.split()
            task_type = tokens[0]
            args = tokens[1:]
            
            if task_type == "AchieveContainsShotCocktail":
                if len(args) >= 2:
                    h_val += self._h_achieve_shot_cocktail(state, args[0], args[1])
            
            elif task_type == "AchieveContainsShakerCocktail":
                if len(args) >= 2:
                    h_val += self._h_achieve_shaker_cocktail(state, args[0], args[1])
            
            elif task_type == "AchieveCleanShot":
                if len(args) >= 1:
                    h_val += self._h_achieve_clean_shot(state, args[0])
            
            elif task_type == "AchieveCleanShaker":
                if len(args) >= 1:
                    h_val += self._h_achieve_clean_shaker(state, args[0])
                    
            elif task_type == "AchieveContainsShakerIngredient":
                 if len(args) >= 2:
                     h_val += self._h_add_ing_shaker(state, args[0], args[1])
                     
            elif task_type == "AchieveContainsShotIngredient":
                if len(args) >= 2:
                    h_val += self._h_add_ing_shot(state, args[0], args[1])
            
            elif task_type == "DoPourShakerToShot":
                # This is a method subtask that leads to pour-shaker-to-shot
                h_val += 1
                
            elif task_type == "AchieveHolding":
                if len(args) >= 2:
                    hand, obj = args[0], args[1]
                    if not self._is_true(state, "holding", (hand, obj)):
                        h_val += 1
            
            elif task_type == "AchieveHandEmpty":
                if len(args) >= 1:
                    hand = args[0]
                    if not self._is_true(state, "handEmpty", (hand,)):
                        h_val += 1
            
            else:
                # Primitive action or unknown/generic task
                if hasattr(task, 'cost'):
                    h_val += getattr(task, 'cost', 1)
                else:
                    h_val += 1
                    
        return h_val

    def _h_achieve_shot_cocktail(self, state, shot, cocktail):
        # Goal: shot contains cocktail
        if self._is_true(state, "contains", (shot, cocktail)):
            return 0
            
        cost = 0
        
        # 1. Shot readiness
        is_empty = self._is_true(state, "empty", (shot,))
        is_clean = self._is_true(state, "clean", (shot,))
        
        if not is_empty:
             # Contains something else (otherwise first check would catch it)
             cost += 2 # Empty + Clean
        elif not is_clean:
             cost += 1 # Clean
             
        # 2. Shaker preparation (Relaxation: Find best cost among all shakers)
        best_shaker_cost = 999
        found_shaker = False
        
        for shaker in self.shakers:
            found_shaker = True
            sc = self._h_achieve_shaker_cocktail(state, shaker, cocktail)
            if sc < best_shaker_cost:
                best_shaker_cost = sc
        
        if not found_shaker:
            best_shaker_cost = 5 # Default penalty if no shakers found
            
        cost += best_shaker_cost
        
        # 3. Pouring action
        cost += 1
        
        return cost

    def _h_achieve_shaker_cocktail(self, state, shaker, cocktail):
        has_cocktail = self._is_true(state, "contains", (shaker, cocktail))
        is_shaked = self._is_true(state, "shaked", (shaker,))
        
        # Best case: ready to pour
        if has_cocktail and is_shaked:
            return 0
        
        # Has cocktail but needs shaking
        if has_cocktail and not is_shaked:
            return 1
            
        # Needs preparation
        ingredients = self.cocktail_recipes.get(cocktail, set())
        if not ingredients:
            return 1 # Fallback for unknown recipe
            
        cost = 0
        
        # Check if shaker is contaminated
        is_empty = self._is_true(state, "empty", (shaker,))
        is_clean = self._is_true(state, "clean", (shaker,))
        is_shaked_state = self._is_true(state, "shaked", (shaker,))
        
        # Count existing valid ingredients
        valid_parts = 0
        for ing in ingredients:
            if self._is_true(state, "contains", (shaker, ing)):
                valid_parts += 1
        
        missing_parts = len(ingredients) - valid_parts
        
        # Cleaning Logic:
        # If it was shaked (contains a previous cocktail), must empty+clean.
        # If dirty (not clean, not empty) and has no valid parts, must clean.
        # Optimistic: if it has valid parts, assume it's not contaminated with others.
        if is_shaked_state:
            cost += 2 # Empty + Clean
        elif not is_empty and not is_clean and valid_parts == 0:
            cost += 2 # Empty + Clean (approx)
        elif is_empty and not is_clean:
            cost += 1 # Clean
            
        # Add ingredients
        # Cost per ingredient approx: Fill Shot (2) + Pour to Shaker (1) = 3
        cost += missing_parts * 3
        
        # Shake action
        cost += 1
        
        return cost

    def _h_add_ing_shaker(self, state, shaker, ing):
        if self._is_true(state, "contains", (shaker, ing)):
            return 0
        # Cycle: Grasp Shot, Fill Shot, Grasp Shaker, Pour
        return 3

    def _h_add_ing_shot(self, state, shot, ing):
        if self._is_true(state, "contains", (shot, ing)):
            return 0
        # Cycle: Clean Shot (if needed) + Fill
        if not self._is_true(state, "clean", (shot,)) and not self._is_true(state, "empty", (shot,)):
             return 3 # Empty + Clean + Fill
        if not self._is_true(state, "clean", (shot,)):
             return 2 # Clean + Fill
        return 1 # Fill
        
    def _h_achieve_clean_shot(self, state, shot):
        if self._is_true(state, "clean", (shot,))
            return 0
        if self._is_true(state, "empty", (shot,))
            return 1
        return 2

    def _h_achieve_clean_shaker(self, state, shaker):
        if self._is_true(state, "clean", (shaker,))
            return 0
        if self._is_true(state, "empty", (shaker,))
            return 1
        return 2