import sys
from pytrich.heuristic import Heuristic

class BarmanGemini3ProHeuristic19(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_14):
        super().__init__(barman_gemini3pro_heuristic_14)
        self.model = barman_gemini3pro_heuristic_14.model
        
        # Domain specific storage
        self.cocktail_recipes = {}  # cocktail_name -> set(ingredient_names)
        self.fact_map = {}          # predicate -> { args_tuple -> global_id }
        self.shakers = set()
        self.shots = set()
        self.ingredients = set()
        self.cocktails = set()
        
        self._parse_static_domain_info()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = tuple(args_str.split(",")) if args_str else ()
            return predicate, args
        return None, ()

    def _parse_static_domain_info(self):
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            
            # Store global_id for fast state lookup
            if pred not in self.fact_map:
                self.fact_map[pred] = {}
            self.fact_map[pred][args] = fact.global_id

            # Extract Objects and Static Relations
            if pred == "cocktailPart1":
                ct, ing = args
                if ct not in self.cocktail_recipes:
                    self.cocktail_recipes[ct] = set()
                self.cocktail_recipes[ct].add(ing)
                self.cocktails.add(ct)
                self.ingredients.add(ing)
            elif pred == "cocktailPart2":
                ct, ing = args
                if ct not in self.cocktail_recipes:
                    self.cocktail_recipes[ct] = set()
                self.cocktail_recipes[ct].add(ing)
                self.cocktails.add(ct)
                self.ingredients.add(ing)
            elif pred == "shakerLevel":
                self.shakers.add(args[0])
            elif pred == "clean" and "shot" in args[0]: # heuristic guess for type
                self.shots.add(args[0])
            elif pred == "ontable":
                # Fallback to identify objects if not found above
                obj = args[0]
                if "shaker" in obj:
                    self.shakers.add(obj)
                elif "shot" in obj:
                    self.shots.add(obj)

    def _is_true(self, state, pred, args):
        if pred not in self.fact_map or args not in self.fact_map[pred]:
            return False
        gid = self.fact_map[pred][args]
        return (state >> gid) & 1

    def h(self, node):
        state = node.state
        tn = node.task_network
        
        if not tn:
            return 0
            
        total_cost = 0
        for task in tn:
            if hasattr(task, 'name'): # AbstractTask or Operator
                total_cost += self._estimate_task_cost(task.name, state)
            else:
                # Should not happen in this framework usually, but safe fallback
                total_cost += 1
                
        return total_cost

    def _estimate_task_cost(self, task_name, state):
        # Parse task name to identify type and arguments
        # Assumed format: "TaskName arg1 arg2" or "TaskName[arg1,arg2]"
        # Normalizing:
        clean_name = task_name.replace("[", " ").replace("]", " ").replace(",", " ")
        parts = clean_name.split()
        if not parts:
            return 0
            
        task_type = parts[0]
        args = parts[1:]

        # --- High Level Tasks ---
        if task_type == "AchieveContainsShotCocktail":
            if len(args) >= 2:
                return self._cost_achieve_contains_shot_cocktail(state, args[0], args[1])
            return 10 # Fallback
            
        elif task_type == "AchieveContainsShakerCocktail":
            if len(args) >= 2:
                return self._cost_achieve_contains_shaker_cocktail(state, args[0], args[1])
            return 5
            
        elif task_type == "AchieveCleanShot":
            if len(args) >= 1:
                return self._cost_clean_shot(state, args[0])
            return 2
            
        elif task_type == "AchieveCleanShaker":
            if len(args) >= 1:
                return self._cost_clean_shaker(state, args[0])
            return 2

        elif task_type == "AchieveContainsShakerIngredient":
            if len(args) >= 2:
                return self._cost_add_ingredient(state, args[0], args[1])
            return 3
            
        # --- Primitive Actions (Operators) ---
        # Default small cost for primitives
        return 1

    def _cost_achieve_contains_shot_cocktail(self, state, shot, cocktail):
        # 0. Check if already done
        if self._is_true(state, "contains", (shot, cocktail)):
            return 0

        cost = 0
        
        # 1. Clean Shot Cost
        # If shot is not empty and not containing target, or dirty, needs cleaning
        shot_clean = self._is_true(state, "clean", (shot,))
        shot_empty = self._is_true(state, "empty", (shot,))
        
        # If we need to pour into it, it effectively needs to be clean/empty
        # We approximate: if not clean or not empty, pay cleaning cost
        if not (shot_clean and shot_empty):
             # Optimization: If it just contains the cocktail, we returned 0 above.
             # So here it implies it has something else or is dirty.
             cost += 3 # Grasp, Clean/Empty, Drop

        # 2. Get Cocktail in Shaker
        # We need to find the best shaker or use a generic estimate
        # The task doesn't specify shaker, but the decomposition will choose one.
        # We scan all shakers and take the minimum cost.
        best_shaker_cost = float('inf')
        
        for shaker in self.shakers:
            s_cost = 0
            
            # Check shaker state
            has_cocktail = self._is_true(state, "contains", (shaker, cocktail))
            is_shaked = self._is_true(state, "shaked", (shaker,))
            
            if has_cocktail and is_shaked:
                # Just pour
                s_cost = 1 # Pour
            elif has_cocktail and not is_shaked:
                # Shake + Pour
                s_cost = 2
            else:
                # Need to make cocktail in this shaker
                # Make cost = Clean Shaker (if needed) + Add Ingredients + Shake + Pour
                make_cost = 0
                
                # Check cleanliness
                shaker_clean = self._is_true(state, "clean", (shaker,))
                shaker_empty = self._is_true(state, "empty", (shaker,))
                
                # If dirty/used, we must clean it unless it contains parts of our cocktail
                # For heuristic simplicity, if not empty and not containing our parts, assume clean needed
                # We check ingredients
                required_ings = self.cocktail_recipes.get(cocktail, set())
                
                # Check what's currently inside
                # This is hard without iterating all facts or checking all ingredients.
                # We'll use a simpler proxy: if not empty and not clean, add cleaning cost.
                # A more accurate check would see if the contents are a subset of required ingredients.
                
                # Let's count missing ingredients
                missing_count = 0
                present_count = 0
                for ing in required_ings:
                    if self._is_true(state, "contains", (shaker, ing)):
                        present_count += 1
                    else:
                        missing_count += 1
                
                if not shaker_empty and not shaker_clean and present_count == 0:
                    make_cost += 3 # Clean shaker sequence
                
                # Cost to add missing ingredients
                # Sequence: Fill Shot (1) + Grasp Shot (1) + Pour to Shaker (1) = 3 per ingredient
                # Plus potentially cleaning the shot used for transfer (ignored for lower bound/simplicity)
                make_cost += missing_count * 3
                
                make_cost += 1 # Shake
                make_cost += 1 # Pour to destination shot
                s_cost = make_cost
                
            if s_cost < best_shaker_cost:
                best_shaker_cost = s_cost
        
        if best_shaker_cost == float('inf'):
             # Fallback if no shakers found (shouldn't happen)
             best_shaker_cost = 10
             
        return cost + best_shaker_cost

    def _cost_achieve_contains_shaker_cocktail(self, state, shaker, cocktail):
        if self._is_true(state, "contains", (shaker, cocktail)) and self._is_true(state, "shaked", (shaker,)):
            return 0
            
        cost = 0
        # Ingredients
        required_ings = self.cocktail_recipes.get(cocktail, set())
        missing_count = 0
        for ing in required_ings:
            if not self._is_true(state, "contains", (shaker, ing)):
                missing_count += 1
        
        # If not clean/empty and no correct ingredients, add cleaning
        shaker_empty = self._is_true(state, "empty", (shaker,))
        shaker_clean = self._is_true(state, "clean", (shaker,))
        if not shaker_empty and not shaker_clean and missing_count == len(required_ings):
            cost += 3
            
        cost += missing_count * 3 # Add ingredients
        cost += 1 # Shake
        return cost

    def _cost_clean_shot(self, state, shot):
        if self._is_true(state, "clean", (shot,)) and self._is_true(state, "empty", (shot,)):
            return 0
        # Approx sequence: Grasp, Empty/Clean, Drop
        return 3
        
    def _cost_clean_shaker(self, state, shaker):
        if self._is_true(state, "clean", (shaker,)) and self._is_true(state, "empty", (shaker,)):
            return 0
        return 3
        
    def _cost_add_ingredient(self, state, shaker, ingredient):
        if self._is_true(state, "contains", (shaker, ingredient)):
            return 0
        # Need to transfer via shot: Fill shot, Grasp, Pour
        return 3
