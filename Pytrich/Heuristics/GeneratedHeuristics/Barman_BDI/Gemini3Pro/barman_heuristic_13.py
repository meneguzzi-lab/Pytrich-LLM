import sys
from pytrich.heuristic import Heuristic

class BarmanGemini3ProHeuristic13(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_8=None):
        super().__init__(model)
        self.barman_gemini3pro_heuristic_8 = barman_gemini3pro_heuristic_8
        self.fact_id_map = {}
        self.recipes = {}  # cocktail -> set(ingredients)
        self.shakers = []
        self.shots = []
        
        self._initialize_domain_knowledge()

    def _initialize_domain_knowledge(self):
        """
        Pre-process facts to build ID lookups and discover static domain information
        like recipes and object types.
        """
        potential_shots = set()
        potential_shakers = set()

        for fact in self.model.facts:
            # Store map for fast bitwise checks
            self.fact_id_map[fact.name] = fact.global_id
            
            # Parse fact name to extract static info
            # Format: +predicate[arg1,arg2] or -predicate[...]
            raw_name = fact.name
            is_pos = not raw_name.startswith("-")
            name = raw_name[1:] if raw_name.startswith("+") or raw_name.startswith("-") else raw_name
            
            if "[" in name and name.endswith("]"):
                pred_part, args_part = name.split("[", 1)
                args = args_part[:-1].split(",")
                
                # Extract Recipes: cocktailPart1 and cocktailPart2 are static
                if is_pos and ("cocktailPart1" == pred_part or "cocktailPart2" == pred_part):
                    if len(args) == 2:
                        cocktail = args[0]
                        ingredient = args[1]
                        if cocktail not in self.recipes:
                            self.recipes[cocktail] = set()
                        self.recipes[cocktail].add(ingredient)
                
                # Identify Shakers via 'shakerLevel' predicate
                if pred_part == "shakerLevel":
                    potential_shakers.add(args[0])
                
                # Identify containers (shots and shakers) via 'clean'
                if pred_part == "clean":
                    potential_shots.add(args[0])

        self.shakers = list(potential_shakers)
        # Shots are containers that are not shakers
        self.shots = [s for s in potential_shots if s not in potential_shakers]

    def _get_fid(self, pred, args):
        # Helper to construct grounded fact name and get ID
        name = f"+{pred}[{','.join(args)}]"
        return self.fact_id_map.get(name, -1)

    def _is_true(self, state, pred, args):
        # Helper for fast bitwise state checks
        fid = self._get_fid(pred, args)
        if fid == -1:
            return False
        return (state >> fid) & 1

    def h(self, node):
        """
        Heuristic estimates the cost to complete the remaining task network.
        It sums the estimated cost of each abstract task based on the current state.
        """
        state = node.state
        tn = node.task_network
        
        # If task network is empty, we assume success in HTN (goals are usually tasks)
        if not tn:
            return 0
            
        cost = 0
        
        for task in tn:
            # Primitive Operator: Cost 1
            if hasattr(task, 'add_effects'):
                cost += 1
                continue
            
            # Abstract Task Analysis
            # task.name format: "TaskName arg1 arg2 ..."
            parts = task.name.split()
            if not parts: continue
            
            task_type = parts[0]
            args = parts[1:]
            
            if task_type == "AchieveContainsShotCocktail":
                cost += self._h_achieve_shot_cocktail(state, args)
            elif task_type == "AchieveCleanShot":
                cost += self._h_achieve_clean_shot(state, args)
            elif task_type == "AchieveCleanShaker":
                cost += self._h_achieve_clean_shaker(state, args)
            elif task_type == "AchieveContainsShakerIngredient":
                cost += self._h_achieve_shaker_ingredient(state, args)
            elif task_type == "AchieveContainsShotIngredient":
                cost += self._h_achieve_shot_ingredient(state, args)
            elif task_type == "DoPourShakerToShot":
                 # Requires holding shaker, pouring. Approx 1 action if ready.
                 cost += 1
            elif task_type == "MakeCocktail":
                 # Estimate remaining work to make cocktail
                 # Args: shaker, cocktail, ing1, hand, hand, ing2
                 if len(args) >= 2:
                     cost += self._h_make_cocktail_in_shaker(state, args[0], args[1])
            else:
                # Default small cost for administrative tasks (holding, etc.)
                cost += 1
                
        return cost

    def _h_achieve_clean_shot(self, state, args):
        if not args: return 0
        shot = args[0]
        if self._is_true(state, "clean", [shot]):
            return 0
        # If dirty, typically requires: grasp + clean (+ optional empty if full)
        # Heuristic estimate: 2 actions
        return 2

    def _h_achieve_clean_shaker(self, state, args):
        if not args: return 0
        shaker = args[0]
        if self._is_true(state, "clean", [shaker]):
            return 0
        # If dirty: grasp + empty + clean ~ 2-3 actions
        return 2

    def _h_achieve_shot_ingredient(self, state, args):
        # args: shot, ingredient
        if len(args) < 2: return 0
        shot, ing = args[0], args[1]
        if self._is_true(state, "contains", [shot, ing]):
            return 0
        
        c = 0
        # Needs clean shot first?
        if not self._is_true(state, "clean", [shot]):
            c += 2
        # Dispense/Fill action
        c += 2 # grasp + fill
        return c

    def _h_achieve_shaker_ingredient(self, state, args):
        # args: shaker, ingredient
        if len(args) < 2: return 0
        shaker, ing = args[0], args[1]
        if self._is_true(state, "contains", [shaker, ing]):
            return 0
        # Complex chain: clean shot -> fill shot -> pour to shaker
        # Approx 4 actions
        return 4 

    def _h_make_cocktail_in_shaker(self, state, shaker, cocktail):
        """Estimate cost to make a cocktail in a specific shaker"""
        # If already done
        if self._is_true(state, "contains", [shaker, cocktail]):
            # If unshaked (rare for final cocktail state), need shake
            if self._is_true(state, "unshaked", [shaker]):
                return 1
            return 0
            
        cost = 0
        needed_ingredients = self.recipes.get(cocktail, set())
        
        # Check cleanliness/content of shaker
        is_empty = self._is_true(state, "empty", [shaker])
        is_clean = self._is_true(state, "clean", [shaker])
        
        # Count existing ingredients
        existing_parts = 0
        for ing in needed_ingredients:
            if self._is_true(state, "contains", [shaker, ing]):
                existing_parts += 1
        
        # Cleaning penalty
        # If not empty and has 0 parts -> dirty with something else -> clean
        if not is_empty and existing_parts == 0:
            cost += 3 # empty, clean
        elif is_empty and not is_clean:
            cost += 2 # clean
        
        # Add missing ingredients
        # Cost per ingredient: dispense -> shot -> shaker ~ 4 steps
        missing_count = len(needed_ingredients) - existing_parts
        cost += missing_count * 4
        
        # Shake action
        cost += 1
        
        return cost

    def _h_achieve_shot_cocktail(self, state, args):
        # args: shot, cocktail
        if len(args) < 2: return 0
        shot, cocktail = args[0], args[1]
        
        # 1. Goal satisfied?
        if self._is_true(state, "contains", [shot, cocktail]):
            return 0
            
        total_cost = 0
        
        # 2. Preparation: Shot needs to be clean
        if not self._is_true(state, "clean", [shot]):
            total_cost += 2
            
        # 3. Source: Get cocktail ready in a shaker
        # We can use any shaker. Choose the one that minimizes cost.
        best_shaker_cost = float('inf')
        
        if not self.shakers:
            return total_cost + 10 # Fallback

        for shaker in self.shakers:
            # Estimate cost to make cocktail in this specific shaker
            s_cost = self._h_make_cocktail_in_shaker(state, shaker, cocktail)
            
            # Add Pour cost (1 action)
            s_cost += 1
            
            if s_cost < best_shaker_cost:
                best_shaker_cost = s_cost
                
        total_cost += best_shaker_cost
        return total_cost
