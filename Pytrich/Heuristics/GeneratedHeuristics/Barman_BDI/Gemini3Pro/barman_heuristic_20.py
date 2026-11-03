from Pytrich.heuristics.heuristic import Heuristic
import sys

class BarmanGemini3ProHeuristic20(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_15):
        super().__init__(barman_gemini3pro_heuristic_15)
        self.model = barman_gemini3pro_heuristic_15.model
        
        # Precompute lookups for fast state evaluation
        self.fact_id_map = {} 
        self.recipes = {} # cocktail -> list of ingredients
        self.shakers = []
        
        # Parse static facts and objects from the model
        self._parse_static_info()

    def _parse_static_info(self):
        """
        Parses model facts to build:
        1. A mapping from (predicate, args) to global_id for O(1) state checks.
        2. Cocktail recipes (static cocktailPart facts).
        3. A list of available shakers.
        """
        for fact in self.model.facts:
            name = fact.name
            # Expected format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
            if name.startswith("+"):
                content = name[1:]
                # Simple parser for predicate[args]
                if "[" in content and content.endswith("]"):
                    pred, args_str = content.split("[", 1)
                    args_str = args_str[:-1]
                    args = tuple(args_str.split(",")) if args_str else ()
                    
                    # Store in map
                    if pred not in self.fact_id_map:
                        self.fact_id_map[pred] = {}
                    self.fact_id_map[pred][args] = fact.global_id
                    
                    # Extract Domain Knowledge
                    if pred == "cocktailPart1":
                        c, i = args
                        if c not in self.recipes: self.recipes[c] = []
                        self.recipes[c].append(i)
                    elif pred == "cocktailPart2":
                        c, i = args
                        if c not in self.recipes: self.recipes[c] = []
                        self.recipes[c].append(i)
                    elif pred == "shakerLevel":
                        # If a shaker has a level, it is a shaker object
                        if args[0] not in self.shakers:
                            self.shakers.append(args[0])
        
        # Fallback if no shakers found (e.g. initial state has no level facts)
        if not self.shakers:
            self.shakers = ["shaker1"] # Default assumption for standard instances

    def _get_fact_id(self, pred, args):
        if pred in self.fact_id_map and args in self.fact_id_map[pred]:
            return self.fact_id_map[pred][args]
        return -1

    def _is_true(self, state, pred, args):
        """Checks if a grounded predicate is true in the bitwise state."""
        fid = self._get_fact_id(pred, args)
        if fid != -1:
            return (state >> fid) & 1
        return False

    def h(self, node):
        state = node.state
        tn = node.tasks
        
        total_cost = 0
        
        for task in tn:
            # 1. Primitive Tasks (Operators)
            if hasattr(task, 'cost'):
                total_cost += task.cost
                continue
                
            # 2. Abstract Tasks
            # Task names are expected to be strings like "TaskName arg1 arg2"
            name_parts = task.name.split()
            if not name_parts:
                continue
                
            task_type = name_parts[0]
            args = name_parts[1:]
            
            if task_type == "AchieveContainsShotCocktail":
                if len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    total_cost += self._cost_shot_cocktail(state, shot, cocktail)
                else:
                    total_cost += 5 # Fallback
            
            elif task_type == "AchieveCleanShot":
                if len(args) >= 1:
                    total_cost += self._cost_clean_shot(state, args[0])
                else:
                    total_cost += 1
                    
            elif task_type == "AchieveCleanShaker":
                if len(args) >= 1:
                    total_cost += self._cost_clean_shaker(state, args[0])
                else:
                    total_cost += 1
            
            elif task_type == "AchieveContainsShakerCocktail":
                if len(args) >= 2:
                    shaker, cocktail = args[0], args[1]
                    total_cost += self._cost_shaker_cocktail(state, shaker, cocktail)
                else:
                    total_cost += 5
            
            else:
                # Generic cost for other tasks (e.g. navigation, helper tasks)
                total_cost += 1
                
        return total_cost

    def _cost_shot_cocktail(self, state, shot, cocktail):
        """Estimate cost to get cocktail into shot."""
        if self._is_true(state, "contains", (shot, cocktail)):
            return 0
            
        cost = 0
        # Action: Pour shaker to shot
        cost += 1
        
        # Shot Cleanliness
        if not self._is_true(state, "clean", (shot,)):
             # If not clean, we likely need to clean it. 
             # If it's empty but used, cost 1. If full, cost 2.
             cost += 2 
        
        # We need the cocktail prepared in a shaker.
        # Find the minimum cost across all available shakers
        shaker_costs = []
        for shaker in self.shakers:
            sc = self._cost_shaker_cocktail(state, shaker, cocktail)
            shaker_costs.append(sc)
            
        if shaker_costs:
            cost += min(shaker_costs)
        else:
            cost += 10 # Penalty if no shaker found
            
        return cost

    def _cost_shaker_cocktail(self, state, shaker, cocktail):
        """Estimate cost to prepare cocktail in shaker."""
        has_cocktail = self._is_true(state, "contains", (shaker, cocktail))
        is_shaked = self._is_true(state, "shaked", (shaker,))
        
        # Best case: ready to pour
        if has_cocktail and is_shaked:
            return 0
        
        cost = 0
        
        # Case: Contains unmixed cocktail
        if has_cocktail and not is_shaked:
            return 1 # Shake
            
        # Case: Need to mix
        # 1. Action: Shake
        cost += 1 
        
        # 2. Check Ingredients
        needed = self.recipes.get(cocktail, [])
        ingredients_needed_count = 0
        
        # Check if shaker needs cleaning first
        # Simplification: If shaker is not empty and doesn't contain needed parts, it needs cleaning
        needs_cleaning = False
        if not self._is_true(state, "empty", (shaker,)):
            # Does it contain something useful?
            contains_useful = False
            for ing in needed:
                if self._is_true(state, "contains", (shaker, ing)):
                    contains_useful = True
                    break
            if not contains_useful:
                needs_cleaning = True
        
        if needs_cleaning:
            cost += 3 # Empty (variable) + Clean
            
        # Add missing ingredients
        for ing in needed:
            if not self._is_true(state, "contains", (shaker, ing)):
                # Cost to add ingredient:
                # 1. Grasp Shot
                # 2. Fill Shot
                # 3. Grasp Shaker (maybe held)
                # 4. Pour Shot to Shaker
                # Approximate cycle cost: 4
                cost += 4
                
        return cost

    def _cost_clean_shot(self, state, shot):
        if self._is_true(state, "clean", (shot,)):
            return 0
        if self._is_true(state, "empty", (shot,)):
            return 1 # Clean
        return 2 # Empty + Clean

    def _cost_clean_shaker(self, state, shaker):
        if self._is_true(state, "clean", (shaker,)):
            return 0
        if self._is_true(state, "empty", (shaker,)):
            return 1 # Clean
        return 3 # Empty (complex) + Clean