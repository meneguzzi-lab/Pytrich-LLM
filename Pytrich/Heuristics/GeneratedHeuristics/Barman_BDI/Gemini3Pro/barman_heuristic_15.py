from pytrich.heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic15(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_10=None):
        super().__init__(model)
        self.model = model
        self.fact_name_to_id = {f.name: f.global_id for f in model.facts}
        self.shakers = set()
        self.recipes = {} # cocktail -> set of ingredients

        # Pre-process static facts and identify objects from the initial state
        init_state = self.model.initial_state
        # Usually init_state is an integer bitmask. If it's a set in some versions, handle that, 
        # but context implies bitwise. We iterate all facts to find statics.
        
        for fact in self.model.facts:
            # Check if fact is true in initial state
            is_true = False
            if isinstance(init_state, int):
                is_true = (init_state >> fact.global_id) & 1
            elif isinstance(init_state, set):
                is_true = fact.global_id in init_state
                
            if is_true:
                name = fact.name
                # Name format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
                # We expect positive facts in init
                if not name.startswith("+"): continue
                
                # Simple parsing
                clean_name = name[1:].replace("]", "").replace("[", " ").replace(",", " ")
                parts = clean_name.split()
                if not parts: continue
                
                pred = parts[0]
                args = parts[1:]
                
                if pred == "cocktailPart1" or pred == "cocktailPart2":
                    if len(args) >= 2:
                        c, i = args[0], args[1]
                        if c not in self.recipes: self.recipes[c] = set()
                        self.recipes[c].add(i)
                        
                elif pred == "shakerLevel":
                    if len(args) >= 1:
                        self.shakers.add(args[0])
                        
        self.shakers = list(self.shakers)

    def _get_fact_id(self, predicate, args):
        # Construct grounded fact name: +predicate[arg1,arg2]
        name = f"+{predicate}[{','.join(args)}]"
        return self.fact_name_to_id.get(name, -1)

    def _is_true(self, state, predicate, args):
        fid = self._get_fact_id(predicate, args)
        if fid == -1: return False
        return (state >> fid) & 1

    def __call__(self, node):
        h_val = 0
        state = node.state
        tn = node.tn
        
        for task in tn:
            if hasattr(task, 'decompositions'): # AbstractTask
                # Parse task name and arguments
                # Task names in Pytrich are strings, typically "TaskName arg1 arg2" or "TaskName[arg1,arg2]"
                raw_name = task.name
                clean_name = raw_name.replace("]", "").replace("[", " ").replace(",", " ")
                parts = clean_name.split()
                if not parts: 
                    h_val += 1
                    continue
                    
                task_type = parts[0]
                args = parts[1:]
                
                if task_type == "AchieveContainsShotCocktail":
                    if len(args) >= 2:
                        shot, cocktail = args[0], args[1]
                        
                        # If already done, cost 0
                        if self._is_true(state, "contains", [shot, cocktail]):
                            continue
                            
                        # If not done, we estimate cost
                        # 1. Check if cocktail is already in a shaker
                        cocktail_ready_in_shaker = False
                        shaker_with_parts = None
                        
                        for s in self.shakers:
                            if self._is_true(state, "contains", [s, cocktail]):
                                if self._is_true(state, "shaked", [s]):
                                    cocktail_ready_in_shaker = True
                                    break
                                else:
                                    # Contains cocktail but not shaken? (Actually, logic says it contains ingredients
                                    # until shaken, then contains cocktail. But let's handle the predicate logic.)
                                    # If 'contains cocktail' is true, it is usually after shaking.
                                    cocktail_ready_in_shaker = True 
                                    break
                        
                        if cocktail_ready_in_shaker:
                            h_val += 1 # Pour
                        else:
                            # Need to make it.
                            # Cost involves: Clean Shaker (maybe) + Add Ing1 + Add Ing2 + Shake + Pour
                            
                            # Base cost for mixing and pouring
                            h_val += 2 # Shake + Pour
                            
                            # Check ingredients
                            ingredients = self.recipes.get(cocktail, [])
                            shaker_needs_cleaning = True
                            
                            # Check if any shaker is clean or has partial ingredients
                            # Heuristic simplification: assume we use best shaker
                            best_shaker_cost = 100
                            
                            for s in self.shakers:
                                s_cost = 0
                                if not self._is_true(state, "clean", [s]) and not self._is_true(state, "empty", [s]):
                                    # Shaker dirty/full of something else
                                    # Unless it contains required ingredients
                                    pass # Logic below checks ingredients
                                
                                # Check ingredients present
                                missing_count = 0
                                has_bad_content = False
                                
                                # Rough check for bad content: 
                                # If not empty and not clean, and we don't know what's in it, assume bad.
                                # We refine: if it contains an ingredient not in recipe, it's bad.
                                # This is expensive to check exhaustively. 
                                # Simpler: If not clean/empty, add cleaning cost, subtract if we find useful ingredients.
                                
                                if not self._is_true(state, "clean", [s]) and not self._is_true(state, "empty", [s]):
                                    s_cost += 2 # Empty + Clean
                                elif self._is_true(state, "empty", [s]) and not self._is_true(state, "clean", [s]):
                                    s_cost += 1 # Clean
                                    
                                for ing in ingredients:
                                    if not self._is_true(state, "contains", [s, ing]):
                                        missing_count += 1
                                    else:
                                        # If it contains a useful ingredient, we shouldn't have charged for cleaning if we did
                                        # But simplistic logic is fine. 
                                        # If we found useful ingredient, assume we don't clean.
                                        if s_cost > 0: s_cost = 0
                                        
                                s_cost += (missing_count * 4) # 4 = Clean shot + Fill + Pickup + Pour
                                if s_cost < best_shaker_cost:
                                    best_shaker_cost = s_cost
                            
                            h_val += best_shaker_cost

                        # Clean Shot Cost
                        if not self._is_true(state, "clean", [shot]):
                            if not self._is_true(state, "empty", [shot]):
                                h_val += 2 # Empty + Clean
                            else:
                                h_val += 1 # Clean
                                
                elif task_type == "AchieveContainsShakerCocktail":
                     if len(args) >= 2:
                        shaker, cocktail = args[0], args[1]
                        if not self._is_true(state, "contains", [shaker, cocktail]):
                             h_val += 1 # Shake
                             ingredients = self.recipes.get(cocktail, [])
                             for ing in ingredients:
                                 if not self._is_true(state, "contains", [shaker, ing]):
                                     h_val += 4

                elif task_type == "AchieveCleanShot":
                    if len(args) >= 1:
                        shot = args[0]
                        if not self._is_true(state, "clean", [shot]):
                            h_val += 1
                            if not self._is_true(state, "empty", [shot]):
                                h_val += 1

                elif task_type == "AchieveCleanShaker":
                    if len(args) >= 1:
                        shaker = args[0]
                        if not self._is_true(state, "clean", [shaker]):
                            h_val += 1
                            if not self._is_true(state, "empty", [shaker]):
                                h_val += 1
                
                elif task_type == "AchieveContainsShakerIngredient":
                    if len(args) >= 2:
                        shaker, ing = args[0], args[1]
                        if not self._is_true(state, "contains", [shaker, ing]):
                            h_val += 3
                            
                elif task_type == "AchieveHandEmpty":
                    if len(args) >= 1:
                        hand = args[0]
                        if not self._is_true(state, "handEmpty", [hand]):
                            h_val += 1
                            
                elif task_type == "AchieveHolding":
                    if len(args) >= 2:
                        hand, obj = args[0], args[1]
                        if not self._is_true(state, "holding", [hand, obj]):
                            h_val += 1
                
                else:
                    # Default for unknown tasks
                    h_val += 1
            else:
                # Primitive Operator
                h_val += 1
                
        return h_val
