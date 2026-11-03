from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic18(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_13=None):
        super().__init__(model)
        self.model = model
        
        # Object sets identified from facts
        self.shakers = set()
        self.shots = set()
        self.cocktails = set()
        self.ingredients = set()
        
        # Mappings for fast state checking: (predicate_key) -> global_id
        self.contains_map = {}
        self.clean_map = {}
        self.shaked_map = {}
        self.empty_map = {}
        
        # Recipes: cocktail -> set(ingredients)
        self.recipes = {}
        
        # Precompute indices from model facts
        self._initialize_indices()
        
        # Convert sets to lists for iteration if needed
        self.shaker_list = list(self.shakers)

    def _initialize_indices(self):
        for fact in self.model.facts:
            name = fact.name
            gid = fact.global_id
            
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
                
            if pred == "contains" and len(args) == 2:
                self.contains_map[(args[0], args[1])] = gid
                # Infer object types if possible
                if "shaker" in args[0]: self.shakers.add(args[0])
                if "shot" in args[0]: self.shots.add(args[0])
                
            elif pred == "clean" and len(args) == 1:
                self.clean_map[args[0]] = gid
                
            elif pred == "shaked" and len(args) == 1:
                self.shaked_map[args[0]] = gid
                self.shakers.add(args[0])
                
            elif pred == "empty" and len(args) == 1:
                self.empty_map[args[0]] = gid
                
            elif pred == "cocktailPart1" or pred == "cocktailPart2":
                # Static facts for recipes
                if len(args) == 2:
                    c, i = args[0], args[1]
                    self.cocktails.add(c)
                    self.ingredients.add(i)
                    if c not in self.recipes:
                        self.recipes[c] = set()
                    self.recipes[c].add(i)
                    
            elif pred == "shakerLevel":
                if len(args) >= 1: 
                    self.shakers.add(args[0])

    def _parse_fact_name(self, name: str):
        # Format: +predicate[arg1,arg2]
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

    def _parse_task_name(self, task_name: str):
        # Handles "TaskName arg1 arg2" or "TaskName[arg1,arg2]"
        t = task_name.strip()
        if "[" in t:
            idx = t.find("[")
            name = t[:idx]
            args = t[idx+1:-1].split(",")
            return name, args
        
        parts = t.split()
        if not parts:
            return "", []
        return parts[0], parts[1:]

    def _is_true(self, state, global_id):
        return (state >> global_id) & 1

    def h(self, node):
        state = node.state
        tn = node.tn
        
        total_cost = 0
        
        for task in tn:
            # Handle primitive operators
            if hasattr(task, 'cost') and hasattr(task, 'add_effects'):
                total_cost += getattr(task, 'cost', 1)
                continue
                
            # Handle abstract tasks
            name, args = self._parse_task_name(task.name)
            
            if name == "AchieveContainsShotCocktail":
                if len(args) >= 2:
                    total_cost += self._estimate_shot_cocktail(state, args[0], args[1])
                else:
                    total_cost += 10
                    
            elif name == "AchieveContainsShakerCocktail":
                if len(args) >= 2:
                    total_cost += self._estimate_shaker_cocktail(state, args[0], args[1])
                else:
                    total_cost += 8
                    
            elif name == "AchieveContainsShakerIngredient":
                if len(args) >= 2:
                    total_cost += self._estimate_shaker_ingredient(state, args[0], args[1])
                else:
                    total_cost += 4
                    
            elif name == "AchieveCleanShot":
                if len(args) >= 1:
                    total_cost += self._estimate_clean(state, args[0])
                else:
                    total_cost += 2
                    
            elif name == "AchieveCleanShaker":
                if len(args) >= 1:
                    total_cost += self._estimate_clean(state, args[0])
                else:
                    total_cost += 2
            
            elif name == "MakeCocktail":
                total_cost += 10
                
            elif name == "DoPourShakerToShot":
                total_cost += 1
                
            elif name == "AchieveHolding":
                total_cost += 1
                
            elif name == "AchieveHandEmpty":
                # Optimistic: 0 if already empty, 1 if holding something
                # Hard to check cheaply without checking all hands. Assume minimal cost.
                total_cost += 0 
            
            else:
                # Default fallback for unknown tasks
                total_cost += 1
                
        return total_cost

    def _estimate_shot_cocktail(self, state, shot, cocktail):
        # 1. Check if goal is already satisfied
        gid = self.contains_map.get((shot, cocktail))
        if gid is not None and self._is_true(state, gid):
            return 0
            
        # 2. Check if cocktail is ready in any shaker
        cocktail_ready_in_shaker = False
        for shaker in self.shaker_list:
            c_gid = self.contains_map.get((shaker, cocktail))
            s_gid = self.shaked_map.get(shaker)
            
            is_contains = c_gid is not None and self._is_true(state, c_gid)
            is_shaked = s_gid is not None and self._is_true(state, s_gid)
            
            if is_contains and is_shaked:
                cocktail_ready_in_shaker = True
                break
        
        clean_cost = self._estimate_clean(state, shot)
        
        if cocktail_ready_in_shaker:
            # Remaining: [Clean Shot] + Grasp Shaker + Pour
            return 2 + clean_cost
            
        # 3. Cocktail not made. Need to make it.
        # Cost decomposition: 
        #   Clean Shaker (2) + Add Ing1 (4) + Add Ing2 (4) + Shake (1) + Pour (1) + Clean Shot (varies)
        #   Total approx 12 + clean_cost
        return 12 + clean_cost

    def _estimate_shaker_cocktail(self, state, shaker, cocktail):
        # Goal: shaker contains cocktail and is shaked
        c_gid = self.contains_map.get((shaker, cocktail))
        s_gid = self.shaked_map.get(shaker)
        
        is_contains = c_gid is not None and self._is_true(state, c_gid)
        is_shaked = s_gid is not None and self._is_true(state, s_gid)
        
        if is_contains and is_shaked:
            return 0
            
        if is_contains:
            # Just need to shake
            return 1
            
        # Not contains: need to add ingredients and shake
        # Assumes empty/clean shaker or includes cleaning cost implicitly in higher abstract task
        # But here we just estimate steps to fill and shake
        # Add Ing1 (4) + Add Ing2 (4) + Shake (1)
        return 9

    def _estimate_shaker_ingredient(self, state, shaker, ingredient):
        # Goal: shaker contains ingredient
        gid = self.contains_map.get((shaker, ingredient))
        if gid is not None and self._is_true(state, gid):
            return 0
            
        # Actions: Clean Shot + Fill Shot + Pour to Shaker
        # Avg cost ~ 4
        return 4

    def _estimate_clean(self, state, container):
        gid = self.clean_map.get(container)
        if gid is not None and self._is_true(state, gid):
            return 0
        
        # If not clean, needs cleaning.
        # Actions: Grasp + Clean (if empty) OR Grasp + Empty + Clean (if full)
        # Check if empty to be more precise
        e_gid = self.empty_map.get(container)
        if e_gid is not None and self._is_true(state, e_gid):
            return 2 # Grasp + Clean
        
        return 3 # Grasp + Empty + Clean