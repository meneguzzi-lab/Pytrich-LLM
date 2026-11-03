from Pytrich.Heuristics.heuristic import Heuristic
import sys

class BarmanGemini3ProHeuristic14(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_9):
        super().__init__(barman_gemini3pro_heuristic_9)
        self.barman_gemini3pro_heuristic_9 = barman_gemini3pro_heuristic_9
        self.model = barman_gemini3pro_heuristic_9.model
        
        # Estimated costs for high-level actions
        self.COST_POUR = 1
        self.COST_SHAKE = 1
        self.COST_CLEAN_SHOT = 3     # Grasp + Empty + Clean
        self.COST_CLEAN_SHAKER = 3   # Grasp + Empty + Clean
        self.COST_ADD_ING = 4        # Grasp Shot + Fill + Pour + Drop
        
        # Domain Knowledge Caches
        self.cocktail_recipes = {}  # cocktail -> set(ingredients)
        self.shakers = []
        self.shots = []
        self.goals = []  # List of dicts: {'shot': s, 'cocktail': c, 'fact_id': id}
        
        # Fact ID Lookup Maps
        self.contains_fact_ids = {} # (container, content) -> global_id
        self.clean_fact_ids = {}    # container -> global_id
        self.empty_fact_ids = {}    # container -> global_id
        self.shaked_fact_ids = {}   # shaker -> global_id
        self.unshaked_fact_ids = {} # shaker -> global_id
        
        self._initialize_domain_knowledge()

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _initialize_domain_knowledge(self):
        # 1. First pass: Identify objects and static relationships (recipes)
        all_containers = set()
        
        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
                
            if pred == 'cocktailPart1' or pred == 'cocktailPart2':
                # args: [cocktail, ingredient]
                c, i = args[0], args[1]
                if c not in self.cocktail_recipes:
                    self.cocktail_recipes[c] = set()
                self.cocktail_recipes[c].add(i)
                
            elif pred == 'shakerLevel':
                # args: [shaker, level]
                if args[0] not in self.shakers:
                    self.shakers.append(args[0])
                all_containers.add(args[0])
                
            elif pred == 'clean':
                # args: [container]
                all_containers.add(args[0])

        # Infer shots: Containers that are not shakers
        for c in all_containers:
            if c not in self.shakers:
                self.shots.append(c)

        # 2. Second pass: Map dynamic facts to IDs for fast state lookup
        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            fid = fact.global_id
            
            if not pred:
                continue

            if pred == 'contains':
                self.contains_fact_ids[(args[0], args[1])] = fid
            elif pred == 'clean':
                self.clean_fact_ids[args[0]] = fid
            elif pred == 'empty':
                self.empty_fact_ids[args[0]] = fid
            elif pred == 'shaked':
                self.shaked_fact_ids[args[0]] = fid
            elif pred == 'unshaked':
                self.unshaked_fact_ids[args[0]] = fid

        # 3. Identify Goals from the Initial Task Network
        # We look for 'AchieveContainsShotCocktail' tasks which define the problem goals
        if self.model.initial_tn:
            for task in self.model.initial_tn:
                # Handle task names that might be formatted differently
                # Expected: "AchieveContainsShotCocktail shot1 cocktail1"
                raw_name = task.name
                # Clean up name to handle potential function-style formatting
                clean_name = raw_name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
                parts = clean_name.split()
                
                if not parts:
                    continue
                    
                task_type = parts[0]
                # Check for the main goal task type
                if "AchieveContainsShotCocktail" in task_type and len(parts) >= 3:
                    shot = parts[1]
                    cocktail = parts[2]
                    
                    # Find the specific ground fact ID for this goal
                    fid = self.contains_fact_ids.get((shot, cocktail))
                    if fid is not None:
                        self.goals.append({
                            'shot': shot,
                            'cocktail': cocktail,
                            'fact_id': fid
                        })

    def h(self, node):
        state = node.state
        total_cost = 0
        
        # Iterate over all identified goals
        for goal in self.goals:
            goal_fid = goal['fact_id']
            
            # If goal is already satisfied in current state, cost is 0
            if (state >> goal_fid) & 1:
                continue
            
            shot = goal['shot']
            cocktail = goal['cocktail']
            
            # Base cost: We definitely need to pour the cocktail into the shot
            total_cost += self.COST_POUR
            
            # 1. Analyze Shot State
            # If shot is not clean or not empty, we must clean it
            clean_id = self.clean_fact_ids.get(shot)
            empty_id = self.empty_fact_ids.get(shot)
            
            is_clean = (state >> clean_id) & 1 if clean_id is not None else 0
            is_empty = (state >> empty_id) & 1 if empty_id is not None else 0
            
            if not (is_clean and is_empty):
                total_cost += self.COST_CLEAN_SHOT
                
            # 2. Analyze Cocktail Production State
            # Check if the cocktail is already available in any shaker
            cocktail_ready = False
            cocktail_mixed_but_unshaked = False
            
            for shaker in self.shakers:
                cont_id = self.contains_fact_ids.get((shaker, cocktail))
                if cont_id is not None and ((state >> cont_id) & 1):
                    # Shaker contains the cocktail
                    # Check if it is shaked
                    shaked_id = self.shaked_fact_ids.get(shaker)
                    if shaked_id is not None and ((state >> shaked_id) & 1):
                        cocktail_ready = True
                        break # Ready to pour, no extra production cost
                    else:
                        cocktail_mixed_but_unshaked = True
            
            if cocktail_ready:
                # Only pour cost (already added)
                pass
            elif cocktail_mixed_but_unshaked:
                # Need to shake before pouring
                total_cost += self.COST_SHAKE
            else:
                # 3. Cocktail needs to be made
                # Estimate cost to make: Add ingredients + Shake
                ingredients = self.cocktail_recipes.get(cocktail, set())
                num_ings = len(ingredients) if ingredients else 2 # Default to 2 if unknown
                
                make_cost = (num_ings * self.COST_ADD_ING) + self.COST_SHAKE
                
                # We also need a clean shaker. Find the best candidate.
                # Using a dirty shaker adds cleaning cost.
                min_shaker_prep_cost = 9999
                
                for shaker in self.shakers:
                    prep_cost = 0
                    s_empty_id = self.empty_fact_ids.get(shaker)
                    s_clean_id = self.clean_fact_ids.get(shaker)
                    
                    is_s_empty = (state >> s_empty_id) & 1 if s_empty_id is not None else 0
                    is_s_clean = (state >> s_clean_id) & 1 if s_clean_id is not None else 0
                    
                    if is_s_empty and is_s_clean:
                        prep_cost = 0
                    elif is_s_empty and not is_s_clean:
                        # Empty but dirty (used)
                        prep_cost = self.COST_CLEAN_SHAKER
                    elif not is_s_empty:
                        # Full (likely used or has wrong ingredients)
                        # Optimistic: just clean it
                        prep_cost = self.COST_CLEAN_SHAKER
                    
                    if prep_cost < min_shaker_prep_cost:
                        min_shaker_prep_cost = prep_cost
                
                # If no shakers found (shouldn't happen), fallback
                if min_shaker_prep_cost == 9999:
                    min_shaker_prep_cost = self.COST_CLEAN_SHAKER
                    
                total_cost += make_cost + min_shaker_prep_cost
                
        return float(total_cost)