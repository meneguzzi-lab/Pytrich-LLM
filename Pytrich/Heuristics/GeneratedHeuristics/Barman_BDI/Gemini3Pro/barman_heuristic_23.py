from pytrich_planner.heuristic import Heuristic

class BarmanGemini3ProHeuristic23(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_18):
        super().__init__(barman_gemini3pro_heuristic_18)
        self.model = barman_gemini3pro_heuristic_18
        
        # Domain objects and maps
        self.shakers = set()
        self.shots = set()
        self.cocktail_ingredients = {}  # cocktail -> list of ingredients
        
        # Fact ID lookups for fast state checking
        self.shaker_clean_ids = {}      # shaker -> global_id
        self.shaker_empty_ids = {}      # shaker -> global_id
        self.shaker_shaked_ids = {}     # shaker -> global_id
        self.shot_clean_ids = {}        # shot -> global_id
        self.shot_empty_ids = {}        # shot -> global_id
        
        # Map (container, content) -> global_id
        self.contains_ids = {}
        
        # Goals parsed from model.goals
        self.goal_tasks = []  # List of (shot, cocktail)
        
        self._parse_domain_structure()
        self._identify_goals()

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

    def _parse_domain_structure(self):
        """Parses static facts and builds object sets and ID lookups."""
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            gid = fact.global_id

            if pred == "shakerLevel":
                self.shakers.add(args[0])
            
            if pred == "cocktailPart1" or pred == "cocktailPart2":
                # Check if this static fact is true in initial state
                if (self.model.initial_state >> gid) & 1:
                    c, i = args[0], args[1]
                    if c not in self.cocktail_ingredients:
                        self.cocktail_ingredients[c] = set()
                    self.cocktail_ingredients[c].add(i)
            
            if pred == "clean":
                obj = args[0]
                # We will distinguish shots from shakers later, or store for both
                # But we can store in temporary dicts first
                self.shaker_clean_ids[obj] = gid
                self.shot_clean_ids[obj] = gid
            
            if pred == "empty":
                obj = args[0]
                self.shaker_empty_ids[obj] = gid
                self.shot_empty_ids[obj] = gid
            
            if pred == "shaked":
                self.shakers.add(args[0])
                self.shaker_shaked_ids[args[0]] = gid
            
            if pred == "contains":
                container, content = args[0], args[1]
                self.contains_ids[(container, content)] = gid

        # Filter shots: things that are cleaned/emptied but not shakers
        all_cleaned = set(self.shot_clean_ids.keys())
        self.shots = all_cleaned - self.shakers
        
        # Cleanup ID maps to be type specific
        self.shaker_clean_ids = {k: v for k, v in self.shaker_clean_ids.items() if k in self.shakers}
        self.shaker_empty_ids = {k: v for k, v in self.shaker_empty_ids.items() if k in self.shakers}
        self.shot_clean_ids = {k: v for k, v in self.shot_clean_ids.items() if k in self.shots}
        self.shot_empty_ids = {k: v for k, v in self.shot_empty_ids.items() if k in self.shots}

    def _identify_goals(self):
        """Identifies specific cocktail goals from the bitwise model.goals."""
        goals_mask = self.model.goals
        if goals_mask == 0:
            return
            
        # Iterate through all 'contains' facts to see if they are goals
        for (container, content), gid in self.contains_ids.items():
            if (goals_mask >> gid) & 1:
                # This is a goal
                if container in self.shots:
                    self.goal_tasks.append((container, content))

    def h(self, node):
        state = node.state
        total_cost = 0
        
        for shot, cocktail in self.goal_tasks:
            # 1. Check if goal is already met
            gid = self.contains_ids.get((shot, cocktail))
            if gid is not None and ((state >> gid) & 1):
                continue
                
            # 2. Estimate cost to achieve this cocktail in this shot
            # Cost breakdown: Clean Shot + Acquire Cocktail
            
            # A. Shot State Cost
            shot_cost = 0
            is_shot_clean = False
            is_shot_empty = False
            
            clean_id = self.shot_clean_ids.get(shot)
            if clean_id is not None and ((state >> clean_id) & 1):
                is_shot_clean = True
            
            empty_id = self.shot_empty_ids.get(shot)
            if empty_id is not None and ((state >> empty_id) & 1):
                is_shot_empty = True
            
            if not is_shot_clean:
                if not is_shot_empty:
                    shot_cost += 3  # Empty + Clean + (overhead)
                else:
                    shot_cost += 2  # Clean + (overhead)
            elif is_shot_clean and not is_shot_empty:
                # Technically impossible state for 'clean' predicate usually, 
                # but implies holding something compatible or just filled.
                # If we need a specific cocktail and it's not there (checked above),
                # we must empty it.
                shot_cost += 3

            # B. Cocktail Availability Cost
            # Check if cocktail exists in any shaker
            cocktail_found_cost = 1000
            found_in_shaker = False
            
            for shaker in self.shakers:
                cont_id = self.contains_ids.get((shaker, cocktail))
                if cont_id is not None and ((state >> cont_id) & 1):
                    # Found cocktail in shaker
                    found_in_shaker = True
                    
                    # Check if shaked
                    shaked_id = self.shaker_shaked_ids.get(shaker)
                    is_shaked = False
                    if shaked_id is not None and ((state >> shaked_id) & 1):
                        is_shaked = True
                    
                    cost_here = 1 if is_shaked else 2 # Pour vs Shake+Pour
                    if cost_here < cocktail_found_cost:
                        cocktail_found_cost = cost_here
            
            if not found_in_shaker:
                # Must make cocktail
                # Cost: Prepare Shaker + Add Ing1 + Add Ing2 + Shake + Pour
                
                # Find best shaker to use
                best_shaker_prep_cost = 100
                for shaker in self.shakers:
                    s_cost = 0
                    is_clean = False
                    is_empty = False
                    
                    sc_id = self.shaker_clean_ids.get(shaker)
                    if sc_id is not None and ((state >> sc_id) & 1): is_clean = True
                    
                    se_id = self.shaker_empty_ids.get(shaker)
                    if se_id is not None and ((state >> se_id) & 1): is_empty = True
                    
                    if is_clean and is_empty:
                        s_cost = 0
                    elif is_empty and not is_clean:
                        s_cost = 2 # Clean
                    elif not is_empty:
                        s_cost = 3 # Empty + Clean
                        
                    if s_cost < best_shaker_prep_cost:
                        best_shaker_prep_cost = s_cost
                
                # Add ingredients cost
                # Approx 3 steps per ingredient (Grasp, Fill, Pour-to-shaker)
                num_ingredients = len(self.cocktail_ingredients.get(cocktail, []))
                if num_ingredients == 0: num_ingredients = 2 # Fallback
                
                ingredients_cost = num_ingredients * 3
                
                # Total to make: Prep + Ings + Shake + Pour
                cocktail_found_cost = best_shaker_prep_cost + ingredients_cost + 1 + 1
            
            total_cost += shot_cost + cocktail_found_cost
            
        return total_cost
