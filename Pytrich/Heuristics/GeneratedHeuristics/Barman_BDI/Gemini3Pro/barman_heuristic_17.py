import sys

from Pytrich.Heuristics.heuristic import Heuristic
:
        def __init__(self, model):
            self.model = model

class BarmanGemini3ProHeuristic17(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_12=None):
        super().__init__(model)
        self.barman_gemini3pro_heuristic_12 = barman_gemini3pro_heuristic_12
        self.cocktail_ingredients = {}  # cocktail -> {ing1, ing2, ...}
        self.goals = []  # List of {'shot': s, 'cocktail': c, 'fact_id': id}
        
        # Maps for ID lookups
        self.id_clean = {}
        self.id_empty = {}
        self.id_shaked = {}
        self.id_unshaked = {}
        self.id_contains = {} # container -> {content: id}
        self.shakers = []
        
        # 1. Parse static domain knowledge (Recipes)
        self._parse_recipes()
        
        # 2. Index predicates for fast lookup (Names -> IDs) and identify objects
        self._index_predicates()
        
        # 3. Identify Goals (from model.goals or model.initial_tn)
        self._identify_goals()

    def _parse_fact_name(self, name):
        """Parses +predicate[arg1,arg2] into (predicate, [arg1, arg2])"""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
        bracket_idx = name.find("[")
        if bracket_idx > 0 and name.endswith("]"):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx+1:-1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        
        # Fallback for task names which might be "TaskName arg1 arg2" or "TaskName[arg1,arg2]"
        parts = name.replace("[", " ").replace("]", " ").replace(",", " ").split()
        if len(parts) > 1:
            return parts[0], parts[1:]
        return name, []

    def _parse_recipes(self):
        """Extracts cocktail recipes from static facts."""
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred in ['cocktailPart1', 'cocktailPart2'] and len(args) == 2:
                cocktail, ingredient = args
                if cocktail not in self.cocktail_ingredients:
                    self.cocktail_ingredients[cocktail] = set()
                self.cocktail_ingredients[cocktail].add(ingredient)

    def _index_predicates(self):
        """Builds maps from object names to fact IDs for fast state checking."""
        shaker_set = set()
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            fid = fact.global_id
            
            if pred == 'clean':
                self.id_clean[args[0]] = fid
            elif pred == 'empty':
                self.id_empty[args[0]] = fid
            elif pred == 'shaked':
                self.id_shaked[args[0]] = fid
                shaker_set.add(args[0])
            elif pred == 'unshaked':
                self.id_unshaked[args[0]] = fid
                shaker_set.add(args[0])
            elif pred == 'shakerLevel' or pred == 'shakerEmptyLevel':
                # Definitive way to identify shakers even in init state
                if args:
                    shaker_set.add(args[0])
            elif pred == 'contains':
                container, content = args
                if container not in self.id_contains:
                    self.id_contains[container] = {}
                self.id_contains[container][content] = fid
        
        self.shakers = list(shaker_set)

    def _identify_goals(self):
        """Identifies target cocktails from bitwise goals or initial task network."""
        # 1. Try bitwise goals (if provided in problem definition)
        goal_mask = self.model.goals
        found_bitwise_goals = False
        
        # Check if goal_mask is non-trivial
        if goal_mask > 0:
            for fact in self.model.facts:
                if (goal_mask >> fact.global_id) & 1:
                    pred, args = self._parse_fact_name(fact.name)
                    if pred == 'contains' and len(args) == 2:
                        container, content = args
                        if content in self.cocktail_ingredients:
                            self.goals.append({
                                'shot': container,
                                'cocktail': content,
                                'fact_id': fact.global_id
                            })
                            found_bitwise_goals = True
        
        if found_bitwise_goals:
            return

        # 2. Try Task Network if bitwise goals are empty
        # Scan for tasks like 'AchieveContainsShotCocktail'
        if hasattr(self.model, 'initial_tn'):
            for task in self.model.initial_tn:
                name = task.name
                if "AchieveContainsShotCocktail" in name:
                    # Extract args robustly using known cocktails
                    # Name might be "AchieveContainsShotCocktail shot1 cocktail1"
                    
                    # Find which cocktail is in the name
                    target_cocktail = None
                    # Sort by length to avoid partial matches (e.g. cocktail1 vs cocktail10)
                    sorted_cocktails = sorted(self.cocktail_ingredients.keys(), key=len, reverse=True)
                    
                    for c in sorted_cocktails:
                        if c in name:
                            target_cocktail = c
                            break
                    
                    if target_cocktail:
                        # Parse the name to find the shot
                        # Clean up delimiters
                        clean_name = name.replace('[', ' ').replace(']', ' ').replace(',', ' ')
                        parts = clean_name.split()
                        
                        target_shot = None
                        for p in parts:
                            if p != target_cocktail and "shot" in p and "Achieve" not in p:
                                target_shot = p
                                break
                        
                        if target_shot and target_shot in self.id_contains:
                            # Find the fact ID for (contains target_shot target_cocktail)
                            if target_cocktail in self.id_contains[target_shot]:
                                fid = self.id_contains[target_shot][target_cocktail]
                                self.goals.append({
                                    'shot': target_shot,
                                    'cocktail': target_cocktail,
                                    'fact_id': fid
                                })

    def h(self, node):
        """
        Calculates the estimated cost to reach the goal.
        Strategy: Additive cost for each unfulfilled cocktail request.
        Includes costs for: cleaning shots, mixing in shakers, and adding ingredients.
        """
        state = node.state
        cost = 0
        
        for goal in self.goals:
            fid = goal['fact_id']
            # If goal achieved, skip
            if (state >> fid) & 1:
                continue
                
            shot = goal['shot']
            cocktail = goal['cocktail']
            ingredients = self.cocktail_ingredients.get(cocktail, set())
            
            # --- 1. Shot Preparation Cost ---
            shot_cost = 0
            shot_is_clean = (shot in self.id_clean and ((state >> self.id_clean[shot]) & 1))
            shot_is_empty = (shot in self.id_empty and ((state >> self.id_empty[shot]) & 1))
            
            # If shot is not empty and doesn't contain the target (implied by goal not met),
            # it's either dirty or occupied.
            if not shot_is_empty:
                shot_cost += 3  # Empty + Clean + Grasp
            elif not shot_is_clean:
                shot_cost += 2  # Clean + Grasp
            # else: clean and empty -> cost 0
            
            cost += shot_cost
            
            # --- 2. Cocktail Preparation Cost ---
            best_prep_cost = 999
            
            # Scenario A: Cocktail is already in a shaker
            for shaker in self.shakers:
                if shaker in self.id_contains and cocktail in self.id_contains[shaker]:
                    if (state >> self.id_contains[shaker][cocktail]) & 1:
                        # Found in shaker.
                        # Check if shaked
                        is_shaked = False
                        if shaker in self.id_shaked and ((state >> self.id_shaked[shaker]) & 1):
                            is_shaked = True
                        
                        c = 1 # Pour
                        if not is_shaked:
                            c += 1 # Shake
                        
                        if c < best_prep_cost:
                            best_prep_cost = c
            
            # Scenario B: Cocktail must be made
            # Only evaluate if we haven't found it ready-made
            if best_prep_cost > 10:
                make_cost = 2 # Base: Shake + Pour
                
                # Evaluate available shakers to find the best candidate
                shaker_estimates = []
                for shaker in self.shakers:
                    s_cost = 0
                    
                    # Shaker status
                    s_clean = (shaker in self.id_clean and ((state >> self.id_clean[shaker]) & 1))
                    s_empty = (shaker in self.id_empty and ((state >> self.id_empty[shaker]) & 1))
                    
                    # Get current contents
                    contents = set()
                    if shaker in self.id_contains:
                        for c_name, c_fid in self.id_contains[shaker].items():
                            if (state >> c_fid) & 1:
                                contents.add(c_name)
                    
                    # Check for contamination (content not in recipe)
                    dirty = False
                    for c in contents:
                        if c not in ingredients:
                            dirty = True
                            break
                    
                    if dirty:
                        s_cost += 3 # Empty + Clean + Grasp
                    elif not s_clean and not s_empty and len(contents) == 0:
                        # Empty but dirty
                        s_cost += 2
                    
                    # Cost to add missing ingredients
                    for ing in ingredients:
                        if ing not in contents:
                            # Estimate: Dispense + Grasp Shot + Fill + Pour -> Shaker
                            # This is roughly 4 steps
                            s_cost += 4
                    
                    shaker_estimates.append(s_cost)
                
                if shaker_estimates:
                    make_cost += min(shaker_estimates)
                else:
                    make_cost += 20 # Penalty if no shakers exist
                
                if make_cost < best_prep_cost:
                    best_prep_cost = make_cost
            
            cost += best_prep_cost
            
        return cost
