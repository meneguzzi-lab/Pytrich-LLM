import sys
from pytrich_planner.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic11(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_6):
        super().__init__(barman_gemini3pro_heuristic_6)
        self.facts = self.model.facts
        
        # Domain Knowledge Caches
        self.cocktail_parts = {}
        self.shakers = set()
        self.shots = set()
        self.hands = set()
        self.containers = set()
        
        # Performance Optimization: Indices of dynamic facts to check in the bitwise state
        self.dynamic_fact_indices = []
        
        self._analyze_domain()

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

    def _analyze_domain(self):
        """
        Pre-computes static domain information (types, recipes) and 
        identifies relevant state variables for the heuristic.
        """
        for fact in self.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            # Static: Cocktail Recipes
            if pred == "cocktailPart1" or pred == "cocktailPart2":
                if len(args) == 2:
                    c, i = args
                    if c not in self.cocktail_parts:
                        self.cocktail_parts[c] = set()
                    self.cocktail_parts[c].add(i)
                continue # Static facts don't change, no need to track in dynamic state
            
            # Type Inference based on predicate usage
            if pred in ["shakerLevel", "shakerEmptyLevel", "shaked", "unshaked"]:
                if args:
                    self.shakers.add(args[0])
                    self.containers.add(args[0])
            elif pred == "handEmpty":
                if args:
                    self.hands.add(args[0])
            elif pred in ["clean", "empty", "contains", "ontable"]:
                if args:
                    self.containers.add(args[0])

            # Identify Dynamic Facts for efficient runtime checking
            # We only care about facts that change and affect cost (contents, cleanliness, holding)
            if pred in ["clean", "contains", "shaked", "holding", "empty"]:
                self.dynamic_fact_indices.append((fact.global_id, pred, args))
        
        # Infer Shots: Containers that are not Shakers
        self.shots = self.containers - self.shakers

    def compute(self, node):
        """
        Estimates the cost to reach the goal from the current node.
        Combines task-network analysis with state-based resource availability.
        """
        state = node.state
        task_network = node.task_network
        
        # 1. Efficient State Parsing
        # Build a local view of the world from the bitwise state
        shot_contents = {}
        shot_clean = set()
        shaker_contents = {} # Map: shaker -> set of ingredients/cocktails
        shaker_clean = set()
        shaker_shaked = set()
        holding_hand = {}
        
        # Iterate only over relevant dynamic facts
        for gid, pred, args in self.dynamic_fact_indices:
            if (state >> gid) & 1:
                if pred == "contains":
                    cont, item = args
                    if cont in self.shakers:
                        if cont not in shaker_contents:
                            shaker_contents[cont] = set()
                        shaker_contents[cont].add(item)
                    else:
                        shot_contents[cont] = item
                elif pred == "clean":
                    obj = args[0]
                    if obj in self.shakers:
                        shaker_clean.add(obj)
                    else:
                        shot_clean.add(obj)
                elif pred == "shaked":
                    shaker_shaked.add(args[0])
                elif pred == "holding":
                    hand, obj = args
                    holding_hand[hand] = obj

        # 2. Heuristic Calculation
        h_val = 0
        
        # Estimated Action Costs
        # These are weights representing the minimal sequence length for subgoals
        COST_PICKUP = 1
        COST_POUR = 1
        COST_SHAKE = 1
        # Clean shot sequence: grasp(1) + empty(1) + clean(1) + drop(1) ~ 4
        COST_CLEAN_SHOT = 4 
        # Clean shaker sequence: grasp(1) + empty(1) + clean(1) + drop(1) ~ 4
        COST_CLEAN_SHAKER = 4
        # Add Ingredient sequence: grasp shot(1) + fill(1) + pour(1) + drop(1) ~ 4
        COST_ADD_ING = 4
        
        for task in task_network:
            # Robust name parsing
            tname = task.name.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            parts = tname.split()
            if not parts: continue
            task_type = parts[0]
            
            if "AchieveContainsShotCocktail" in task_type:
                # Expected args: shot, cocktail
                if len(parts) < 3: continue
                shot = parts[1]
                cocktail = parts[2]
                
                # If already achieved, cost is 0
                if shot_contents.get(shot) == cocktail:
                    continue
                
                task_cost = 0
                
                # Subgoal 1: Clean Shot
                # If shot is not clean and doesn't contain the target, it needs cleaning
                # (If it contains target, we caught it above)
                if shot not in shot_clean:
                    task_cost += COST_CLEAN_SHOT
                
                # Subgoal 2: Prepare Cocktail
                # Check resource availability
                recipe = self.cocktail_parts.get(cocktail, set())
                
                ready_shaker = None
                partial_shaker = None
                
                # Scan shakers for cocktail or ingredients
                for s in self.shakers:
                    contents = shaker_contents.get(s, set())
                    
                    # Case A: Cocktail already made and shaken
                    if cocktail in contents and s in shaker_shaked:
                        ready_shaker = s
                        break
                    
                    # Case B: Ingredients present but not shaken
                    # Note: contents is a set, recipe is a set. Check subset.
                    if recipe and recipe.issubset(contents) and s not in shaker_shaked:
                        partial_shaker = s
                
                if ready_shaker:
                    # Just need to pour
                    task_cost += COST_POUR
                    # Plus pickup if not held
                    if not any(holding_hand.get(h) == ready_shaker for h in holding_hand):
                        task_cost += COST_PICKUP
                        
                elif partial_shaker:
                    # Need to shake then pour
                    task_cost += COST_SHAKE + COST_POUR
                    if not any(holding_hand.get(h) == partial_shaker for h in holding_hand):
                        task_cost += COST_PICKUP
                        
                else:
                    # Case C: Make from scratch
                    # 1. Need a shaker. Optimistically assume we can use one.
                    # If all shakers are dirty/busy, add cleaning cost.
                    has_usable_shaker = any(s in shaker_clean and s not in shaker_contents for s in self.shakers)
                    if not has_usable_shaker:
                        task_cost += COST_CLEAN_SHAKER
                    
                    # 2. Add ingredients
                    # Cost depends on number of ingredients
                    num_ings = len(recipe)
                    task_cost += num_ings * COST_ADD_ING
                    
                    # 3. Shake and Pour
                    task_cost += COST_SHAKE + COST_POUR
                
                h_val += task_cost

            elif "AchieveCleanShot" in task_type:
                if len(parts) > 1:
                    shot = parts[1]
                    if shot not in shot_clean:
                        h_val += COST_CLEAN_SHOT

            elif "AchieveCleanShaker" in task_type:
                if len(parts) > 1:
                    shaker = parts[1]
                    if shaker not in shaker_clean:
                        h_val += COST_CLEAN_SHAKER
            
            # Fallback for generic tasks or lower level actions if they appear in TN
            # In HDDL, primitive actions have cost 1 usually.
            # We focus on the high-level tasks dominating the plan length.

        return h_val
