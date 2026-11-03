from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic22(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_17=None):
        super().__init__(model)
        self.barman_gemini3pro_heuristic_17 = barman_gemini3pro_heuristic_17
        
        # Precompute mappings for fast fact lookup
        self.fact_name_to_id = {}
        self.shakers = set()
        self.cocktail_ingredients = {}
        
        # Iterate over all facts in the model to build indices and static knowledge
        for fact in self.model.facts:
            name = fact.name
            self.fact_name_to_id[name] = fact.global_id
            
            # Parse grounded fact: +pred[arg1,arg2]
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            
            # Identify shakers (objects that have a shakerLevel)
            if pred == "shakerLevel" and args:
                self.shakers.add(args[0])
            
            # Identify cocktail recipes (static knowledge)
            if pred == "cocktailPart1" and len(args) == 2:
                c, i = args[0], args[1]
                if c not in self.cocktail_ingredients:
                    self.cocktail_ingredients[c] = set()
                self.cocktail_ingredients[c].add(i)
            
            if pred == "cocktailPart2" and len(args) == 2:
                c, i = args[0], args[1]
                if c not in self.cocktail_ingredients:
                    self.cocktail_ingredients[c] = set()
                self.cocktail_ingredients[c].add(i)
        
        # Fallback if no shakers detected from facts (e.g. if logic changes)
        if not self.shakers:
            # Try to infer from clean facts or standard naming
            for fact in self.model.facts:
                if "shaker" in fact.name.lower() and "clean" in fact.name:
                    pred, args = self._parse_fact_name(fact.name)
                    if args:
                        self.shakers.add(args[0])

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state, pred, args):
        arg_str = ",".join(args)
        name = f"+{pred}[{arg_str}]"
        fid = self.fact_name_to_id.get(name, -1)
        if fid != -1:
            return (state >> fid) & 1
        return False

    def h(self, node):
        state = node.state
        tn = node.tn
        cost = 0
        
        # Helper for succinct checks
        def check(p, a): return self._is_true(state, p, a)

        # 1. Analyze Task Network
        # In HTN, the remaining effort is determined by the tasks in the network.
        # We iterate over the tasks and assign costs based on domain knowledge.
        
        for task in tn:
            # Robust name parsing: remove parens and split
            # Expected format: "TaskName arg1 arg2 ..."
            clean_name = task.name.replace("(", " ").replace(")", " ").strip()
            parts = clean_name.split()
            if not parts: continue
            
            task_type = parts[0]
            args = parts[1:]
            
            if "AchieveContainsShotCocktail" in task_type:
                # Args: ?shot ?cocktail
                if len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    
                    # If goal already physically met, cost is 0 (or close to)
                    if check("contains", [shot, cocktail]):
                        continue

                    # Base cost for the workflow
                    task_cost = 0
                    
                    # 1. Shot preparation
                    is_clean_shot = check("clean", [shot])
                    is_empty_shot = check("empty", [shot])
                    
                    if not is_clean_shot:
                        if not is_empty_shot:
                            task_cost += 4 # Empty (1) + Clean (3)
                        else:
                            task_cost += 3 # Clean (3)
                            
                    # 2. Shaker preparation & Mixing
                    # We assume we use the best available shaker (lowest cost)
                    best_shaker_cost = float('inf')
                    
                    for shaker in self.shakers:
                        s_cost = 0
                        has_cocktail = check("contains", [shaker, cocktail])
                        
                        if has_cocktail:
                            if not check("shaked", [shaker]):
                                s_cost += 2 # Shake (1) + Pour (1)
                            else:
                                s_cost += 1 # Pour (1)
                        else:
                            # Needs ingredients. Check if shaker is ready to accept them.
                            is_clean_shaker = check("clean", [shaker])
                            is_empty_shaker = check("empty", [shaker])
                            
                            if not is_clean_shaker or not is_empty_shaker:
                                s_cost += 3 # Clean/Empty sequence approx
                            
                            # Add ingredient costs
                            ings = self.cocktail_ingredients.get(cocktail, set())
                            # Cost: Grasp(1) + Pour(1) per ingredient + Shake(1) + PourToShot(1)
                            # However, 'AddIngredient' method usually involves setup.
                            # Minimal ops: 2 per ingredient.
                            s_cost += (len(ings) * 2) + 2
                        
                        if s_cost < best_shaker_cost:
                            best_shaker_cost = s_cost
                    
                    if not self.shakers:
                        best_shaker_cost = 15 # Fallback
                        
                    cost += task_cost + best_shaker_cost
                else:
                    cost += 10 # Unknown args penalty
            
            elif "AchieveCleanShot" in task_type:
                if len(args) >= 1 and not check("clean", [args[0]]):
                    cost += 3
            
            elif "AchieveCleanShaker" in task_type:
                if len(args) >= 1 and not check("clean", [args[0]]):
                    cost += 3
            
            elif "AchieveContainsShakerIngredient" in task_type:
                if len(args) >= 2:
                    shaker, ing = args[0], args[1]
                    if not check("contains", [shaker, ing]):
                        cost += 3 # Grasp + Pour + Setup
            
            elif "AchieveContainsShotIngredient" in task_type:
                 if len(args) >= 2 and not check("contains", [args[0], args[1]]):
                     cost += 3
                     
            elif "AchieveHandEmpty" in task_type:
                # Check if hands are actually full? We don't have hand arg easily sometimes.
                # Assume cost 1
                cost += 1
                
            elif "AchieveHolding" in task_type:
                if len(args) >= 2 and not check("holding", [args[0], args[1]]):
                    cost += 2 # Move + Grasp
            
            elif "shake" in task_type.lower():
                cost += 1
            elif "pour" in task_type.lower():
                cost += 1
            else:
                # Default small cost for other steps to encourage progress
                cost += 1
                
        # 2. Goal State Check (Bitwise)
        # If the TN is empty or we are in a state where goals define remaining work independent of tasks
        # (e.g. at the end of a decomposition), we check the explicit goals.
        if not tn and self.model.goals:
            # Calculate unsatisfied goal bits
            unsatisfied = (self.model.goals & ~state)
            if unsatisfied != 0:
                # Heuristic: count set bits. 
                # This is a lower bound proxy for "number of unsatisfied goal predicates"
                # Since we can't iterate bits efficiently in Python without loop, 
                # we use bin().count('1') which is optimized in C.
                cost += bin(unsatisfied).count('1') * 5 # Approx cost per goal

        return cost
