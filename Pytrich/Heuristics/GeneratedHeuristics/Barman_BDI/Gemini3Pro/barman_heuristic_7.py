from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic7(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_2):
        super().__init__(barman_gemini3pro_heuristic_2)
        self.model = barman_gemini3pro_heuristic_2
        self.facts_map = {}
        self.shakers = []
        self.cocktail_recipes = {}
        
        # Precompute fact indices and static domain knowledge
        for fact in self.model.facts:
            self.facts_map[fact.name] = fact.global_id
            
            # Parse fact name to extract static relationships
            name = fact.name.strip()
            if name.startswith("+"):
                name = name[1:]
            
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                pred = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",")
                
                if pred == "shakerLevel":
                    if args[0] not in self.shakers:
                        self.shakers.append(args[0])
                elif pred == "cocktailPart1":
                    c, i = args[0], args[1]
                    if c not in self.cocktail_recipes:
                        self.cocktail_recipes[c] = []
                    self.cocktail_recipes[c].append(i)
                elif pred == "cocktailPart2":
                    c, i = args[0], args[1]
                    if c not in self.cocktail_recipes:
                        self.cocktail_recipes[c] = []
                    self.cocktail_recipes[c].append(i)

    def _get_fid(self, pred, args):
        key = f"+{pred}[{','.join(args)}]"
        return self.facts_map.get(key, -1)

    def _is_true(self, state, pred, args):
        fid = self._get_fid(pred, args)
        if fid == -1:
            return False
        return (state >> fid) & 1

    def h(self, node):
        state = node.state
        tn = node.tn
        h_val = 0
        
        if not tn:
            return 0

        for task in tn:
            # Parse grounded task names: e.g., "AchieveContainsShotCocktail shot1 cocktail1"
            # Handle both space and underscore delimiters for robustness
            parts = task.name.split()
            if len(parts) == 1:
                parts = task.name.split('_')
            
            tname = parts[0]
            args = parts[1:]
            
            cost = 1 # Default cost for primitive actions or unknown tasks
            
            if "AchieveContainsShotCocktail" in tname and len(args) >= 2:
                cost = self._cost_shot_cocktail(state, args[0], args[1])
            elif "AchieveContainsShakerCocktail" in tname and len(args) >= 2:
                cost = self._cost_shaker_cocktail(state, args[0], args[1])
            elif "AchieveCleanShot" in tname and len(args) >= 1:
                cost = self._cost_clean(state, args[0], "shot")
            elif "AchieveCleanShaker" in tname and len(args) >= 1:
                cost = self._cost_clean(state, args[0], "shaker")
            elif "AchieveContainsShakerIngredient" in tname and len(args) >= 2:
                cost = self._cost_shaker_ing(state, args[0], args[1])
            elif "AchieveContainsShotIngredient" in tname and len(args) >= 2:
                cost = self._cost_shot_ing(state, args[0], args[1])
            elif "DoPourShakerToShot" in tname:
                cost = 1
            
            h_val += cost
            
        return h_val

    def _cost_clean(self, state, obj, otype):
        # Estimate cost to clean a container
        if self._is_true(state, "clean", [obj]):
            return 0
        # If not clean, we need: Grasp + (Empty if full) + Clean + Drop/Hold
        # Minimum cycle is roughly 3 steps
        return 3

    def _cost_shot_ing(self, state, shot, ing):
        # Estimate cost to get ingredient into a shot
        if self._is_true(state, "contains", [shot, ing]):
            return 0
        
        # Cost breakdown:
        # 1. Clean shot (if needed)
        # 2. Grasp shot
        # 3. Fill shot
        c_clean = self._cost_clean(state, shot, "shot")
        return c_clean + 2

    def _cost_shaker_ing(self, state, shaker, ing):
        # Estimate cost to get ingredient into shaker
        if self._is_true(state, "contains", [shaker, ing]):
            return 0
            
        # Logic: We need a shot containing the ingredient, then pour it.
        # Assume we use the 'best' shot (implicit in heuristic lower bound)
        # Cost: _cost_shot_ing + GraspShot + PourToShaker
        # Approx: 2 (fill) + 1 (grasp) + 1 (pour) = 4 steps minimum (assuming clean shot available)
        # If we include cleaning risk, bump to 6.
        return 6

    def _cost_shaker_cocktail(self, state, shaker, cocktail):
        # Estimate cost to get a finished cocktail in the shaker
        if self._is_true(state, "contains", [shaker, cocktail]):
            if self._is_true(state, "shaked", [shaker]):
                return 0
            if self._is_true(state, "unshaked", [shaker]):
                return 1 # Just shake
        
        # Full process:
        # 1. Clean shaker
        # 2. Add Ingredient 1 (via shot)
        # 3. Add Ingredient 2 (via shot)
        # 4. Shake
        c_clean = self._cost_clean(state, shaker, "shaker")
        c_ing1 = 6 # Approx cost to add one ingredient
        c_ing2 = 6
        return c_clean + c_ing1 + c_ing2 + 1

    def _cost_shot_cocktail(self, state, shot, cocktail):
        # Estimate cost to get cocktail into final shot
        if self._is_true(state, "contains", [shot, cocktail]):
            return 0
            
        # 1. Ensure target shot is clean
        c_clean = self._cost_clean(state, shot, "shot")
        
        # 2. Ensure we have the cocktail in a shaker
        # Check if ANY shaker already has it ready to minimize estimate
        shaker_ready = False
        for s in self.shakers:
            if self._is_true(state, "contains", [s, cocktail]) and self._is_true(state, "shaked", [s]):
                shaker_ready = True
                break
        
        c_make = 1 if shaker_ready else 14 # Large penalty if we must make it from scratch
        
        # 3. Pour (1 step)
        return c_clean + c_make + 1
