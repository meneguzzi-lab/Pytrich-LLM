from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini3ProHeuristic6(Heuristic):
    def __init__(self, barman_gemini3pro_heuristic_1, model):
        super().__init__(barman_gemini3pro_heuristic_1, model)
        self.model = model
        self.facts_map = {}  # predicate -> args -> global_id
        self.cocktail_parts = {}  # cocktail -> {'p1': ing, 'p2': ing}
        self.shakers = set()

        # Parse facts to build efficient lookup tables
        # Facts are strings like "+contains[shot1,cocktail1]"
        for fact in self.model.facts:
            name = fact.name
            # Skip negative facts for indexing state features
            if name.startswith("-"):
                continue
                
            # Clean name: remove + prefix
            clean_name = name[1:] if name.startswith("+") else name
            
            # Parse predicate and arguments
            if "[" in clean_name and clean_name.endswith("]"):
                pred_end = clean_name.find("[")
                predicate = clean_name[:pred_end]
                args_str = clean_name[pred_end+1:-1]
                args = tuple(args_str.split(",")) if args_str else ()
                
                # Store mapping
                if predicate not in self.facts_map:
                    self.facts_map[predicate] = {}
                self.facts_map[predicate][args] = fact.global_id
                
                # Domain-specific static analysis for recipes and objects
                if predicate == "cocktailPart1":
                    c, i = args
                    if c not in self.cocktail_parts:
                        self.cocktail_parts[c] = {}
                    self.cocktail_parts[c]["p1"] = i
                elif predicate == "cocktailPart2":
                    c, i = args
                    if c not in self.cocktail_parts:
                        self.cocktail_parts[c] = {}
                    self.cocktail_parts[c]["p2"] = i
                elif predicate == "shakerLevel":
                    # Objects involved in shakerLevel are shakers
                    self.shakers.add(args[0])

        self.shakers = list(self.shakers)

    def _get_fact_id(self, pred, args):
        return self.facts_map.get(pred, {}).get(args)

    def _is_true(self, state, pred, args):
        """Fast bitwise check for a fact in the state."""
        fid = self._get_fact_id(pred, args)
        if fid is None:
            return False
        return (state >> fid) & 1

    def compute(self, state, tn):
        """
        Estimate cost-to-go based on the current task network (tn).
        Iterates through tasks in the TN and sums their estimated costs.
        """
        h_val = 0
        
        for task in tn:
            # If it is a primitive Operator, add its base cost (usually 1)
            if hasattr(task, "add_effects"):
                h_val += 1
                continue
            
            # AbstractTask: Parse name to determine type and arguments
            # Name format example: "AchieveContainsShotCocktail shot1 cocktail1"
            parts = task.name.split()
            if not parts: 
                continue
                
            task_type = parts[0]
            args = tuple(parts[1:])
            
            if task_type == "AchieveContainsShotCocktail":
                h_val += self._h_shot_cocktail(state, args[0], args[1])
            elif task_type == "AchieveCleanShot":
                h_val += self._h_clean_shot(state, args[0])
            elif task_type == "AchieveCleanShaker":
                h_val += self._h_clean_shaker(state, args[0])
            elif task_type == "AchieveContainsShakerCocktail":
                h_val += self._h_shaker_cocktail(state, args[0], args[1])
            elif task_type == "AchieveContainsShakerIngredient":
                h_val += self._h_shaker_ingredient(state, args[0], args[1])
            elif task_type == "AchieveContainsShotIngredient":
                h_val += self._h_shot_ingredient(state, args[0], args[1])
            elif task_type == "AchieveHolding":
                h_val += self._h_holding(state, args[0], args[1])
            elif task_type == "DoPourShakerToShot":
                # This is a specific subtask usually mapping to 1 action
                h_val += 1 
            elif task_type == "AchieveHandEmpty":
                if not self._is_true(state, "handEmpty", (args[0],)):
                    h_val += 1
            elif task_type == "AchieveOnTable":
                if not self._is_true(state, "ontable", (args[0],)):
                    h_val += 1
            else:
                # Default small penalty for unknown tasks to avoid 0 cost loops
                h_val += 1
                
        return h_val

    def _h_shot_cocktail(self, state, shot, cocktail):
        """Estimate cost to get a cocktail into a shot."""
        if self._is_true(state, "contains", (shot, cocktail)):
            return 0
        
        # Case 1: Cocktail is already made in a shaker
        for shaker in self.shakers:
            if self._is_true(state, "contains", (shaker, cocktail)):
                # Found cocktail. Is it shaken?
                if self._is_true(state, "shaked", (shaker,)):
                    # Ready to pour. Cost = Pour (1) + potential setup
                    cost = 1
                    # If destination shot is not ready, add cost
                    if not self._is_true(state, "clean", (shot,)):
                        cost += 2 # Clean/Empty shot actions
                    return cost
                else:
                    # Needs shaking then pouring
                    # Cost = Shake (1) + Pour (1) + Holding/Setup (~1)
                    return 3
        
        # Case 2: Cocktail is not made. Needs full cycle.
        # Cycle: Clean Shaker -> Add Ing1 -> Add Ing2 -> Shake -> Pour
        # Add Ing involves: Grasp Shot, Fill, Pour, Drop (approx 4 steps)
        # Total: 3 (Clean) + 4 (Ing1) + 4 (Ing2) + 1 (Shake) + 1 (Pour) = ~13
        cost = 12 # slightly optimistic lower bound
        
        # If destination shot is dirty, add cost
        if not self._is_true(state, "clean", (shot,)):
            cost += 2
            
        return cost

    def _h_shaker_cocktail(self, state, shaker, cocktail):
        """Estimate cost to get a cocktail into a shaker."""
        if self._is_true(state, "contains", (shaker, cocktail)):
            if self._is_true(state, "shaked", (shaker,)):
                return 0
            return 1 # Just needs shaking
        
        # Needs ingredients adding. 
        # 2 ingredients * ~4 steps each = 8 steps
        return 8 

    def _h_clean_shot(self, state, shot):
        if self._is_true(state, "clean", (shot,)):
            return 0
        if self._is_true(state, "empty", (shot,)):
            return 2 # Grasp + Clean
        return 3 # Grasp + Empty + Clean

    def _h_clean_shaker(self, state, shaker):
        if self._is_true(state, "clean", (shaker,)):
            return 0
        if self._is_true(state, "empty", (shaker,)):
            return 2 # Grasp + Clean
        return 3 # Grasp + Empty + Clean

    def _h_holding(self, state, hand, obj):
        if self._is_true(state, "holding", (hand, obj)):
            return 0
        return 1 # Grasp action

    def _h_shaker_ingredient(self, state, shaker, ing):
        if self._is_true(state, "contains", (shaker, ing)):
            return 0
        # Sequence: Grasp Shot -> Fill -> Pour -> Drop
        return 4 

    def _h_shot_ingredient(self, state, shot, ing):
        if self._is_true(state, "contains", (shot, ing)):
            return 0
        # Sequence: Clean Shot (if needed) -> Grasp -> Fill
        cost = 2 # Grasp + Fill
        if not self._is_true(state, "clean", (shot,)):
            cost += 1 # Cleaning step
        return cost
