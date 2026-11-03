from Pytrich.Heuristics.heuristic import Heuristic
from typing import List, Dict, Tuple, Set

class BarmanGemini3ProHeuristic24(Heuristic):
    def __init__(self, model, barman_gemini3pro_heuristic_19=None):
        super().__init__(model)
        self.model = model
        
        # Precompute Fact IDs for fast lookup
        # Key: tuple(predicate, arg1, arg2, ...)
        self.fact_map: Dict[Tuple[str, ...], int] = {}
        self.cocktail_recipes: Dict[str, List[str]] = {}
        self.shakers: List[str] = []
        
        # Primitive cost estimates (lower bounds)
        self.cost_grasp = 1
        self.cost_drop = 1
        self.cost_fill = 1
        self.cost_pour = 1
        self.cost_shake = 1
        self.cost_clean = 1
        self.cost_empty = 1

        # Parse facts to populate maps and extract static info
        for fact in self.model.facts:
            name = fact.name.strip()
            # Format: +pred[arg1,arg2] or -pred[...]
            # We only care about finding the global ID for positive facts
            if name.startswith("+") or name.startswith("-"):
                is_neg = name.startswith("-")
                clean_name = name[1:]
                
                if "[" in clean_name and clean_name.endswith("]"):
                    pred = clean_name[:clean_name.index("[")]
                    args_str = clean_name[clean_name.index("[")+1:-1]
                    args = tuple(args_str.split(",")) if args_str else ()
                    
                    if not is_neg:
                        self.fact_map[(pred,) + args] = fact.global_id
                        
                        # Extract static domain info
                        if pred == "cocktailPart1":
                            c, i = args
                            if c not in self.cocktail_recipes: self.cocktail_recipes[c] = []
                            self.cocktail_recipes[c].append(i)
                        elif pred == "cocktailPart2":
                            c, i = args
                            if c not in self.cocktail_recipes: self.cocktail_recipes[c] = []
                            self.cocktail_recipes[c].append(i)
                        elif pred == "shakerLevel":
                            # Identifies shakers
                            if args[0] not in self.shakers:
                                self.shakers.append(args[0])
                        elif pred == "clean" and "shaker" in args[0]:
                            # Backup identification
                            if args[0] not in self.shakers:
                                self.shakers.append(args[0])
                                
        self.shakers = list(set(self.shakers))

    def _is_true(self, state_bitwise, pred_tuple):
        """Check if a fact is true in the current state bitmask."""
        gid = self.fact_map.get(pred_tuple)
        if gid is None: return False
        return (state_bitwise >> gid) & 1

    def compute(self, state, task_network):
        """Estimate cost to refine the current task network to a primitive plan."""
        h_val = 0
        
        for task in task_network:
            # If task is an Operator, it is a primitive action in the TN
            if hasattr(task, 'add_effects'):
                h_val += 1
                continue
                
            # Parse Abstract Task name
            parts = task.name.split(" ")
            task_name = parts[0]
            args = parts[1:]
            
            if task_name == "AchieveContainsShotCocktail":
                h_val += self._est_contains_shot_cocktail(state, args[0], args[1])
            elif task_name == "AchieveCleanShot":
                h_val += self._est_clean_shot(state, args[0])
            elif task_name == "AchieveCleanShaker":
                h_val += self._est_clean_shaker(state, args[0])
            elif task_name == "AchieveContainsShakerCocktail":
                h_val += self._est_contains_shaker_cocktail(state, args[0], args[1])
            elif task_name == "AchieveContainsShotIngredient":
                h_val += self._est_contains_shot_ingredient(state, args[0], args[1])
            elif task_name == "AchieveContainsShakerIngredient":
                h_val += self._est_contains_shaker_ingredient(state, args[0], args[1])
            elif task_name == "AchieveHandEmpty":
                h_val += self._est_hand_empty(state, args[0])
            elif task_name == "AchieveHolding":
                h_val += self._est_holding(state, args[0], args[1])
            elif task_name == "DoPourShakerToShot":
                h_val += 1
            else:
                # Default cost for unknown or intermediate tasks
                h_val += 1
                
        return h_val

    def _est_clean_shot(self, state, shot):
        if self._is_true(state, ("clean", shot)):
            return 0
        if self._is_true(state, ("empty", shot)):
            return self.cost_clean
        # Needs to be emptied then cleaned
        return self.cost_empty + self.cost_clean

    def _est_clean_shaker(self, state, shaker):
        if self._is_true(state, ("clean", shaker)):
            return 0
        if self._is_true(state, ("empty", shaker)):
            return self.cost_clean
        # Needs to be emptied then cleaned
        return self.cost_empty + self.cost_clean

    def _est_hand_empty(self, state, hand):
        if self._is_true(state, ("handEmpty", hand)):
            return 0
        return self.cost_drop

    def _est_holding(self, state, hand, obj):
        if self._is_true(state, ("holding", hand, obj)):
            return 0
        # Cost to grasp
        cost = self.cost_grasp
        # If hand not empty, must drop current item
        if not self._is_true(state, ("handEmpty", hand)):
            cost += self.cost_drop
        return cost

    def _est_contains_shot_ingredient(self, state, shot, ing):
        if self._is_true(state, ("contains", shot, ing)):
            return 0
        # Cost breakdown: Clean Shot + Grasp + Fill
        cost = self._est_clean_shot(state, shot)
        cost += self.cost_fill
        cost += self.cost_grasp
        return cost

    def _est_contains_shaker_ingredient(self, state, shaker, ing):
        if self._is_true(state, ("contains", shaker, ing)):
            return 0
        # To add ingredient to shaker: Fill a shot, Pour shot to shaker
        # Approx Cost: Grasp Shot + Fill + Pour
        return self.cost_grasp + self.cost_fill + self.cost_pour

    def _est_contains_shaker_cocktail(self, state, shaker, cocktail):
        # Goal: Shaker contains cocktail AND is shaked
        if self._is_true(state, ("contains", shaker, cocktail)) and self._is_true(state, ("shaked", shaker)):
            return 0
            
        # Base cost: Shake + Hold (needed for shaking)
        cost = self.cost_shake + self.cost_grasp
        
        # Check if ingredients are present
        needed = self.cocktail_recipes.get(cocktail, [])
        for ing in needed:
            if not self._is_true(state, ("contains", shaker, ing)):
                cost += self._est_contains_shaker_ingredient(state, shaker, ing)
                
        # Check if shaker needs cleaning (dirty or contains wrong stuff)
        is_clean = self._is_true(state, ("clean", shaker))
        is_empty = self._is_true(state, ("empty", shaker))
        
        if not is_clean and not is_empty:
             # Check if it contains partial useful ingredients. 
             # If it contains stuff not in recipe, it needs cleaning.
             # Simplified check: if it contains ANY needed ingredient, assume we don't clean.
             has_part = False
             for ing in needed:
                 if self._is_true(state, ("contains", shaker, ing)):
                     has_part = True
                     break
             if not has_part:
                 cost += self._est_clean_shaker(state, shaker)
                 
        return cost

    def _est_contains_shot_cocktail(self, state, shot, cocktail):
        if self._is_true(state, ("contains", shot, cocktail)):
            return 0
            
        # Strategy: 
        # 1. Prepare Cocktail in best available shaker
        # 2. Clean Shot
        # 3. Pour
        
        best_shaker_cost = 9999
        if not self.shakers:
            best_shaker_cost = 5 # Fallback
        else:
            # Find the cheapest shaker to use
            for s in self.shakers:
                c = self._est_contains_shaker_cocktail(state, s, cocktail)
                if c < best_shaker_cost:
                    best_shaker_cost = c
            
        cost = best_shaker_cost
        cost += self._est_clean_shot(state, shot)
        cost += self.cost_pour
        cost += self.cost_grasp # Need to hold shaker to pour
        
        return cost
