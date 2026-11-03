import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanBDIGemini3FlashHeuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.
    
    Bottleneck Analysis:
    The primary bottleneck is the Sequential Shaker Cycle. Each cocktail requires a strict sequence: 
    Cleaning Shaker -> Filling Ingredient 1 -> Filling Ingredient 2 -> Shaking -> Pouring to Shot.
    Since there is usually only one shaker, these cycles must happen serially. Hands also act as 
    resources for cleaning, grasping, and shaking.

    Admissibility Sketches:
    Term 1: Cocktail Action Progress. For every cocktail goal not met, we count the minimum number of 
    remaining primitive actions (pour=1, shake=1, fill-ingredient=2) required to complete it. 
    This is additive because the shaker must be emptied and processed per cocktail.
    Term 2: Cleaning Requirements. Every container (shot or shaker) that needs to receive a beverage 
    but is currently dirty or full of something else must undergo a cleaning action (1-2 steps). 
    This is a distinct lower bound on necessary work.
    """

    def __init__(self, barman_gemini3flash_heuristic_3=None, name="BarmanBDIGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.recipe = {}  # cocktail -> (ing1, ing2)
        self.contains_ids = {} # (container, beverage) -> global_id
        self.clean_ids = {} # container -> global_id
        self.shaker_id = None
        self.shaker_name = ""
        self.goal_cocktails = [] # list of (shot, cocktail)

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse Recipes and Map Fact IDs
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            
            if pred == "cocktailPart1":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.recipe: self.recipe[cocktail] = [None, None]
                self.recipe[cocktail][0] = ing
            elif pred == "cocktailPart2":
                cocktail, ing = args[0], args[1]
                if cocktail not in self.recipe: self.recipe[cocktail] = [None, None]
                self.recipe[cocktail][1] = ing
            elif pred == "contains":
                self.contains_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clean":
                self.clean_ids[args[0]] = fact.global_id
            
            # Identify the shaker (heuristic assumes a primary shaker for logic)
            if "shaker" in fact.name.lower() and not self.shaker_id:
                # This is heuristic, usually shaker1 is the only shaker
                for arg in args:
                    if "shaker" in arg.lower():
                        self.shaker_name = arg
                        break

        # 2. Parse Goals from model.goals bitmask
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "contains" and len(args) == 2:
                    # Usually Goal is (contains shot cocktail)
                    self.goal_cocktails.append((args[0], args[1]))

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts: return parts[0], parts[1:]
        return None, []

    def _parse_task_name(self, name: str):
        # Task network names: "TaskName[arg1,arg2]" or "TaskName arg1 arg2"
        name = name.strip()
        if "[" in name:
            bracket_idx = name.find("[")
            task_name = name[:bracket_idx]
            args = name[bracket_idx+1:-1].split(",")
            return task_name, args
        else:
            parts = name.split()
            if parts: return parts[0], parts[1:]
        return "", []

    def _compute(self, state, task_network):
        total_h = 0
        
        # Term 1: Analyze Tasks in the Network
        # Tasks represent explicit work to be done.
        for task in task_network:
            t_name, args = self._parse_task_name(task.name)
            
            if t_name == "AchieveContainsShotCocktail":
                shot, cocktail = args[0], args[1]
                # Check if already done in state
                f_id = self.contains_ids.get((shot, cocktail))
                if f_id is not None and (state >> f_id) & 1:
                    continue
                
                # Minimum sequence: pour(1) + shake(1) + 2*fill(2) = 6
                cost = 1 # pour
                # Check if cocktail is already in shaker
                sh_c_id = self.contains_ids.get((self.shaker_name, cocktail))
                if not (sh_c_id is not None and (state >> sh_c_id) & 1):
                    cost += 1 # shake
                    # Check ingredients in shaker
                    if cocktail in self.recipe:
                        for ing in self.recipe[cocktail]:
                            sh_i_id = self.contains_ids.get((self.shaker_name, ing))
                            if not (sh_i_id is not None and (state >> sh_i_id) & 1):
                                cost += 2 # fill + pour
                total_h += cost

            elif t_name == "AchieveContainsShakerCocktail":
                shaker, cocktail = args[0], args[1]
                sh_c_id = self.contains_ids.get((shaker, cocktail))
                if not (sh_c_id is not None and (state >> sh_c_id) & 1):
                    cost = 1 # shake
                    if cocktail in self.recipe:
                        for ing in self.recipe[cocktail]:
                            sh_i_id = self.contains_ids.get((shaker, ing))
                            if not (sh_i_id is not None and (state >> sh_i_id) & 1):
                                cost += 2
                    total_h += cost

            elif t_name == "AchieveContainsShakerIngredient":
                shaker, ing = args[0], args[1]
                sh_i_id = self.contains_ids.get((shaker, ing))
                if not (sh_i_id is not None and (state >> sh_i_id) & 1):
                    total_h += 2

            elif t_name == "AchieveCleanShot" or t_name == "AchieveCleanShaker":
                cont = args[0]
                cl_id = self.clean_ids.get(cont)
                if not (cl_id is not None and (state >> cl_id) & 1):
                    total_h += 1

        # Term 2: Static goal fact check (Goal Stability)
        # If the task network is empty but goals aren't reached, count them.
        if not task_network:
            for shot, cocktail in self.goal_cocktails:
                f_id = self.contains_ids.get((shot, cocktail))
                if f_id is not None and not ((state >> f_id) & 1):
                    total_h += 6 # Minimum cycle cost

        # Small tie-breaker for holding/cleanliness
        # Prefers states where containers are clean or hands are free
        clean_bonus = 0
        for cont, c_id in self.clean_ids.items():
            if (state >> c_id) & 1:
                clean_bonus -= 0.01
        
        return max(0, total_h) + clean_bonus

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
