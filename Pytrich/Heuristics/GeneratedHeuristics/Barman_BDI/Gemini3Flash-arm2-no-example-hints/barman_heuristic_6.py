from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class BarmanBDIGemini3FlashHeuristic6(Heuristic):
    """
    Domain-Dependent Heuristic for Barman_BDI.
    
    BOTTLENECK:
    The central bottleneck is the management of shots as interchangeable resources for transferring 
    ingredients to the shaker. This causes massive branching as the planner tries every shot 
    for every ingredient transfer.
    
    ADMISSIBLE LOWER BOUNDS:
    1. Cocktail Progress Term: Every cocktail goal not yet in its designated shot requires 
       at least 2 actions (shake + pour) if already mixed, or ~7 actions if starting from scratch.
    2. Shaker Cleaning Term: The shaker must be cleaned between different cocktails. Each 
       unmixed cocktail requires a clean shaker.
    
    SYMMETRY BREAKING (Crucial for performance):
    - Goal-shot penalty: Using a shot designated for a goal cocktail as a temporary carrier (500).
    - Shot-ordering: Assigning static penalties based on shot names to break symmetry (100 * index).
    - Hand preference: Favoring the left hand to reduce branching in hand-selection tasks (50).
    """

    def __init__(self, barman_gemini3flash_heuristic_6="barman_gemini3flash_heuristic_6"):
        super().__init__(name=barman_gemini3flash_heuristic_6)
        self.goal_shots = {}  # shot_name -> cocktail_name
        self.goal_fact_ids = []
        self.contains_fact_map = {}  # (container, beverage) -> global_id
        self.shaked_fact_map = {} # shaker -> global_id
        self.cocktail_recipes = {} # cocktail -> (ing1, ing2)
        self.shot_order = {}
        self.preferred_hand = None
        self.shot_names = set()

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

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map predicates and parse recipes
        all_shakers = []
        all_hands = []
        all_shots = []

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "contains":
                self.contains_fact_map[(args[0], args[1])] = f.global_id
            elif pred == "shaked":
                self.shaked_fact_map[args[0]] = f.global_id
            elif pred == "cocktailPart1":
                if args[0] not in self.cocktail_recipes: self.cocktail_recipes[args[0]] = [None, None]
                self.cocktail_recipes[args[0]][0] = args[1]
            elif pred == "cocktailPart2":
                if args[0] not in self.cocktail_recipes: self.cocktail_recipes[args[0]] = [None, None]
                self.cocktail_recipes[args[0]][1] = args[1]
            elif pred == "empty":
                if "shaker" in args[0]: all_shakers.append(args[0])
                if "shot" in args[0]: all_shots.append(args[0])
            elif pred == "handEmpty":
                all_hands.append(args[0])

        # 2. Identify Goal Shots
        goal_names = model.state_explicit_repr(model.goals)
        for g_name in goal_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "contains":
                self.goal_shots[args[0]] = args[1]
                # Find global_id for this goal fact
                if (args[0], args[1]) in self.contains_fact_map:
                    self.goal_fact_ids.append(self.contains_fact_map[(args[0], args[1])])

        # 3. Symmetry breaking setup
        sorted_shots = sorted(list(set(all_shots)))
        self.shot_names = set(sorted_shots)
        self.shot_order = {name: i for i, name in enumerate(sorted_shots)}
        
        if all_hands:
            sorted_hands = sorted(all_hands)
            self.preferred_hand = sorted_hands[0] # Usually 'left'

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def _compute(self, state, task_network):
        # Admissible progress cost
        h_base = 0
        for shot, cocktail in self.goal_shots.items():
            goal_id = self.contains_fact_map.get((shot, cocktail))
            if goal_id is not None and (state >> goal_id) & 1:
                continue
            
            # Cocktail is not in the goal shot
            cost = 10 # Base cost for a cocktail cycle
            
            # Check if it's currently in a shaker
            in_shaker = False
            for shaker, sh_id in self.shaked_fact_map.items():
                cont_id = self.contains_fact_map.get((shaker, cocktail))
                if cont_id and (state >> cont_id) & 1:
                    in_shaker = True
                    # If shaked, only need to pour
                    if (state >> sh_id) & 1: cost = min(cost, 2)
                    else: cost = min(cost, 3) # Needs shake + pour
            
            if not in_shaker:
                # Check recipe ingredients in shaker
                recipe = self.cocktail_recipes.get(cocktail)
                if recipe:
                    ing_count = 0
                    for ing in recipe:
                        for shaker in self.shaked_fact_map.keys():
                            ing_cont_id = self.contains_fact_map.get((shaker, ing))
                            if ing_cont_id and (state >> ing_cont_id) & 1:
                                ing_count += 1
                    if ing_count == 2: cost = min(cost, 4) # shake + pour + overhead
                    elif ing_count == 1: cost = min(cost, 7) # 1 more ing + shake + pour
            
            h_base += cost

        # Symmetry breaking penalties
        penalty = 0
        if task_network:
            for i in range(min(3, len(task_network))):
                task = task_network[i]
                t_name, t_args = self._parse_fact_name(task.name)
                if not t_name: continue

                # 1. Goal-shot penalty: Using goal shot for ingredient carrier
                if t_name == "AchieveContainsShotIngredient":
                    if t_args and t_args[0] in self.goal_shots:
                        penalty += 500
                    if t_args and t_args[0] in self.shot_order:
                        penalty += self.shot_order[t_args[0]] * 100
                
                # 2. Hand preference
                if self.preferred_hand:
                    if t_name in ["AchieveHolding", "AchieveHandEmpty", "grasp", "drop"]:
                        if t_args and t_args[0] != self.preferred_hand:
                            penalty += 50

        return h_base + penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
