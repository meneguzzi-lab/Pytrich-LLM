from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class BarmanBDIGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Barman_BDI domain.

    Bottleneck Analysis:
    The primary bottleneck in this domain is the single (or limited number of) shaker(s).
    Each cocktail requires a sequential set of steps (clean shaker, fill with two ingredients,
    shake, and pour) that cannot overlap within the same shaker. Because ingredients for 
    one cocktail cannot be mixed with another's in the shaker, the cocktails must be 
    prepared largely one by one.

    Admissibility Sketch:
    1. Cocktail Completion (Summative): The total cost is at least the sum of the remaining 
       actions needed for each goal cocktail. Since a shaker can only process one cocktail 
       at a time, the actions are serial. We estimate the minimum primitive actions remaining 
       for each cocktail (e.g., 2 for pour, 4 for shake+pour, 7 for fill+fill+shake+pour).
    2. Preparation Costs: Containers (shakers/shots) must be clean and empty before use. 
       If a container is dirty or full of the wrong liquid, at least 1-2 actions are required 
       to prepare it.
    3. Hand/Grasp Costs: Every operation (filling, pouring, shaking) requires holding the 
       container. If hands are full with irrelevant objects, drop actions are mandatory.
    """

    def __init__(self, barman_gemini3flash_heuristic_18=None, name="BarmanBDIGemini3FlashHeuristic18"):
        super().__init__(name=name)
        self.goal_cocktails = []  # List of (shot, cocktail)
        self.cocktail_ingredients = {}  # cocktail -> (ing1, ing2)
        self.contains_ids = {}  # (container, beverage) -> global_id
        self.clean_ids = {}  # container -> global_id
        self.shaked_ids = {}  # shaker -> global_id
        self.unshaked_ids = {}  # shaker -> global_id
        self.empty_ids = {}  # container -> global_id
        self.hand_empty_ids = {}  # hand -> global_id
        self.shakers = []
        self.hands = []
        self.shots = []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for gname in goal_fact_names:
                pred, args = self._parse_fact_name(gname)
                if pred == "contains" and len(args) == 2:
                    self.goal_cocktails.append((args[0], args[1]))

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "contains":
                self.contains_ids[(args[0], args[1])] = fact.global_id
            elif pred == "clean":
                self.clean_ids[args[0]] = fact.global_id
            elif pred == "shaked":
                self.shaked_ids[args[0]] = fact.global_id
                if args[0] not in self.shakers: self.shakers.append(args[0])
            elif pred == "unshaked":
                self.unshaked_ids[args[0]] = fact.global_id
                if args[0] not in self.shakers: self.shakers.append(args[0])
            elif pred == "empty":
                self.empty_ids[args[0]] = fact.global_id
            elif pred == "handEmpty":
                self.hand_empty_ids[args[0]] = fact.global_id
                if args[0] not in self.hands: self.hands.append(args[0])
            elif pred == "cocktailPart1":
                if args[0] not in self.cocktail_ingredients: self.cocktail_ingredients[args[0]] = [None, None]
                self.cocktail_ingredients[args[0]][0] = args[1]
            elif pred == "cocktailPart2":
                if args[0] not in self.cocktail_ingredients: self.cocktail_ingredients[args[0]] = [None, None]
                self.cocktail_ingredients[args[0]][1] = args[1]

        h_value = self._compute(initial_node.state)
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
        return None, []

    def _is_true(self, state, fact_id):
        if fact_id is None:
            return False
        return (state >> fact_id) & 1

    def _compute(self, state):
        total_h = 0
        shaker = self.shakers[0] if self.shakers else None
        if not shaker:
            return 0

        for target_shot, cocktail in self.goal_cocktails:
            if self._is_true(state, self.contains_ids.get((target_shot, cocktail))):
                continue

            cocktail_cost = 0
            is_in_shaker = self._is_true(state, self.contains_ids.get((shaker, cocktail)))
            is_shaked = self._is_true(state, self.shaked_ids.get(shaker))

            if is_in_shaker and is_shaked:
                cocktail_cost = 2
            else:
                ing1, ing2 = self.cocktail_ingredients.get(cocktail, (None, None))
                has_ing1 = self._is_true(state, self.contains_ids.get((shaker, ing1)))
                has_ing2 = self._is_true(state, self.contains_ids.get((shaker, ing2)))
                if has_ing1 and has_ing2:
                    cocktail_cost = 4
                elif has_ing1 or has_ing2:
                    cocktail_cost = 7
                else:
                    cocktail_cost = 10

            if not self._is_true(state, self.clean_ids.get(target_shot)):
                cocktail_cost += 2
            total_h += cocktail_cost

        num_busy_hands = 0
        for hand in self.hands:
            if not self._is_true(state, self.hand_empty_ids.get(hand)):
                num_busy_hands += 1
        if num_busy_hands == len(self.hands) and total_h > 0:
            total_h += 1

        return total_h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
