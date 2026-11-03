import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini20FlashHeuristic25(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_25=1.0, name="BarmanGemini20FlashHeuristic25"): # All parameters lowercase
        super().__init__(name=name)
        self.clean_shot_cost = barman_gemini20flash_heuristic_25
        self.add_ingredient_cost = barman_gemini20flash_heuristic_25
        self.make_cocktail_cost = barman_gemini20flash_heuristic_25
        self.pour_cost = barman_gemini20flash_heuristic_25
        self.hand_manipulation_cost = barman_gemini20flash_heuristic_25
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        initial_h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        h_value = 0

        # Check for null nodes
        if node is None or not hasattr(node, 'method') or node.method is None:
            return 0  # No cost for null nodes

        method = node.method
        method_name = method.name if hasattr(method, 'name') else str(method)

        # Prioritize achieving the main goal: cocktail in shot
        if "AchieveContainsShotCocktail" in method_name:
            h_value += self.make_cocktail_cost  # Making a cocktail contributes to the goal

        # Consider the costs for cleaning, adding ingredients and pouring
        if "AchieveCleanShot" in method_name or "clean-shot" in method_name:
            h_value += self.clean_shot_cost # Cleaning is required before adding ingredients
        if "AchieveCleanShaker" in method_name or "clean-shaker" in method_name:
            h_value += self.clean_shot_cost  # Cleaning is required before adding ingredients

        if "AchieveContainsShotIngredient" in method_name or  "AddIngredientToShot" in method_name or "fill-shot" in method_name:
            h_value += self.add_ingredient_cost # Adding ingredients to a shot
        if "AchieveContainsShakerIngredient" in method_name:
            h_value += self.add_ingredient_cost  # Adding ingredients to shaker

        if "DoPourShakerToShot" in method_name or "pour-shaker-to-shot" in method_name:
            h_value += self.pour_cost  # Pouring is important to achieve cocktail in shot

        # Factor in hand manipulations (picking up, dropping)
        if "AchieveHolding" in method_name or "PickUp" in method_name or "EmptyHand" in method_name or "drop" in method_name or "grasp" in method_name:
            h_value += self.hand_manipulation_cost / 2.0 # Picking up/dropping containers

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic(clean={self.clean_shot_cost}, add={self.add_ingredient_cost}, make={self.make_cocktail_cost}, pour={self.pour_cost}, hand={self.hand_manipulation_cost})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic(clean={self.clean_shot_cost}, add={self.add_ingredient_cost}, make={self.make_cocktail_cost}, pour={self.pour_cost}, hand={self.hand_manipulation_cost})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 25 info:\n"
            f"\tName: {self.name}\n"
            f"\tClean Shot Cost: {self.clean_shot_cost}\n"
            f"\tAdd Ingredient Cost: {self.add_ingredient_cost}\n"
            f"\tMake Cocktail Cost: {self.make_cocktail_cost}\n"
            f"\tPour Cost: {self.pour_cost}\n"
            f"\tHand Manipulation Cost: {self.hand_manipulation_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
