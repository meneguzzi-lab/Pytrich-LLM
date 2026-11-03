import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic7(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_7=1.0, name="BarmanGemini20FlashHeuristic7"):
        super().__init__(name=name)
        self.clean_shot_cost = barman_gemini20flash_heuristic_7
        self.add_ingredient_cost = barman_gemini20flash_heuristic_7
        self.make_cocktail_cost = barman_gemini20flash_heuristic_7
        self.pour_cost = barman_gemini20flash_heuristic_7
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        self.method_scores = {}
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = {
                'is_make_cocktail': 'MakeCocktail' in method_name,
                'is_add_ingredient': 'AddIngredient' in method_name,
                'is_clean_shot': 'CleanShot' in method_name,
                'is_pour': 'Pour' in method_name
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on its name.
        """
        method_name = method.name if hasattr(method, 'name') else str(method)
        if method_name in self.method_scores:
            method_info = self.method_scores[method_name]
            if method_info['is_clean_shot']:
                return self.clean_shot_cost
            elif method_info['is_add_ingredient']:
                return self.add_ingredient_cost
            elif method_info['is_make_cocktail']:
                return self.make_cocktail_cost
            elif method_info['is_pour']:
                return self.pour_cost
        return 0

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic7(clean_shot_cost={self.clean_shot_cost}, add_ingredient_cost={self.add_ingredient_cost}, " \
               f"make_cocktail_cost={self.make_cocktail_cost}, pour_cost={self.pour_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 7 info:\n"
            f"\tName: {self.name}\n"
            f"\tClean Shot Cost: {self.clean_shot_cost}\n"
            f"\tAdd Ingredient Cost: {self.add_ingredient_cost}\n"
            f"\tMake Cocktail Cost: {self.make_cocktail_cost}\n"
            f"\tPour Cost: {self.pour_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
