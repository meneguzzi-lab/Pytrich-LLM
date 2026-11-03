import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic24(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_24=1.0, name="BarmanGemini20FlashHeuristic24"):
        super().__init__(name=name)
        self.achieve_contains_shot_cocktail_penalty = barman_gemini20flash_heuristic_24  # Higher value penalizes more
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
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
            is_make_and_pour = 'MakeAndPourCocktail' in method_name
            is_make_cocktail = 'MakeCocktail' in method_name
            is_add_ingredient = 'AddIngredientTo' in method_name
            is_clean = 'Clean' in method_name
            is_pickup = 'PickUp' in method_name
            is_empty_hand = 'EmptyHand' in method_name
            is_put_down = 'PutDown' in method_name
            is_pour_shaker = 'pour_shaker_to_shot_action' in method_name
            self.method_scores[method_name] = {
                'is_make_and_pour': is_make_and_pour,
                'is_make_cocktail': is_make_cocktail,
                'is_add_ingredient': is_add_ingredient,
                'is_clean': is_clean,
                'is_pickup': is_pickup,
                'is_empty_hand': is_empty_hand,
                'is_put_down': is_put_down,
                'is_pour_shaker': is_pour_shaker
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            if hasattr(node, 'task') and str(node.task).startswith("AchieveContainsShotCocktail"): #str cause task is a Task instance
              return self.achieve_contains_shot_cocktail_penalty
            return 0

        method = node.method
        method_name = method.name if hasattr(method, 'name') else str(method)
        score = 0

        if method_name in self.method_scores:
            method_info = self.method_scores[method_name]

            if method_info['is_make_and_pour']:
                score += 3
            elif method_info['is_make_cocktail']:
                score += 4
            elif method_info['is_add_ingredient']:
                score += 2
            elif method_info['is_clean']:
                score += 1
            elif method_info['is_pickup']:
                score += 0.5
            elif method_info['is_empty_hand']:
                score += 0.5
            elif method_info['is_put_down']:
                score += 0.5
            elif method_info['is_pour_shaker']:
                score += 1.5

        if hasattr(node, 'task') and str(node.task).startswith("AchieveContainsShotCocktail"): #str cause task is a Task instance
            score += self.achieve_contains_shot_cocktail_penalty

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic24(achieve_contains_shot_cocktail_penalty={self.achieve_contains_shot_cocktail_penalty})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic24(achieve_contains_shot_cocktail_penalty={self.achieve_contains_shot_cocktail_penalty})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 24 info:\n"
            f"\tName: {self.name}\n"
            f"\tAchieve Contains Shot Cocktail Penalty: {self.achieve_contains_shot_cocktail_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
