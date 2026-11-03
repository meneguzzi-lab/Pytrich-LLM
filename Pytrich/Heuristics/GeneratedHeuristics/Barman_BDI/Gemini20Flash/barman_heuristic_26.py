import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic26(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_26=1.0, name="BarmanGemini20FlashHeuristic26"):
        super().__init__(name=name)
        self.achieve_contains_shot_cocktail_weight = barman_gemini20flash_heuristic_26
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'task') or not node.task:
            return 0

        task_name = node.task.name if hasattr(node.task, 'name') else str(node.task)

        if task_name == 'AchieveContainsShotCocktail':
            return self.achieve_contains_shot_cocktail_weight
        elif task_name == 'AchieveContainsShakerIngredient':
            return self.achieve_contains_shot_cocktail_weight * 0.75  # Give it slightly lower weight
        elif task_name == 'AchieveCleanShaker':
            return self.achieve_contains_shot_cocktail_weight * 0.5
        elif task_name == 'AchieveHandEmpty':
            return 0.1
        elif task_name == 'AchieveContainsShotIngredient':
            return self.achieve_contains_shot_cocktail_weight * 0.6
        elif task_name == 'AchieveContainsShakerCocktail':
            return self.achieve_contains_shot_cocktail_weight * 0.9
        elif task_name == 'DoPourShakerToShot':
            return self.achieve_contains_shot_cocktail_weight * 0.4
        elif task_name == 'AchieveOnTable':
            return 0.05
        elif task_name == 'AchieveHolding':
            return 0.15
        elif task_name == 'AchieveCleanShot':
            return self.achieve_contains_shot_cocktail_weight * 0.4
        else:
            return 0

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic26(achieve_contains_shot_cocktail_weight={self.achieve_contains_shot_cocktail_weight})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic26(achieve_contains_shot_cocktail_weight={self.achieve_contains_shot_cocktail_weight})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 26 info:\n"
            f"\tName: {self.name}\n"
            f"\tAchieveContainsShotCocktail Weight: {self.achieve_contains_shot_cocktail_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
