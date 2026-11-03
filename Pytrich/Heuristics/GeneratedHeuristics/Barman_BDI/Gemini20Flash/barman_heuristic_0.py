import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic0(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_0=1.0, name="BarmanGemini20FlashHeuristic0"):
        super().__init__(name=name)
        self.achieve_contains_shot_cocktail_weight = barman_gemini20flash_heuristic_0
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates a heuristic value for a given node in the search tree.
        The heuristic focuses on the 'AchieveContainsShotCocktail' task, assuming
        that achieving this task is a primary driver for solving the problem.
        """
        h_value = 0

        if not hasattr(node, 'task') or not node.task:
            return 0  # No task, no heuristic value

        task_str = str(node.task).lower()

        if "achievecontainsshotcocktail" in task_str:
            h_value += self.achieve_contains_shot_cocktail_weight

        return h_value

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for a given node. Updates heuristic info.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic0(weight={self.achieve_contains_shot_cocktail_weight})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic0 with AchieveContainsShotCocktail weight: {self.achieve_contains_shot_cocktail_weight}"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 0 info:\n"
            f"\tName: {self.name}\n"
            f"\tAchieveContainsShotCocktail weight: {self.achieve_contains_shot_cocktail_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
