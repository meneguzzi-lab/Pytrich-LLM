import time
from typing import List, Union
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.model import Model, Operator, AbstractTask, Decomposition


class BarmanGemini20FlashHeuristic9(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_9: str = "BarmanGemini20FlashHeuristic9"):
        super().__init__(name=barman_gemini20flash_heuristic_9)
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            achieve_contains_cocktail = 0
            achieve_clean = 0
            subtask_count = 0

            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = ''
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    if 'AchieveContainsShotCocktail' in subtask_name:
                        achieve_contains_cocktail += 1
                    if 'AchieveCleanShot' in subtask_name or 'AchieveCleanShaker' in subtask_name:
                        achieve_clean += 1

            self.method_scores[method_name] = {
                'achieve_contains_cocktail': achieve_contains_cocktail,
                'achieve_clean': achieve_clean,
                'subtask_count': subtask_count,
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['achieve_contains_cocktail'] * 5  # Prioritize making cocktails
            score += info['achieve_clean'] * 2  # Cleaning is important
            score += info['subtask_count'] * 1 # Favor less subtasks
        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic9()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic9()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 9 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
