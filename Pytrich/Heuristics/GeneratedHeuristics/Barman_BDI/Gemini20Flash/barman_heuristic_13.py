import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic13(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_13=1.0, name="BarmanGemini20FlashHeuristic13"):
        super().__init__(name=name)
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
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
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            clean_count = 0
            pour_count = 0
            shake_count = 0
            ingredient_count = 0
            subtask_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'ingredient' in subtask_lower:
                        ingredient_count += 1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'pour_count': pour_count,
                'shake_count': shake_count,
                'ingredient_count': ingredient_count,
                'subtask_count': subtask_count,
                'is_cocktail_method': self._is_cocktail_method(method_name),
                'is_ingredient_method': self._is_ingredient_method(method_name),
                'is_null_method': self._is_null_method(method_name),
                'complexity_score': clean_count + pour_count + shake_count + ingredient_count
            }

    def _is_cocktail_method(self, method_name):
        return 'cocktail' in method_name.lower()

    def _is_ingredient_method(self, method_name):
        return 'ingredient' in method_name.lower()

    def _is_null_method(self, method_name):
        return 'null' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['clean_count'] * 0.1
            score += info['pour_count'] * 0.2
            score += info['shake_count'] * 0.3
            score += info['ingredient_count'] * 0.15
            score += info['subtask_count'] * 0.05

            if info['is_cocktail_method']:
                score += 0.5
            if info['is_ingredient_method']:
                score += 0.3
            if info['is_null_method']:
                score -= 0.2
        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic13()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic13()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 13 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
