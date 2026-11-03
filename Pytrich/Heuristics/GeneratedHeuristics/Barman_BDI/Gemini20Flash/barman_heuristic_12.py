import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini20FlashHeuristic12(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_12=1, name="BarmanGemini20FlashHeuristic12"):
        super().__init__(name=name)
        self.name = name
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
            achieve_contains_shot_cocktail_count = 0
            achieve_clean_shot_count = 0
            achieve_contains_shaker_cocktail_count = 0
            do_pour_shaker_to_shot_count = 0
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

                    if 'AchieveContainsShotCocktail' in subtask_name:
                        achieve_contains_shot_cocktail_count += 1
                    if 'AchieveCleanShot' in subtask_name:
                        achieve_clean_shot_count += 1
                    if 'AchieveContainsShakerCocktail' in subtask_name:
                        achieve_contains_shaker_cocktail_count += 1
                    if 'DoPourShakerToShot' in subtask_name:
                        do_pour_shaker_to_shot_count += 1

            self.method_scores[method_name] = {
                'achieve_contains_shot_cocktail_count': achieve_contains_shot_cocktail_count,
                'achieve_clean_shot_count': achieve_clean_shot_count,
                'achieve_contains_shaker_cocktail_count': achieve_contains_shaker_cocktail_count,
                'do_pour_shaker_to_shot_count': do_pour_shaker_to_shot_count,
                'subtask_count': subtask_count
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

            score += info['achieve_contains_shot_cocktail_count'] * 5
            score += info['achieve_clean_shot_count'] * 3
            score += info['achieve_contains_shaker_cocktail_count'] * 7
            score += info['do_pour_shaker_to_shot_count'] * 2
            score += info['subtask_count'] * 1

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 12 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic12(name='{self.name}')"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic12(name='{self.name}')"
