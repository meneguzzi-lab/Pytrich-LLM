import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic5(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_5: str = "barman_gemini20flash_heuristic_5"):
        super().__init__(name=barman_gemini20flash_heuristic_5)
        self.clean_shot_penalty = 1
        self.clean_shaker_penalty = 2
        self.add_ingredient_penalty = 3
        self.make_cocktail_penalty = 5
        self.pour_cocktail_penalty = 4
        self.holding_penalty = 1
        self.ontable_penalty = 1
        self.handempty_penalty = 1
        self.preprocessing_time = 0
        self.method_scores = {}

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
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            clean_shot_count = 0
            clean_shaker_count = 0
            add_ingredient_count = 0
            make_cocktail_count = 0
            pour_cocktail_count = 0
            holding_count = 0
            ontable_count = 0
            handempty_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'achievecleanshot' in subtask_lower:
                        clean_shot_count += 1
                    if 'achievecleanshaker' in subtask_lower:
                        clean_shaker_count += 1
                    if 'achievecontainsshotingredient' in subtask_lower or 'achievecontainsshakeringredient' in subtask_lower:
                        add_ingredient_count += 1
                    if 'achievecontainsshotcocktail' in subtask_lower or 'achievecontainsshakercocktail' in subtask_lower:
                        make_cocktail_count += 1
                    if 'dopourshakertoshot' in subtask_lower:
                        pour_cocktail_count += 1
                    if 'achieveholding' in subtask_lower:
                        holding_count += 1
                    if 'achieveontable' in subtask_lower:
                        ontable_count += 1
                    if 'achievehandempty' in subtask_lower:
                        handempty_count += 1

            self.method_scores[method_name] = {
                'clean_shot_count': clean_shot_count,
                'clean_shaker_count': clean_shaker_count,
                'add_ingredient_count': add_ingredient_count,
                'make_cocktail_count': make_cocktail_count,
                'pour_cocktail_count': pour_cocktail_count,
                'holding_count': holding_count,
                'ontable_count': ontable_count,
                'handempty_count': handempty_count
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

            score += info['clean_shot_count'] * self.clean_shot_penalty
            score += info['clean_shaker_count'] * self.clean_shaker_penalty
            score += info['add_ingredient_count'] * self.add_ingredient_penalty
            score += info['make_cocktail_count'] * self.make_cocktail_penalty
            score += info['pour_cocktail_count'] * self.pour_cocktail_penalty
            score += info['holding_count'] * self.holding_penalty
            score += info['ontable_count'] * self.ontable_penalty
            score += info['handempty_count'] * self.handempty_penalty

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic5()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic5()"
