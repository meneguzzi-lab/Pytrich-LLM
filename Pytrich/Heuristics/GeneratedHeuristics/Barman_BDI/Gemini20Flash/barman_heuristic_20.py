import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini20FlashHeuristic20(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_20=1.0, name="BarmanGemini20FlashHeuristic20"):
        super().__init__(name=name)
        self.pour_penalty = barman_gemini20flash_heuristic_20
        self.clean_penalty = barman_gemini20flash_heuristic_20
        self.shake_penalty = barman_gemini20flash_heuristic_20
        self.ingredient_penalty = barman_gemini20flash_heuristic_20
        self.holding_penalty = barman_gemini20flash_heuristic_20
        self.ontable_penalty = barman_gemini20flash_heuristic_20
        self.handempty_penalty = barman_gemini20flash_heuristic_20
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
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            pour_count = 0
            clean_count = 0
            shake_count = 0
            ingredient_count = 0
            holding_count = 0
            ontable_count = 0
            handempty_count = 0
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

                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'ingredient' in subtask_lower:
                        ingredient_count += 1
                    if 'holding' in subtask_lower:
                        holding_count += 1
                    if 'ontable' in subtask_lower:
                        ontable_count += 1
                    if 'handempty' in subtask_lower:
                        handempty_count += 1

            self.method_scores[method_name] = {
                'pour_count': pour_count,
                'clean_count': clean_count,
                'shake_count': shake_count,
                'ingredient_count': ingredient_count,
                'holding_count': holding_count,
                'ontable_count': ontable_count,
                'handempty_count': handempty_count,
                'subtask_count': subtask_count,
                'complexity_score': pour_count + clean_count + shake_count + ingredient_count + holding_count + ontable_count + handempty_count
            }

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

            score += info['pour_count'] * self.pour_penalty
            score += info['clean_count'] * self.clean_penalty
            score += info['shake_count'] * self.shake_penalty
            score += info['ingredient_count'] * self.ingredient_penalty
            score += info['holding_count'] * self.holding_penalty
            score += info['ontable_count'] * self.ontable_penalty
            score += info['handempty_count'] * self.handempty_penalty

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic20(pour_penalty={self.pour_penalty}, clean_penalty={self.clean_penalty}, shake_penalty={self.shake_penalty}, ingredient_penalty={self.ingredient_penalty}, holding_penalty={self.holding_penalty}, ontable_penalty={self.ontable_penalty}, handempty_penalty={self.handempty_penalty})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic20(pour_penalty={self.pour_penalty}, clean_penalty={self.clean_penalty}, shake_penalty={self.shake_penalty}, ingredient_penalty={self.ingredient_penalty}, holding_penalty={self.holding_penalty}, ontable_penalty={self.ontable_penalty}, handempty_penalty={self.handempty_penalty})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 20 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPour penalty: {self.pour_penalty}\n"
            f"\tClean penalty: {self.clean_penalty}\n"
            f"\tShake penalty: {self.shake_penalty}\n"
            f"\tIngredient penalty: {self.ingredient_penalty}\n"
            f"\tHolding penalty: {self.holding_penalty}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tHandempty penalty: {self.handempty_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
