import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic21(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_21=1.0, name="BarmanGemini20FlashHeuristic21"):
        super().__init__(name=name)
        self.pour_penalty = barman_gemini20flash_heuristic_21
        self.clean_penalty = barman_gemini20flash_heuristic_21
        self.grasp_penalty = barman_gemini20flash_heuristic_21
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
            grasp_count = 0
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
                    subtask_name = ''
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
                    if 'grasp' in subtask_lower:
                        grasp_count += 1

            self.method_scores[method_name] = {
                'pour_count': pour_count,
                'clean_count': clean_count,
                'grasp_count': grasp_count,
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
            score += info['pour_count'] * self.pour_penalty
            score += info['clean_count'] * self.clean_penalty
            score += info['grasp_count'] * self.grasp_penalty

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic21(pour={self.pour_penalty}, clean={self.clean_penalty}, grasp={self.grasp_penalty})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic21(pour={self.pour_penalty}, clean={self.clean_penalty}, grasp={self.grasp_penalty})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 21 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPour penalty: {self.pour_penalty}\n"
            f"\tClean penalty: {self.clean_penalty}\n"
            f"\tGrasp penalty: {self.grasp_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
