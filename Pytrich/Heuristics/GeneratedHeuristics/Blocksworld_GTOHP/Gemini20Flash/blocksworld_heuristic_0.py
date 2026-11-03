import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic0(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_0: str = "blocksworld_gemini20flash_heuristic_0"):
        super().__init__(name=blocksworld_gemini20flash_heuristic_0)
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

            put_on_count = 0
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

                    if 'do_put_on' in subtask_lower:
                        put_on_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': put_on_count
            }

    def _is_goal_achieving_method(self, method_name):
        return 'do_put_on' in method_name.lower()

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
            score += info['put_on_count'] * 0.5
            score += info['subtask_count'] * 0.1

            if info['is_goal_achieving']:
                score -= 0.2

        if 'do_put_on' in method_lower:
            score += 0.3

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic0()"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic0()"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 0 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
