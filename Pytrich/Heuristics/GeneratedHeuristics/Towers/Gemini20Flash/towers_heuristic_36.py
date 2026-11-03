import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini20FlashHeuristic36(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_36=1.0, name="TowersGemini20FlashHeuristic36"):
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_36 = towers_gemini20flash_heuristic_36
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
            subtask_count = 0
            move_abstract_count = 0
            exchange_count = 0
            rotate_count = 0

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

                    if 'move_abstract' in subtask_name:
                        move_abstract_count += 1
                    elif 'exchange' in subtask_name:
                        exchange_count += 1
                    elif 'rotateTower' in subtask_name:
                        rotate_count +=1

            self.method_scores[method_name] = {
                'subtask_count': subtask_count,
                'move_abstract_count': move_abstract_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count
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
            score += info['subtask_count'] * 0.1
            score += info['move_abstract_count'] * 0.5
            score += info['exchange_count'] * 0.7
            score += info['rotate_count'] * 0.6

        return score * self.towers_gemini20flash_heuristic_36

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic36(towers_gemini20flash_heuristic_36={self.towers_gemini20flash_heuristic_36})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic36(towers_gemini20flash_heuristic_36={self.towers_gemini20flash_heuristic_36})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 36 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\ttowers_gemini20flash_heuristic_36: {self.towers_gemini20flash_heuristic_36}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
