import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic10(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_10=1, name="DepotGemini20FlashHeuristic10"):
        super().__init__(name=name)
        self.name = name
        self.method_scores = {}
        self.preprocessing_time = 0
        self.distance_factor = 1

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

            move_count = 0
            load_count = 0
            unload_count = 0

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

                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'load_count': load_count,
                'unload_count': unload_count,
                'is_move_method': self._is_move_method(method_name),
                'is_load_method': self._is_load_method(method_name),
                'is_unload_method': self._is_unload_method(method_name),
                'is_base_case': self._is_base_case_method(method_name)
            }

    def _is_move_method(self, method_name):
        return 'move' in method_name.lower()

    def _is_load_method(self, method_name):
        return 'load' in method_name.lower()

    def _is_unload_method(self, method_name):
        return 'unload' in method_name.lower()

    def _is_base_case_method(self, method_name):
        return 'base' in method_name.lower()

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

            score += info['move_count'] * 0.1 * self.distance_factor
            score += info['load_count'] * 0.2
            score += info['unload_count'] * 0.2

            if info['is_base_case']:
                score -= 0.1

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic10(distance_factor={self.distance_factor})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic10(distance_factor={self.distance_factor})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 10 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tDistance factor: {self.distance_factor}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
