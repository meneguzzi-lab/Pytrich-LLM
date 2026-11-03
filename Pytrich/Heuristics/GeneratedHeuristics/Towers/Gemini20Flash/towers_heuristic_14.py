import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic14(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_14=1.0, name="TowersGemini20FlashHeuristic14"):  # All parameters lower case
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_14 = towers_gemini20flash_heuristic_14
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        if not self.model:
            return

        methods = []

        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            for abstract_task in self.model.abstract_tasks:
                if hasattr(abstract_task, 'decompositions'):
                    methods.extend(abstract_task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            move_abstract_count = 0
            exchange_count = 0
            rotate_tower_count = 0
            select_direction_count = 0

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

                    if 'move_abstract' in subtask_lower:
                        move_abstract_count += 1
                    elif 'exchange' in subtask_lower:
                        exchange_count += 1
                    elif 'rotatetower' in subtask_lower:
                        rotate_tower_count += 1
                    elif 'selectdirection' in subtask_lower:
                        select_direction_count += 1

            self.method_scores[method_name] = {
                'move_abstract_count': move_abstract_count,
                'exchange_count': exchange_count,
                'rotate_tower_count': rotate_tower_count,
                'select_direction_count': select_direction_count,
                'is_shift_tower': self._is_shift_tower(method_name),
                'is_exchange_clear': self._is_exchange_clear(method_name),
                'is_exchange': self._is_exchange(method_name)
            }

    def _is_shift_tower(self, method_name):
        return 'shiftTower' in method_name

    def _is_exchange_clear(self, method_name):
        return 'exchangeClear' in method_name

    def _is_exchange(self, method_name):
        return 'exchange' in method_name

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

            score += info['move_abstract_count'] * 0.2
            score += info['exchange_count'] * 0.3
            score += info['rotate_tower_count'] * 0.25
            score += info['select_direction_count'] * 0.1

            if info['is_shift_tower']:
                score += 0.1
            if info['is_exchange_clear']:
                score -= 0.2  # Encourage exchange clear
            if info['is_exchange']:
                score += 0.15

        return score * self.towers_gemini20flash_heuristic_14 # Use the parameter

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic14(towers_gemini20flash_heuristic_14={self.towers_gemini20flash_heuristic_14})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic14(towers_gemini20flash_heuristic_14={self.towers_gemini20flash_heuristic_14})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 14 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tTowers Heuristic 14 Parameter: {self.towers_gemini20flash_heuristic_14}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
