import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic4(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_4=1.0, name="TowersGemini20FlashHeuristic4"):
        super().__init__(name=name)
        self.penalty = towers_gemini20flash_heuristic_4
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

            move_count = 0
            exchange_count = 0
            rotate_count = 0
            select_direction_count = 0
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

                    if 'move_abstract' in subtask_lower or 'move' in subtask_lower:
                        move_count += 1
                    if 'exchange' in subtask_lower:
                        exchange_count += 1
                    if 'rotate' in subtask_lower:
                        rotate_count += 1
                    if 'selectdirection' in subtask_lower:
                        select_direction_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'select_direction_count': select_direction_count,
                'subtask_count': subtask_count,
                'is_shift_tower': 'shiftTower' in method_name,
                'complexity_score': move_count + exchange_count + rotate_count + select_direction_count
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

            score += info['move_count'] * 0.2
            score += info['exchange_count'] * 0.3
            score += info['rotate_count'] * 0.25
            score += info['select_direction_count'] * 0.15
            score += info['subtask_count'] * 0.05

            if info['is_shift_tower']:
                score += 0.1  # Encourage methods directly related to the main task

        return score * self.penalty

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic4(penalty={self.penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic4(penalty={self.penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 4 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPenalty: {self.penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
