import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini20FlashHeuristic11(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_11, misplaced_penalty=10, blocking_penalty=5, name="towers_gemini20flash_heuristic_11"):
        super().__init__(name=name)
        self.misplaced_penalty = misplaced_penalty
        self.blocking_penalty = blocking_penalty
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
            move_abstract_count = 0
            exchange_count = 0
            rotate_count = 0
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
                    if 'move_abstract' in subtask_lower:
                        move_abstract_count += 1
                    if 'exchange' in subtask_lower:
                        exchange_count += 1
                    if 'rotateTower' in subtask_lower:
                        rotate_count += 1

            self.method_scores[method_name] = {
                'move_abstract_count': move_abstract_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'subtask_count': subtask_count,
                'is_shift_tower': self._is_shift_tower(method_name),
                'is_exchange': self._is_exchange(method_name)
            }

    def _is_shift_tower(self, method_name):
        return 'shiftTower' in method_name

    def _is_exchange(self, method_name):
        return 'exchange' in method_name

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return self._calculate_misplaced_rings(node)

        return self._score_method(node.method, node) + self._calculate_misplaced_rings(node)

    def _calculate_misplaced_rings(self, node):
        state = node.state
        goal_state = self.model.goals
        misplaced_rings = 0
        for goal_fact in goal_state:
          if goal_fact not in state:
            misplaced_rings+=1

        return misplaced_rings * self.misplaced_penalty

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['move_abstract_count']
            score += info['exchange_count']
            score += info['rotate_count']
            score += info['subtask_count'] * 0.1

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic11(misplaced_penalty={self.misplaced_penalty}, blocking_penalty={self.blocking_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic11(misplaced_penalty={self.misplaced_penalty}, blocking_penalty={self.blocking_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 11 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced penalty: {self.misplaced_penalty}\n"
            f"\tBlocking penalty: {self.blocking_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
