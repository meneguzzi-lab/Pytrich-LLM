import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic30(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_30=1, name="TowersGemini20FlashHeuristic30"): # Changed init parameter name to lower case
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_30 = towers_gemini20flash_heuristic_30 # Changed init parameter name to lower case
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes information about methods for faster scoring.
        """
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

                    if 'move_abstract' in subtask_lower or 'move ' in subtask_lower:
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
                'complexity_score': move_count + exchange_count + rotate_count + select_direction_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Scores a method based on domain-specific factors.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['move_count'] * 0.5  # Higher cost for move
            score += info['exchange_count'] * 1  # Med cost for exchange
            score += info['rotate_count'] * 0.75 # High cost for rotate
            score += info['select_direction_count'] * 0.25 # Low cost for select direction
            score += info['subtask_count'] * 0.1  # Small penality for each subtask
            score += info['complexity_score'] * 0.05
        else:
            score = 1 # default

        return score

    def __call__(self, parent_node, node):
        """
        Calculates heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic30(towers_gemini20flash_heuristic_30={self.towers_gemini20flash_heuristic_30})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic30(towers_gemini20flash_heuristic_30={self.towers_gemini20flash_heuristic_30})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 30 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
