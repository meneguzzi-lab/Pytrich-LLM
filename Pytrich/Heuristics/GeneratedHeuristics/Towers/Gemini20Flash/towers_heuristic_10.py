import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic10(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_10=1.0, name="towers_gemini20flash_heuristic_10"):
        super().__init__(name=name)
        self.misplaced_ring_penalty = towers_gemini20flash_heuristic_10
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

        # Calculate initial heuristic value
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

                    if 'move_abstract' in subtask_lower:
                        move_abstract_count += 1
                    if 'exchange' in subtask_lower:
                        exchange_count += 1
                    if 'rotateTower' in subtask_name:
                        rotate_count += 1
                    if 'selectDirection' in subtask_name:
                        select_direction_count += 1

            self.method_scores[method_name] = {
                'move_abstract_count': move_abstract_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'select_direction_count': select_direction_count,
                'subtask_count': subtask_count,
                'is_shift_tower_method': self._is_shift_tower_method(method_name),
                'complexity_score': move_abstract_count + exchange_count + rotate_count
            }

    def _is_shift_tower_method(self, method_name):
        return 'shiftTower' in method_name

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on misplaced rings.
        """
        state = node.state
        goal = self.model.goals
        misplaced_rings = 0

        # Iterate through goal facts and check if they are in the current state
        for goal_fact_id in goal:
            if goal_fact_id not in state:
                misplaced_rings += 1

        return misplaced_rings * self.misplaced_ring_penalty

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic10(misplaced_ring_penalty={self.misplaced_ring_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic10(misplaced_ring_penalty={self.misplaced_ring_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 10 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced ring penalty: {self.misplaced_ring_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
