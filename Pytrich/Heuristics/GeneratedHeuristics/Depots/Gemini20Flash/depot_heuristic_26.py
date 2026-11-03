import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Depot domain
class DepotGemini20FlashHeuristic26(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_26=1, name="DepotGemini20FlashHeuristic26"):
        super().__init__(name=name)
        self.depot_gemini20flash_heuristic_26 = depot_gemini20flash_heuristic_26
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes
        method characteristics.
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
        Precomputes information about methods for faster scoring.
        This could include analyzing subtasks for common patterns.
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
            # Example: Count number of 'move' and 'load' actions in subtasks
            move_count = 0
            load_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                for subtask in subtasks:
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name.lower()
                        if 'move' in subtask_name:
                            move_count += 1
                        elif 'load' in subtask_name:
                            load_count += 1

            # Store precomputed information
            self.method_scores[method_name] = {
                'move_count': move_count,
                'load_count': load_count,
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        This considers the method used for decomposition and the current state.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Scores a method based on its precomputed characteristics.
        Lower scores are better.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            # Penalize methods with more move and load actions
            score += info['move_count'] * 0.2  # Example penalty
            score += info['load_count'] * 0.3  # Example penalty

        # Example: Prefer methods that reduce the number of open tasks
        if hasattr(node, 'open_tasks'):
            score += len(node.open_tasks) * 0.1 # Example influence

        return score

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for a node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic26(depot_gemini20flash_heuristic_26={self.depot_gemini20flash_heuristic_26})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic26(depot_gemini20flash_heuristic_26={self.depot_gemini20flash_heuristic_26})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 26 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tdepot_gemini20flash_heuristic_26: {self.depot_gemini20flash_heuristic_26}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
