import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic10(Heuristic):
    """A domain-dependent heuristic for the barman domain.

    This heuristic estimates the cost of achieving the goal by considering
    the number of 'AchieveContainsShotCocktail' tasks in the initial task network,
    as well as some common subgoals that need to be achieved. It uses a simplified
    approach, focusing on task counts to estimate the remaining effort.

    """

    def __init__(self, barman_gemini20flash_heuristic_10=1.0, name="BarmanGemini20FlashHeuristic10"):
        super().__init__(name=name)
        self.task_penalty = barman_gemini20flash_heuristic_10
        self.preprocessing_time = 0
        self.method_scores = {}

    def initialize(self, model, initial_node):
        """Initializes the heuristic with the model and precomputes method info."""
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_initial_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """Precomputes information about methods, such as subtask counts."""
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

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            subtask_count = 0
            if hasattr(method, 'subtasks'):
                subtask_count = len(method.subtasks)
            elif hasattr(method, 'task_network'):
                subtask_count = len(method.task_network)
            elif hasattr(method, 'decomposition'):
                subtask_count = len(method.decomposition)

            self.method_scores[method_name] = {
                'subtask_count': subtask_count
            }

    def _calculate_initial_heuristic(self, initial_node):
        """Calculates the initial heuristic value based on the initial task network.

        Counts the number of 'AchieveContainsShotCocktail' tasks and estimates
        the cost based on that count.
        """
        count = 0
        for task in initial_node.task_network:
            if hasattr(task, 'name') and task.name == 'AchieveContainsShotCocktail':
                count += 1
            elif hasattr(task, 'task') and hasattr(task.task, 'name') and task.task.name == 'AchieveContainsShotCocktail':
                count += 1 #Handles nested task structures
        return count * self.task_penalty

    def __call__(self, parent_node, node):
        """Calculates the heuristic value for a given node.

        Estimates the remaining cost based on the number of open
        'AchieveContainsShotCocktail' tasks in the node's task network.
        """
        count = 0
        for task in node.task_network:
             if hasattr(task, 'name') and task.name == 'AchieveContainsShotCocktail':
                count += 1
             elif hasattr(task, 'task') and hasattr(task.task, 'name') and task.task.name == 'AchieveContainsShotCocktail':
                count += 1 #Handles nested task structures

        h_value = count * self.task_penalty
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic10(task_penalty={self.task_penalty})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic10(task_penalty={self.task_penalty})"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic Info 10:\n"
            f"\tName: {self.name}\n"
            f"\tTask Penalty: {self.task_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
        )
