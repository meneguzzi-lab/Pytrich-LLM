import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for depot domain
class DepotGemini20FlashHeuristic14(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_14=1, name="depot_gemini20flash_heuristic_14"):
        super().__init__(name=name)
        self.name = name
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
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks
            load_count = 0
            unload_count = 0
            move_count = 0
            refuel_count = 0
            subtask_count = 0

            # Get subtasks from method
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

                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1
                    if 'move' in subtask_lower or 'drive' in subtask_lower:
                        move_count += 1
                    if 'refuel' in subtask_lower:
                        refuel_count += 1

            self.method_scores[method_name] = {
                'load_count': load_count,
                'unload_count': unload_count,
                'move_count': move_count,
                'refuel_count': refuel_count,
                'subtask_count': subtask_count,
                'is_achieve_method': self._is_achieve_method(method_name),
                'is_transport_method': self._is_transport_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': load_count + unload_count + move_count + refuel_count
            }

    def _is_achieve_method(self, method_name):
        """Check if method is for achieving a goal."""
        return 'achieve' in method_name.lower() or 'maintenance' in method_name.lower()

    def _is_transport_method(self, method_name):
        """Check if method involves transportation operations."""
        transport_indicators = ['load', 'unload', 'move', 'drive', 'refuel']
        return any(indicator in method_name.lower() for indicator in transport_indicators)

    def _is_base_case_method(self, method_name):
        """Check if method is a base case (no further decomposition needed)."""
        return 'done' in method_name.lower() or 'at-truck' in method_name.lower()

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly contributes to achieving goals."""
        goal_indicators = ['achieve', 'maintenance']
        return any(indicator in method_name.lower() for indicator in goal_indicators)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on various factors.
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0  # Start with 0 cost

        method_name = method.name if hasattr(method, 'name') else str(method)

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Penalize complexity and reward goal achievement
            score += info['load_count'] * 0.1
            score += info['unload_count'] * 0.1
            score += info['move_count'] * 0.2
            score += info['refuel_count'] * 0.15
            score -= info['is_goal_achieving'] * 0.3

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic14(name={self.name})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic14(name={self.name})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 14 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
