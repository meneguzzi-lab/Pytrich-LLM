import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Depot domain
class DepotGemini20FlashHeuristic24(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_24=1, name="DepotGemini20FlashHeuristic24"):
        super().__init__(name=name)
        self.refuel_penalty = 5  # Cost for refueling actions
        self.move_penalty = 2    # Cost for moving vehicles
        self.lift_penalty = 1    # Cost for lifting (loading/unloading)
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
        Counts occurrences of refuel, move, and lift operations in subtasks.
        """
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
            refuel_count = 0
            move_count = 0
            lift_count = 0
            subtask_count = 0

            # Determine the kind of attribute for subtasks the model has.
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

                    if 'refuel' in subtask_lower:
                        refuel_count += 1
                    if any(move_term in subtask_lower for move_term in ['drive', 'move']):
                        move_count += 1
                    if any(lift_term in subtask_lower for lift_term in ['lift', 'load', 'unload']):
                        lift_count += 1

            self.method_scores[method_name] = {
                'refuel_count': refuel_count,
                'move_count': move_count,
                'lift_count': lift_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'complexity_score': refuel_count + move_count + lift_count
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if a method directly contributes to achieving goals."""
        return 'achieve' in method_name.lower() or 'do-' in method_name.lower()

    def _is_base_case_method(self, method_name):
        """Check if a method is a base case (no further decomposition needed)."""
        return 'm11-' in method_name.lower() or 'm21-' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a node based on its decomposition method.
        Penalizes refuel, move, and lift operations.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Scores a method based on the number of refuel, move, and lift operations involved.
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['refuel_count'] * self.refuel_penalty
            score += info['move_count'] * self.move_penalty
            score += info['lift_count'] * self.lift_penalty

            if info['is_goal_achieving']:
                score -= 1  # Slight bonus for goal-achieving methods
            if info['is_base_case']:
                score -= 0.5 # Small bonus for terminating methods
        else:
            # If method is not found in precomputed scores, penalize it moderately.
            score += 2

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic24(refuel={self.refuel_penalty}, move={self.move_penalty}, lift={self.lift_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic24(refuel={self.refuel_penalty}, move={self.move_penalty}, lift={self.lift_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 24 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tRefuel penalty: {self.refuel_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLift penalty: {self.lift_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
