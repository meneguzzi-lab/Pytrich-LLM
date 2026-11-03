import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic4(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost of a plan by analyzing the methods chosen
    during decomposition. It assigns a higher cost to methods that involve clearing
    blocks, as this is typically the most complex part of a Blocksworld problem.
    The heuristic is designed to be admissible by keeping the estimated costs low.

    Rationale:
    1.  **Domain Insight**: The primary challenge in Blocksworld is not just moving a block
        to its destination, but clearing any blocks that are on top of it or on top
        of its destination. This often requires a series of moves.
    2.  **Heuristic Strategy**: The heuristic pre-computes a score for each method in the
        domain based on its subtasks. It penalizes methods more heavily if their
        subtasks involve abstract 'clear' tasks or multiple primitive move actions
        (stack, unstack, etc.).
    3.  **Admissibility**: The penalties are small floating-point numbers (e.g., 0.1, 0.3)
        to ensure the heuristic value remains an underestimate of the true cost, thus
        guiding the search without compromising optimality.
    4.  **Static Analysis**: Following the provided examples, this heuristic relies on static
        analysis of method names and their subtask structure, as direct state inspection
        is not assumed to be available or efficient within the heuristic's call.
    """

    def __init__(self, move_penalty=0.1, clear_penalty=0.3, name="blocksworld_gemini25pro_heuristic_4"):
        """
        Initializes the heuristic with penalties for different types of operations.

        Args:
            move_penalty (float): The base cost assigned to a single primitive move action.
            clear_penalty (float): The additional cost for methods that involve clearing blocks.
            name (str): The name of the heuristic instance.
        """
        super().__init__(name=name)
        self.move_penalty = move_penalty
        self.clear_penalty = clear_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing information about the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain model to assign a heuristic score based on
        the complexity of their subtasks.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            move_actions = 0
            clear_tasks = 0
            subtask_count = 0

            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'subtasks'):
                subtasks = method.subtasks

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = str(subtask.name if hasattr(subtask, 'name') else subtask)
                    subtask_lower = subtask_name.lower()

                    if any(move_op in subtask_lower for move_op in ['unstack', 'stack', 'pick-up', 'put-down']):
                        move_actions += 1
                    if any(clear_op in subtask_lower for clear_op in ['clear', 'make-clear']):
                        clear_tasks += 1

            self.method_scores[method_name] = {
                'move_actions': move_actions,
                'clear_tasks': clear_tasks,
                'is_base_case': subtask_count == 0,
                'is_complex': clear_tasks > 0,
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node based on its method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0
        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on its precomputed characteristics.
        A lower score indicates a more promising path.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            if info['is_base_case']:
                return 0
            score += info['move_actions'] * self.move_penalty
            score += info['clear_tasks'] * self.clear_penalty
        else:
            # Fallback scoring if method not found in precomputed scores
            method_lower = method_name.lower()
            if any(term in method_lower for term in ['unstack', 'stack', 'pick-up', 'put-down']):
                score += self.move_penalty
            if any(term in method_lower for term in ['clear', 'make-clear']):
                score += self.clear_penalty

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        The main callable method for the heuristic, invoked by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        # Cap the heuristic value to help maintain admissibility
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic4(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __str__(self):
        return f"BlocksworldGemini25ProHeuristic4(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tClear penalty: {self.clear_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
