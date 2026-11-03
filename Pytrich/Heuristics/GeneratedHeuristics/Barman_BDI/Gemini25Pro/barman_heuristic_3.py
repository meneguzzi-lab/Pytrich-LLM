import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# This domain-dependent heuristic was generated for the barman domain.
# Its design is based on an analysis of the domain's hierarchical structure
# and key planning challenges, such as resource contention (hands, shaker) and
# the lifecycle of container states (clean, used, empty).
class BarmanGemini25ProHeuristic3(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    Rationale:
    The heuristic estimates the cost of a decomposition method by analyzing its subtasks.
    It assigns penalties to different categories of actions, reflecting their contribution
    to the overall planning effort. The core idea is to guide the search towards plans
    that minimize resource-intensive and preparatory actions like cleaning and excessive
    manipulation, while prioritizing productive actions like preparing ingredients and
    mixing cocktails.

    Key Principles:
    1.  **Cost by Action Type**: Different actions have different costs. Cleaning is the most
        expensive as it's a significant reset step and a bottleneck. Shaking and pouring
        are productive but still have a cost. Basic manipulation (grasp/drop) is cheapest.
    2.  **Pre-computation**: Method scores are pre-computed during initialization to ensure
        the heuristic is fast during the search. It inspects the subtasks of each method
        once.
    3.  **Hierarchy-Aware**: The heuristic evaluates methods, which are the core of the
        hierarchical structure. By scoring a method based on its subtasks, it inherently
        understands the cost of a decomposition step.
    4.  **Admissibility**: The penalty values are kept small (<< 1.0) to ensure the heuristic
        is more of a tie-breaker that guides the search without overestimating the true
        cost to the goal, thus aiming for admissibility.
    """

    def __init__(self, cleaning_penalty=0.4, shaking_penalty=0.3, pouring_penalty=0.2, manipulation_penalty=0.05, name="barman_gemini25pro_heuristic_3"):
        """
        Initializes the heuristic with penalties for different action types.

        Args:
            cleaning_penalty (float): Penalty for methods involving cleaning tasks.
            shaking_penalty (float): Penalty for the shake action.
            pouring_penalty (float): Penalty for pouring and filling actions.
            manipulation_penalty (float): Penalty for basic manipulation (grasp, drop, etc.).
            name (str): The name of the heuristic instance.
        """
        super().__init__(name=name)
        self.cleaning_penalty = cleaning_penalty
        self.shaking_penalty = shaking_penalty
        self.pouring_penalty = pouring_penalty
        self.manipulation_penalty = manipulation_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing method scores from the domain model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model to assign a heuristic score.
        This score is based on the types and number of subtasks in the method's expansion.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            method_lower = method_name.lower()

            # Null methods represent already achieved goals and have zero cost.
            if "null" in method_lower:
                self.method_scores[method_name] = {'score': 0.0, 'is_base_case': True}
                continue

            # Initialize counters for different operation types.
            cleaning_ops = 0
            manipulation_ops = 0
            pouring_ops = 0
            shaking_ops = 0

            subtasks = getattr(method, 'task_network', [])

            for subtask in subtasks:
                subtask_name = subtask.name.lower()

                if 'clean' in subtask_name:
                    cleaning_ops += 1
                elif any(op in subtask_name for op in ['grasp', 'drop', 'holding', 'handempty', 'ontable']):
                    manipulation_ops += 1
                elif any(op in subtask_name for op in ['pour', 'fill', 'do']):
                    pouring_ops += 1
                elif 'shake' in subtask_name:
                    shaking_ops += 1

            # The score is a weighted sum of the identified operations.
            score = (
                cleaning_ops * self.cleaning_penalty +
                manipulation_ops * self.manipulation_penalty +
                pouring_ops * self.pouring_penalty +
                shaking_ops * self.shaking_penalty
            )

            self.method_scores[method_name] = {
                'score': score,
                'is_base_case': False
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is the pre-computed score of the method used to reach this node's state.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name

        if method_name in self.method_scores:
            return self.method_scores[method_name]['score']

        # Return a small default cost for any method not found during pre-computation.
        return 0.1

    def __call__(self, parent_node, node):
        """
        Evaluates the heuristic for a given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic3(clean={self.cleaning_penalty}, shake={self.shaking_penalty}, pour={self.pouring_penalty}, manip={self.manipulation_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 3 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tCleaning Penalty: {self.cleaning_penalty}\n"
            f"\tShaking Penalty: {self.shaking_penalty}\n"
            f"\tPouring Penalty: {self.pouring_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
