import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic19(Heuristic):
    """
    A domain-dependent heuristic for the Barman HTN planning domain.

    This heuristic analyzes the structure of decomposition methods to estimate the cost
    of applying them. It aims to guide the search towards more efficient plans by
    penalizing methods that involve complex or resource-intensive operations.

    The core idea is to pre-calculate a score for each method based on the types
    of subtasks it generates. Key operations are identified and assigned penalties:
    - Cleaning: The most expensive operation, as it often requires emptying a container first.
    - Shaking: A critical and costly step in cocktail preparation.
    - Pouring/Filling: Core actions that advance the state towards the goal.
    - Manipulation: Basic actions like grasping and dropping, which have a small cost.

    By assigning higher costs to methods involving cleaning and shaking, the heuristic
    encourages the planner to find solutions that minimize these steps, for example,
    by reusing a shaker for cocktails with shared ingredients if possible, or by preparing
    all cocktails before cleaning the shaker.

    The heuristic is state-independent, meaning the scores are computed once during
    initialization. This makes it computationally efficient during the search process.
    The heuristic values are kept small to guide the search without compromising admissibility.
    """
    def __init__(self, clean_penalty=0.5, shake_penalty=0.4, pour_penalty=0.3, manipulation_penalty=0.1, barman_gemini25pro_heuristic_19=None):
        """
        Initializes the heuristic with penalties for different action types.

        Args:
            clean_penalty (float): The cost associated with a cleaning subtask.
            shake_penalty (float): The cost associated with a shaking subtask.
            pour_penalty (float): The cost for pouring or filling subtasks.
            manipulation_penalty (float): The cost for hand-related actions (grasp, drop).
            barman_gemini25pro_heuristic_19: Placeholder to match the required parameter name.
        """
        super().__init__(name="barman_gemini25pro_heuristic_19")
        self.clean_penalty = clean_penalty
        self.shake_penalty = shake_penalty
        self.pour_penalty = pour_penalty
        self.manipulation_penalty = manipulation_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic and precompute method scores from the domain model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes a score for each decomposition method based on its subtasks.
        Methods that are base cases ('null') are assigned a score of 0.
        Other methods are scored based on the weighted sum of their complex subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            method_lower = method_name.lower()

            # Base cases (e.g., 'MakeCocktailNull') represent an already achieved goal and have zero cost.
            if 'null' in method_lower:
                self.method_scores[method_name] = 0
                continue

            # Analyze subtasks to estimate the method's intrinsic cost.
            clean_count = 0
            shake_count = 0
            pour_fill_count = 0
            manipulation_count = 0
            
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network

            if subtasks:
                for subtask in subtasks:
                    subtask_name = str(getattr(subtask, 'name', ''))
                    subtask_lower = subtask_name.lower()

                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'pour' in subtask_lower or 'fill' in subtask_lower:
                        pour_fill_count += 1
                    if any(m in subtask_lower for m in ['holding', 'handempty', 'grasp', 'drop']):
                        manipulation_count += 1
            
            # The score is a weighted sum of the counts of different action types.
            score = 0
            score += clean_count * self.clean_penalty
            score += shake_count * self.shake_penalty
            score += pour_fill_count * self.pour_penalty
            score += manipulation_count * self.manipulation_penalty
            
            # Add a small base cost for any non-null decomposition to prefer shorter derivations.
            if score > 0 or subtasks:
                score += 0.05

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node: HTNNode):
        """
        Calculate the heuristic value for a node by retrieving the precomputed
        score of the method that generated it.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # Initial node or a node from a primitive action has no method cost.

        method_name = node.method.name
        # Use the precomputed score. A default penalty is returned for unknown methods.
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic19(clean={self.clean_penalty}, shake={self.shake_penalty}, pour={self.pour_penalty})"

    def __str__(self):
        return self.__repr__()
    
    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 19 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean Penalty: {self.clean_penalty}\n"
            f"\tShake Penalty: {self.shake_penalty}\n"
            f"\tPour/Fill Penalty: {self.pour_penalty}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
