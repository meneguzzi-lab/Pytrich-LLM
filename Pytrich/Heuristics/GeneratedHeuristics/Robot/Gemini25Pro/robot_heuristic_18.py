import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic18(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic guides the search by assigning costs to decomposition methods
    based on their strategic value in solving the problem. The core idea is to
    prioritize methods that make tangible progress toward moving packages to their
    goal locations.

    The heuristic values are structured as follows (lower is better):
    1.  **Completion Methods (lowest cost):** Methods that represent a terminal state,
        like `finished`, have a cost of 0.
    2.  **Goal Achievement Methods:** Methods that directly achieve a sub-goal, such as
        `release-putdown_abstract` (placing a package correctly), are given a low cost.
    3.  **Progress-Initiating Methods:** Methods that start a key task sequence, like
        `achieve-goals-pickup` (picking up a misplaced package), have a slightly higher cost.
    4.  **Enabling Methods (highest cost):** Methods for general navigation, such as moving
        or opening doors, are assigned the highest cost, as they are necessary but do not
        directly manipulate packages towards their goals.

    This strategy encourages the planner to focus on delivering packages first
    and only explore general movement or preparatory actions when no direct package
    manipulation is possible.
    """
    def __init__(self, robot_gemini25pro_heuristic_18, manipulation_cost=0.1, movement_cost=0.2):
        """
        Initializes the RobotGemini25ProHeuristic18.

        :param robot_gemini25pro_heuristic_18: The name for the heuristic instance.
        :param manipulation_cost: Base cost for actions that handle packages (pickup/putdown).
        :param movement_cost: Base cost for actions that involve robot movement (move/open).
        """
        super().__init__(name=robot_gemini25pro_heuristic_18)
        self.manipulation_cost = manipulation_cost
        self.movement_cost = movement_cost
        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for each method in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model and assigns a heuristic score to each.
        The scoring is based on the method's name, which implies its function in the domain.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name.lower()
            score = 0.0

            # Prioritize methods based on their strategic importance
            if 'finished' in method_name:
                # Base case for completion, no cost.
                score = 0.0
            elif 'release-putdown' in method_name:
                # Best action: completes a package delivery sub-goal.
                score = self.manipulation_cost
            elif 'achieve-goals-pickup' in method_name:
                # Good action: starts the process of delivering a misplaced package.
                score = self.manipulation_cost * 1.5
            elif 'release-move' in method_name or 'release-open' in method_name:
                # Necessary action: moving towards the goal while holding a package.
                score = self.movement_cost
            elif 'achieve-goals-move' in method_name or 'achieve-goals-open' in method_name:
                # General navigation, less preferred than purposeful movement.
                score = self.movement_cost * 1.2
            # Handle primitive action wrappers from 'newMethod' decompositions
            elif 'putdown' in method_name:
                score = self.manipulation_cost
            elif 'pickup' in method_name:
                score = self.manipulation_cost * 1.5
            elif 'move' in method_name or 'open' in method_name:
                score = self.movement_cost
            else:
                # Default cost for any unrecognized method.
                score = self.manipulation_cost + self.movement_cost

            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the
        pre-computed score of its associated method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        # Return the pre-computed score, or a default penalty if the method is unknown.
        return self.method_scores.get(method_name, self.manipulation_cost + self.movement_cost)

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic18(manipulation_cost={self.manipulation_cost}, movement_cost={self.movement_cost})"

    def __str__(self):
        return repr(self)

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 18 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tManipulation Cost: {self.manipulation_cost}\n"
            f"\tMovement Cost: {self.movement_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
