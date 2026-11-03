import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic14(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic works by assigning a small, admissible cost to each decomposition
    method based on its semantic role in the planning process. The core idea is to
    guide the search towards methods that make tangible progress towards the goal of
    delivering packages.

    The heuristic values are structured to create a preference order:
    1. Termination methods (cost 0.0): The 'finished' method is the most preferred as it ends a branch.
    2. Goal Achievement methods (low cost): 'release-putdown' directly achieves a subgoal (delivering a package).
    3. Goal Setup methods (medium-low cost): 'achieve-goals-pickup' initiates a delivery process.
    4. Navigation methods (medium cost): Methods for moving and opening doors are necessary but less direct.
       - A distinction is made between moving while holding a package (delivery) vs. moving while empty (searching).
    5. Wrapper methods (high cost): Simple decompositions to primitive actions are given a higher cost to reflect that they are just a step in a larger process.

    This approach avoids complex state analysis, which may not be readily available
    to the heuristic function, and instead leverages the explicit hierarchical structure
    of the domain to provide guidance. The costs are kept small to ensure admissibility,
    acting as a tie-breaker and guide rather than a precise cost-to-go estimate.
    """
    def __init__(self, robot_gemini25pro_heuristic_14=None):
        super().__init__(name="robot_gemini25pro_heuristic_14")
        self.termination_score = 0.0
        self.goal_achievement_score = 0.1
        self.goal_setup_score = 0.2
        self.delivery_navigation_score = 0.4
        self.search_navigation_score = 0.5
        self.primitive_wrapper_score = 0.6
        self.default_score = 1.0

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic by pre-computing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods from the model and assigns a heuristic score.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = self._get_method_score(method_name)

    def _get_method_score(self, method_name):
        """
        Calculates a heuristic score for a method based on its name, which reflects its role.
        """
        mn_lower = method_name.lower()

        if "finished" in mn_lower:
            return self.termination_score
        elif "release-putdown" in mn_lower:
            return self.goal_achievement_score
        elif "achieve-goals-pickup" in mn_lower:
            return self.goal_setup_score
        elif "release-move" in mn_lower or "release-open" in mn_lower:
            return self.delivery_navigation_score
        elif "achieve-goals-move" in mn_lower or "achieve-goals-open" in mn_lower:
            return self.search_navigation_score
        elif "newmethod" in mn_lower:
            return self.primitive_wrapper_score
        else:
            return self.default_score

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given search node by looking up the
        pre-computed score of the method used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        return self.method_scores.get(method_name, self.default_score)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic14()"

    def __str__(self):
        return f"RobotGemini25ProHeuristic14 - A heuristic for the Robot domain."

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 14 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tScores:\n"
            f"\t  - Termination: {self.termination_score}\n"
            f"\t  - Goal Achievement: {self.goal_achievement_score}\n"
            f"\t  - Goal Setup: {self.goal_setup_score}\n"
            f"\t  - Delivery Navigation: {self.delivery_navigation_score}\n"
            f"\t  - Search Navigation: {self.search_navigation_score}\n"
            f"\t  - Primitive Wrapper: {self.primitive_wrapper_score}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
