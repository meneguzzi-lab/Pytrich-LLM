import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic5(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic guides the search by assigning small, optimistic costs (penalties)
    to the decomposition methods based on the type of action they introduce.
    The core idea is to prioritize methods that make direct progress towards the goal
    of delivering packages, while penalizing methods that only enable movement or
    other preparatory actions.

    The scoring hierarchy is as follows (lower is better):
    1.  **Termination Methods**: Methods that complete the plan (e.g., 'finished') have zero cost.
    2.  **Goal-Achieving Methods**: Methods involving 'putdown' are prioritized as they complete a package delivery.
    3.  **Goal-Initiating Methods**: Methods involving 'pickup' are next, as they start a delivery sequence.
    4.  **Enabling Methods**: Methods for 'move' and 'open' are penalized more, as they are necessary but don't directly handle packages. Opening a door is penalized slightly more than moving.

    The heuristic values are kept small and constant (<< 1) to ensure admissibility, as any primitive action has a cost of 1. This makes the heuristic effective for guiding the search and breaking ties without compromising optimality.
    """

    def __init__(self, robot_gemini25pro_heuristic_5={'putdown': 0.1, 'pickup': 0.2, 'move': 0.3, 'open': 0.4}):
        """
        Initializes the heuristic with configurable penalties for different action types.

        :param robot_gemini25pro_heuristic_5: A dictionary mapping action types to penalty scores.
        """
        super().__init__(name="robot_gemini25pro_heuristic_5")
        self.penalties = robot_gemini25pro_heuristic_5
        self.method_scores = {}
        self.model = None
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-calculating scores for all methods in the domain.

        :param model: The planning model containing domain information.
        :param initial_node: The root node of the search.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain and assigns a heuristic score
        based on the actions they are associated with. The scores are stored for quick lookup.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name.lower()
            # Assign a default penalty for any method not explicitly matched.
            score = 0.5

            # Score methods based on keywords related to their function in the plan.
            if 'finished' in method_name:
                score = 0.0  # Best case: plan is complete.
            elif 'putdown' in method_name:
                score = self.penalties.get('putdown', 0.1)  # Completes a package delivery.
            elif 'pickup' in method_name:
                score = self.penalties.get('pickup', 0.2)  # Initiates a package delivery.
            elif 'move' in method_name:
                score = self.penalties.get('move', 0.3)  # Necessary movement.
            elif 'open' in method_name:
                score = self.penalties.get('open', 0.4)  # Necessary interaction with doors.

            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the score
        of the method used to generate this node.

        :param node: The HTN search node to evaluate.
        :return: The heuristic score (a small cost).
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, e.g., the initial task network.

        method_name = node.method.name
        # Return the pre-computed score, with a default for safety.
        return self.method_scores.get(method_name, 0.5)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.

        :param parent_node: The parent node in the search tree.
        :param node: The current node to evaluate.
        :return: The calculated heuristic value for the node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic5(penalties={self.penalties})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a formatted string with information about the heuristic's configuration and performance.
        """
        return (
            f"Robot Gemini 2.5 Pro Heuristic 5 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPenalties: {self.penalties}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
