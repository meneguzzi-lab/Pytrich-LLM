import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic7(Heuristic):
    """
    A domain-dependent heuristic for the HDDL 'robot' domain.

    This heuristic guides the search by assigning a small, static cost to each
    decomposition method based on its strategic value in the planning process.
    The core idea is to prioritize methods that make direct progress towards the
    goal of delivering all packages.

    The heuristic values are structured as follows (lower is better):
    - Finishing the plan ('finished' method) is the top priority (lowest cost).
    - Delivering a package ('putdown') is the next best action.
    - Picking up a misplaced package ('pickup') is the next step.
    - Moving the robot and opening doors are necessary but less direct, so they
      receive higher heuristic costs.

    This approach is computationally lightweight as it relies on a one-time
    analysis of the domain's methods during initialization and follows the pattern
    of providing search guidance by scoring decompositions rather than states.
    """

    def __init__(self, robot_gemini25pro_heuristic_7):
        """
        Initializes the heuristic with a name and predefined scores for method types.
        The parameter `robot_gemini25pro_heuristic_7` is included to match the task requirements.
        """
        super().__init__(name="robot_gemini25pro_heuristic_7")
        # Heuristic costs for different method types. Lower scores are better.
        self.method_type_scores = {
            'goal_achieved': 0.0,       # For the 'finished' method
            'package_delivered': 0.1,   # For 'release-putdown_abstract'
            'package_pickup': 0.2,        # For 'achieve-goals-pickup'
            'action_wrapper': 0.3,      # For 'newMethod' wrappers
            'robot_move': 0.4,          # For methods involving 'move'
            'door_open': 0.5,           # For methods involving 'open'
            'default': 1.0              # Default for any other method
        }
        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for all domain methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain model and assigns a heuristic score
        based on their name, which reflects their function in this domain.
        """
        if not self.model:
            return

        decompositions = getattr(self.model, 'decompositions', [])
        if not decompositions:
            return

        for method in decompositions:
            method_name = method.name
            score = self.method_type_scores['default']

            # Score methods based on their strategic importance
            if 'finished' in method_name:
                score = self.method_type_scores['goal_achieved']
            elif 'release-putdown' in method_name:
                score = self.method_type_scores['package_delivered']
            elif 'achieve-goals-pickup' in method_name:
                score = self.method_type_scores['package_pickup']
            elif 'move' in method_name:
                score = self.method_type_scores['robot_move']
            elif 'open' in method_name:
                score = self.method_type_scores['door_open']
            elif 'newMethod' in method_name:
                score = self.method_type_scores['action_wrapper']

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is based on the method used for the last decomposition step,
        retrieved from the pre-computed scores.
        """
        if not hasattr(node, 'method') or not node.method:
            # For nodes representing tasks that are not yet decomposed.
            return self.method_type_scores['default']

        # The method name might be grounded, so we extract the base name.
        # e.g., from "(release-putdown_abstract room1 pkg1)" get "release-putdown_abstract"
        method_name_full = node.method.name
        method_base_name = method_name_full.split()[0].strip("()")[0]

        # Look up the precomputed score for the ungrounded method name.
        score = self.method_scores.get(method_base_name, self.method_type_scores['default'])

        return score

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic7(scores={self.method_type_scores})"

    def __str__(self):
        return "RobotGemini25ProHeuristic7 - A static method-scoring heuristic for the Robot domain."

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 7 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
            f"\tScoring Strategy: {self.method_type_scores}"
        )
