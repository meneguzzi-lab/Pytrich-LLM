import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini25ProHeuristic16(Heuristic):
    """
    A domain-dependent heuristic for the Robot HDDL domain.

    This heuristic guides the search by assigning a cost to each decomposition
    method based on the type of action it progresses towards. The core idea is to
    prioritize actions that directly contribute to goal achievement (putdown)
    over preparatory actions (pickup) and enabling actions (move, open).

    The heuristic values are small floats, representing a cost. Lower values
    are preferred by the planner.

    - `finished`: Has the lowest cost (0.0), as it terminates a goal-achieving branch.
    - `putdown`: Low cost, as it achieves the goal for a package.
    - `pickup`: Higher cost than putdown, as it's a step towards the goal.
    - `move`: Higher cost, as it's a general enabling action.
    - `open`: Highest cost, as it's an enabling action that may not always be necessary.
    """
    def __init__(self, robot_gemini25pro_heuristic_16=None):
        """
        Initializes the heuristic with penalties for different action types.

        :param robot_gemini25pro_heuristic_16: A dictionary of penalties to override defaults.
        """
        # Define default penalties for different method types.
        penalties = {
            'putdown_penalty': 0.1,
            'pickup_penalty': 0.2,
            'move_penalty': 0.3,
            'open_penalty': 0.4,
            'finish_penalty': 0.0,
            'default_penalty': 0.5
        }
        if isinstance(robot_gemini25pro_heuristic_16, dict):
            penalties.update(robot_gemini25pro_heuristic_16)

        super().__init__(name="robot_gemini25pro_heuristic_16")
        self.putdown_penalty = penalties['putdown_penalty']
        self.pickup_penalty = penalties['pickup_penalty']
        self.move_penalty = penalties['move_penalty']
        self.open_penalty = penalties['open_penalty']
        self.finish_penalty = penalties['finish_penalty']
        self.default_penalty = penalties['default_penalty']

        self.method_scores = {}
        self.preprocessing_time = 0
        self.model = None

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model and assigns them a score
        based on their name, which indicates their function in the domain.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = self._score_method_by_name(method_name)
            self.method_scores[method_name] = score

    def _score_method_by_name(self, method_name):
        """
        Assigns a score to a method based on keywords in its name.
        Lower scores are better.
        """
        mn_lower = method_name.lower()
        if 'finished' in mn_lower:
            return self.finish_penalty  # Base case for successful goal achievement.
        if 'putdown' in mn_lower:
            return self.putdown_penalty  # Directly achieves a package's goal.
        if 'pickup' in mn_lower:
            return self.pickup_penalty  # Progresses towards delivering a package.
        if 'move' in mn_lower:
            return self.move_penalty  # General enabling action for navigation.
        if 'open' in mn_lower:
            return self.open_penalty  # Enabling action, but only needed for closed doors.

        # Default score for recursive or unclassified methods.
        return self.default_penalty

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node by looking up the score
        of the method used for decomposition.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no cost.

        method_name = node.method.name
        return self.method_scores.get(method_name, self.default_penalty)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic16(penalties={{'putdown': {self.putdown_penalty}, 'pickup': {self.pickup_penalty}, 'move': {self.move_penalty}, 'open': {self.open_penalty}}})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """Returns a formatted string with information about the heuristic."""
        return (
            f"Robot Gemini 2.5 Pro Heuristic 16 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tFinish Penalty: {self.finish_penalty}\n"
            f"\tPutdown Penalty: {self.putdown_penalty}\n"
            f"\tPickup Penalty: {self.pickup_penalty}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tOpen Penalty: {self.open_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
