import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic0(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic guides the search by assigning a cost to each decomposition
    method based on the type of action it leads to. It prioritizes methods that
    are closer to achieving the final goal (placing a package).

    The cost hierarchy is as follows (lower is better):
    1. `finished`: Completing the plan (cost 0).
    2. `putdown`: Placing a package, which directly satisfies a goal condition.
    3. `pickup`: A necessary step to begin moving a package towards its goal.
    4. `move`: Navigation, a fundamental but intermediate step.
    5. `open`: An auxiliary navigation action, considered slightly more costly than moving.

    By preferring methods in this order, the planner is encouraged to complete
    delivery tasks it has started before beginning new ones, and to favor direct
    goal-oriented actions over general navigation.
    """
    def __init__(self, robot_gemini25pro_heuristic_0, putdown_score=0.1, pickup_score=0.2, move_score=0.5, open_score=0.6):
        super().__init__(name="robot_gemini25pro_heuristic_0")
        # Store scores as costs (lower is better)
        self.putdown_score = putdown_score
        self.pickup_score = pickup_score
        self.move_score = move_score
        self.open_score = open_score
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-calculating scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # The initial node has no method applied yet, so its heuristic cost is 0.
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain and assigns a heuristic score
        to each one based on its name, which indicates its function.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            score = 1.0  # Default score for unknown methods

            # Assign scores based on keywords in the method name
            if self._is_terminal_method(method_name):
                score = 0.0
            elif self._is_goal_achieving_method(method_name):
                score = self.putdown_score
            elif self._is_setup_method(method_name):
                score = self.pickup_score
            elif self._is_move_method(method_name):
                score = self.move_score
            elif self._is_open_method(method_name):
                score = self.open_score

            self.method_scores[method_name] = score

    def _is_terminal_method(self, name):
        """Checks if the method is the final step of the plan."""
        return 'finished' in name.lower()

    def _is_goal_achieving_method(self, name):
        """Checks if the method involves putting down a package."""
        return 'putdown' in name.lower()

    def _is_setup_method(self, name):
        """Checks if the method involves picking up a package."""
        return 'pickup' in name.lower()

    def _is_move_method(self, name):
        """Checks if the method involves moving the robot."""
        return 'move' in name.lower()

    def _is_open_method(self, name):
        """Checks if the method involves opening a door."""
        return 'open' in name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node by looking up the
        precomputed score of the method that generated it.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0  # Root node or nodes without an applied method have 0 cost

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)

        # Return precomputed score, or a default value if not found
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic0(scores={{'putdown': {self.putdown_score}, 'pickup': {self.pickup_score}, ...}})"

    def __str__(self):
        return f"RobotGemini25ProHeuristic0 with precomputed scores for {len(self.method_scores)} methods."

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 0 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tScores (costs): putdown={self.putdown_score}, pickup={self.pickup_score}, "
            f"move={self.move_score}, open={self.open_score}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
