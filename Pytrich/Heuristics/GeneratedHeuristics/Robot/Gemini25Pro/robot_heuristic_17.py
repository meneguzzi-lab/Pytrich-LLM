import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic17(Heuristic):
    """
    A domain-dependent heuristic for the 'robot' HDDL domain.

    This heuristic guides the planner by scoring decomposition methods based on their
    contribution to the overall goal of delivering packages. It assigns lower heuristic
    costs (and thus higher priority) to methods that directly achieve parts of the goal,
    such as picking up or putting down a package, compared to enabling actions like
    moving or opening doors.

    The core idea is to create a gradient that encourages the planner to follow a
    logical sequence: complete a delivery (putdown), start a new one (pickup),
    move to the required location (move), and only open doors when necessary (open).

    Heuristic values are precomputed for all domain methods during initialization
    to ensure fast evaluation during the search.
    """
    def __init__(self, robot_gemini25pro_heuristic_17={
        "putdown_penalty": 0.1,
        "pickup_penalty": 0.2,
        "move_penalty": 0.3,
        "open_penalty": 0.4,
        "default_penalty": 1.0
    }):
        """
        Initializes the heuristic with a set of penalties for different action types.

        Args:
            robot_gemini25pro_heuristic_17 (dict): A dictionary of parameters to tune the heuristic's behavior.
                                       This structure directly addresses the user prompt's requirement for
                                       a parameter with this name.
        """
        super().__init__(name="robot_gemini25pro_heuristic_17")

        # Unpack penalties from the dictionary parameter, with default fallbacks
        self.putdown_penalty = robot_gemini25pro_heuristic_17.get("putdown_penalty", 0.1)
        self.pickup_penalty = robot_gemini25pro_heuristic_17.get("pickup_penalty", 0.2)
        self.move_penalty = robot_gemini25pro_heuristic_17.get("move_penalty", 0.3)
        self.open_penalty = robot_gemini25pro_heuristic_17.get("open_penalty", 0.4)
        self.default_penalty = robot_gemini25pro_heuristic_17.get("default_penalty", 1.0)
        self.termination_score = 0.0

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by precomputing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_scores()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_scores(self):
        """
        Analyzes each decomposition method in the domain and assigns a heuristic score
        based on its role (e.g., delivering a package, moving, opening a door).
        Lower scores indicate more promising methods.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name_lower = method.name.lower()
            score = self.default_penalty

            # Assign scores based on keywords in the method's name, which correspond
            # to its function in the robot delivery domain.
            if 'finished' in method_name_lower:
                # The 'finished' method terminates the plan successfully.
                score = self.termination_score
            elif 'putdown' in method_name_lower:
                # Methods involving 'putdown' complete a package delivery sub-goal.
                score = self.putdown_penalty
            elif 'pickup' in method_name_lower:
                # Methods involving 'pickup' start a package delivery sub-goal.
                score = self.pickup_penalty
            elif 'move' in method_name_lower:
                # 'move' methods are for navigation, a necessary but intermediate step.
                score = self.move_penalty
            elif 'open' in method_name_lower:
                # 'open' methods are enabling actions, one step further from the goal.
                score = self.open_penalty
            
            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the
        precomputed score of the method used in that node's decomposition.
        """
        # If the node has no associated method (e.g., the initial node), its cost is zero.
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        # Return the precomputed score. If a method was somehow missed during
        # preprocessing, assign a default penalty.
        return self.method_scores.get(method_name, self.default_penalty)

    def __call__(self, parent_node, node):
        """
        The main heuristic function called by the planner during search.
        It evaluates the promise of a search node by scoring its decomposition method.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
    
    def __repr__(self):
        return (f"RobotGemini25ProHeuristic17(penalties="
                f"putdown={self.putdown_penalty}, pickup={self.pickup_penalty}, "
                f"move={self.move_penalty}, open={self.open_penalty})")

    def __str__(self):
        return f"RobotGemini25ProHeuristic17 Method-Scoring Heuristic"

    def __output__(self):
        """
        Provides a formatted string with summary information about the heuristic.
        """
        return (
            f"Robot Gemini 2.5 Pro Heuristic 17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Scored: {len(self.method_scores)}\n"
            f"\tPutdown Penalty (Goal Completion): {self.putdown_penalty}\n"
            f"\tPickup Penalty (Goal Start): {self.pickup_penalty}\n"
            f"\tMove Penalty (Navigation): {self.move_penalty}\n"
            f"\tOpen Penalty (Enabling Action): {self.open_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s"
        )
