import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic13(Heuristic):
    """
    A domain-dependent heuristic for the 'robot' domain.

    This heuristic guides the planner by assigning costs to different methods (decompositions)
    based on their strategic value in solving the problem. It operates without inspecting
    the dynamic state, aligning with the Pytrich framework's method-based evaluation.

    The core idea is to prioritize actions that directly contribute to the main goal of
    delivering packages:
    1. Goal Achievement (putdown): Lowest cost, as it completes a sub-goal.
    2. Goal Initiation (pickup): Low cost, as it starts the delivery process.
    3. Transport (move): Medium cost, as it enables pickup and putdown.
    4. Setup (open): Highest cost, as it only removes obstacles for transport.

    The heuristic also differentiates between moving *with* a package (more valuable)
    and moving *to* a package, assigning a slightly lower cost to the former.
    """
    def __init__(self, robot_gemini25pro_heuristic_13={"name": "robot_gemini25pro_heuristic_13"}):
        # Default penalties (costs) for different types of methods. Lower is better.
        self.penalties = {
            'achieve': 0.1,   # Cost for putting down a package (goal completion)
            'initiate': 0.5,  # Cost for picking up a package
            'transport': 1.0, # Base cost for moving
            'setup': 2.0,     # Base cost for opening doors
            'default': 3.0    # Cost for any other method
        }
        # Allow overriding default penalties via the initialization dictionary.
        if isinstance(robot_gemini25pro_heuristic_13, dict) and "penalties" in robot_gemini25pro_heuristic_13:
            self.penalties.update(robot_gemini25pro_heuristic_13["penalties"])

        super().__init__(name=robot_gemini25pro_heuristic_13.get("name", "robot_gemini25pro_heuristic_13") if isinstance(robot_gemini25pro_heuristic_13, dict) else "robot_gemini25pro_heuristic_13")
        
        self.method_info = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-analyzing the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes and classifies all methods from the domain model based on their names
        to determine their strategic role.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            method_lower = method_name.lower()

            info = {
                'is_goal_achieving': self._is_goal_achieving(method_lower),
                'is_goal_initiating': self._is_goal_initiating(method_lower),
                'is_transport': self._is_transport(method_lower),
                'is_setup': self._is_setup(method_lower),
                'is_terminating': self._is_terminating(method_lower)
            }
            self.method_info[method_name] = info

    def _is_goal_achieving(self, name):
        """Checks if a method completes a package delivery (putdown)."""
        return 'release-putdown' in name or 'newmethod23' in name

    def _is_goal_initiating(self, name):
        """Checks if a method starts a package delivery (pickup)."""
        return 'achieve-goals-pickup' in name or 'newmethod22' in name

    def _is_transport(self, name):
        """Checks if a method involves moving the robot."""
        return 'move' in name

    def _is_setup(self, name):
        """Checks if a method is for setup, like opening a door."""
        return 'open' in name

    def _is_terminating(self, name):
        """Checks if a method is a base case that ends a task branch."""
        return 'finished' in name

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a node by scoring its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Assigns a cost to a method based on its precomputed classification.
        A small adjustment is made to prioritize more strategic actions within a class
        (e.g., moving with a package is better than moving to find one).
        """
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name not in self.method_info:
            return self.penalties['default']

        info = self.method_info[method_name]
        method_lower = method_name.lower()
        
        if info['is_terminating']:
            return 0.0
        if info['is_goal_achieving']:
            return self.penalties['achieve']
        if info['is_goal_initiating']:
            return self.penalties['initiate']
        if info['is_setup']:
            # Opening a door while carrying a package ('release') is slightly more urgent.
            return self.penalties['setup'] if 'release-open' in method_lower else self.penalties['setup'] + 0.1
        if info['is_transport']:
            # Moving while carrying a package ('release') is better than moving to find one ('achieve-goals').
            return self.penalties['transport'] if 'release-move' in method_lower else self.penalties['transport'] + 0.1

        return self.penalties['default']

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"RobotGemini25ProHeuristic13(penalties={self.penalties})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 13 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_info)}\n"
            f"\tPenalties: {self.penalties}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
        )
