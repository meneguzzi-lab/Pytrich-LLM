import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic10(Heuristic):
    """
    A domain-dependent heuristic for the Robot domain.

    This heuristic provides search guidance by assigning a small, static cost to each
    decomposition method based on its role in the planning process. The core idea is to
    prefer methods that directly contribute to the main goal of moving packages over
    methods that are purely for navigation or searching.

    The heuristic values are structured as follows:
    - **Goal Achievement (Lowest Cost):** The 'finished' method, which terminates the
      main 'achieve-goals' task, has a cost of 0.
    - **Package Delivery:** Putting down a package at its goal location ('release-putdown')
      is the next most preferred action, as it completes a sub-goal.
    - **Package Acquisition:** Picking up a misplaced package ('achieve-goals-pickup') is
      the crucial first step in solving a sub-goal.
    - **Enabling Actions (Higher Cost):** Moving between rooms or opening doors are necessary
      but are considered enabling actions. They have a higher cost than direct package
      manipulation.
    - **Recursive Search (Highest Cost):** Methods that simply trigger a recursive call to
      search from a new location (e.g., 'achieve-goals-move') are given the highest
      cost, as they represent search effort rather than direct progress.

    By pre-computing these costs during initialization, the heuristic provides very fast
    evaluations during the search, guiding the planner towards more promising decompositions
    and improving performance, especially in larger problem instances.
    """
    def __init__(self, robot_gemini25pro_heuristic_10=None):
        """
        Initializes the RobotGemini25ProHeuristic10.
        :param robot_gemini25pro_heuristic_10: A placeholder parameter to match the required naming convention.
        """
        super().__init__(name="robot_gemini25pro_heuristic_10")
        self.method_scores = {}
        self.preprocessing_time = 0
        # Define static costs for different types of operations.
        # Lower values are more preferred.
        self.goal_achieved_cost = 0.0
        self.putdown_cost = 0.1
        self.pickup_cost = 0.2
        self.move_cost = 0.3
        self.open_cost = 0.35
        self.search_cost = 0.4

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for all methods.
        :param model: The planning model.
        :param initial_node: The initial search node.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model and assigns a pre-computed
        heuristic score to each based on its name and function.
        """
        methods = []
        if hasattr(self.model, 'decompositions') and self.model.decompositions:
            methods.extend(self.model.decompositions)
        elif hasattr(self.model, 'methods') and self.model.methods:
            methods.extend(self.model.methods)
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        for method in methods:
            method_name = method.name
            score = self._score_method_by_name(method_name)
            self.method_scores[method_name] = score

    def _score_method_by_name(self, method_name: str) -> float:
        """
        Scores a single method based on its name.
        :param method_name: The name of the method to score.
        :return: A float representing the heuristic cost.
        """
        name_lower = method_name.lower()

        if 'finished' in name_lower:
            return self.goal_achieved_cost

        if 'release-putdown' in name_lower or 'newmethod23' in name_lower:  # putdown_abstract
            return self.putdown_cost

        if 'achieve-goals-pickup' in name_lower or 'newmethod22' in name_lower:  # pickup_abstract
            return self.pickup_cost

        if 'move' in name_lower:
            if 'achieve-goals-move' in name_lower or 'release-move' in name_lower:
                return self.search_cost  # Recursive search step
            else:  # move_abstract or newMethod24
                return self.move_cost

        if 'open' in name_lower:
            if 'achieve-goals-open' in name_lower or 'release-open' in name_lower:
                return self.search_cost  # Recursive search step
            else:  # open_abstract or newMethod25
                return self.open_cost

        return 1.0  # Default high cost for any unrecognized methods

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        :param node: The HTNNode to evaluate.
        :return: The heuristic value.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        # Return the pre-computed score for the method applied in this node.
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        :param parent_node: The parent search node.
        :param node: The current search node to evaluate.
        :return: The calculated heuristic value.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self) -> str:
        return f"RobotGemini25ProHeuristic10()"

    def __str__(self) -> str:
        return self.name

    def __output__(self) -> str:
        return (
            f"Robot Gemini 2.5 Pro Heuristic 10 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
