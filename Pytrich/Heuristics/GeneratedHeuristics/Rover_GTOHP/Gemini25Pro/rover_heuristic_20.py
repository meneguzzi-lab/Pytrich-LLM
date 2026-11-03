import time
from typing import Dict, Any

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask, Decomposition

class RoverGemini25ProHeuristic20(Heuristic):
    """
    A domain-dependent heuristic for the HDDL ROVER domain.

    This heuristic estimates the cost of a decomposition by analyzing the subtasks
    introduced by a method. It assigns penalties based on the type of operation,
    with the highest penalties for actions that imply significant effort, such as
    navigation, communication, and calibration.

    Rationale:
    1.  **Navigation is the Dominant Cost**: The primary cost in the ROVER domain is moving
        the rover between waypoints. Tasks like `navigate`, `do_navigate1`, `do_navigate2`,
        and tasks that require navigation as a prerequisite (e.g., `send_*_data`,
        `do_calibrate`) are given the highest penalties.

    2.  **Hierarchical Cost Estimation**: The heuristic evaluates methods, which are the core of
        hierarchical planning. By scoring each decomposition choice, it guides the planner
        towards methods that introduce fewer costly operations, pruning the search space effectively.

    3.  **Admissibility**: The penalties are small, abstract cost values, not direct estimates
        of plan length. They serve as a lower bound on the effort required, ensuring the
        heuristic remains admissible for optimal search algorithms like A*.

    4.  **Efficiency**: All method scores are precomputed during the `initialize` phase.
        During the search, the heuristic value for a node is a fast dictionary lookup,
        imposing minimal computational overhead.

    5.  **Domain-Specific Insights**:
        - `communicate_*` and `send_*` tasks are penalized heavily because they require the
          rover to navigate to a location with line-of-sight to the lander.
        - `calibrate` is penalized as it also requires navigating to a specific calibration target.
        - `sample_*` and `image_*` tasks have a moderate penalty as they are the primary goal-
          achieving actions but require less effort than the associated navigation.
        - `store` operations have the lowest penalty as they are simple, local actions.
    """

    def __init__(self, rover_gemini25pro_heuristic_20: Dict[str, float] = None):
        """
        Initializes the RoverGemini25ProHeuristic0.

        :param rover_gemini25pro_heuristic_20: A dictionary of penalties for different actions.
        """
        name = "rover_gemini25pro_heuristic_20"
        super().__init__(name=name)

        if rover_gemini25pro_heuristic_20 is None:
            rover_gemini25pro_heuristic_20 = {}

        # Define penalties with default values, which can be overridden by the parameter.
        self.penalties: Dict[str, float] = {
            'navigate': rover_gemini25pro_heuristic_20.get('navigate', 0.5),
            'communicate': rover_gemini25pro_heuristic_20.get('communicate', 0.4),
            'calibrate': rover_gemini25pro_heuristic_20.get('calibrate', 0.4),
            'sample': rover_gemini25pro_heuristic_20.get('sample', 0.2),
            'image': rover_gemini25pro_heuristic_20.get('image', 0.2),
            'store': rover_gemini25pro_heuristic_20.get('store', 0.1)
        }

        self.method_scores: Dict[str, float] = {}
        self.preprocessing_time: float = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain model.

        :param model: The planning domain model.
        :param initial_node: The initial node of the search.
        :return: The result of the parent class's initialize method.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # The heuristic value of the initial node is 0, as no method has been applied yet.
        initial_h_value = 0.0
        return super().initialize(model, initial_h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model and calculates a heuristic score for each.
        The score is based on the penalties associated with the subtasks introduced by the method.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = 0.0
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network

            for subtask in subtasks:
                subtask_name = subtask.name.lower()

                # Assign score based on keywords in subtask names
                if 'navigate' in subtask_name:
                    score += self.penalties['navigate']
                elif 'communicate' in subtask_name or 'send_' in subtask_name:
                    score += self.penalties['communicate']
                elif 'calibrate' in subtask_name:
                    score += self.penalties['calibrate']
                elif 'sample' in subtask_name or 'get_soil' in subtask_name or 'get_rock' in subtask_name:
                    score += self.penalties['sample']
                elif 'image' in subtask_name or 'get_image' in subtask_name:
                    score += self.penalties['image']
                elif 'store' in subtask_name or 'drop' in subtask_name:
                    score += self.penalties['store']

            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.

        :param node: The HTNNode to evaluate.
        :return: The pre-calculated heuristic score for the method that led to this node's state.
        """
        if not hasattr(node, 'method') or node.method is None:
            return 0.0

        method_name = node.method.name
        return self.method_scores.get(method_name, 0.0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main entry point for the heuristic evaluation during search.

        :param parent_node: The parent node of the node being evaluated.
        :param node: The current node to evaluate.
        :return: The heuristic value for the node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        """Return a formal representation of the heuristic object."""
        return f"{self.__class__.__name__}({self.penalties})"

    def __str__(self):
        """Return a string representation of the heuristic object."""
        return f"Heuristic: {self.name}"

    def __output__(self):
        """Return a formatted string with detailed information about the heuristic."""
        return (
            f"Rover Gemini 2.5 Pro Domain-Dependent Heuristic info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPenalties: {self.penalties}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
