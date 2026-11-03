import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for rover domain. Aims to estimate remaining effort
# taking into account navigation, sampling (soil and rock), imaging and data communication.
class RoverGemini20FlashHeuristic20(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_20=1.0, name="RoverGemini20FlashHeuristic20"):
        super().__init__(name=name)
        self.rover_gemini20flash_heuristic_20 = rover_gemini20flash_heuristic_20
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic, precomputing method information and estimating the initial heuristic value.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes information about each method, such as whether it involves navigation,
        sampling, or communication, and assigns a score based on its role in the planning process.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count operations to estimate method complexity and relevance
            navigate_count = 0
            sample_count = 0
            communicate_count = 0
            calibrate_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'navigate' in subtask_lower:
                        navigate_count += 1
                    if 'sample' in subtask_lower:
                        sample_count += 1
                    if 'communicate' in subtask_lower:
                        communicate_count += 1
                    if 'calibrate' in subtask_lower:
                        calibrate_count += 1

            self.method_scores[method_name] = {
                'navigate_count': navigate_count,
                'sample_count': sample_count,
                'communicate_count': communicate_count,
                'calibrate_count': calibrate_count,
                'is_data_collection': 'get_' in method_name.lower(),
                'is_data_transfer': 'send_' in method_name.lower(),
            }


    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given HTN node.
        The heuristic estimates the cost to achieve the remaining tasks based on the chosen method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Assigns a score to a method based on its domain-specific characteristics.
        Lower scores indicate more promising methods.
        """
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Prioritize methods that directly address data collection or transfer
        if self._is_data_collection_method(method_name):
            score -= 0.3 # Give a slight bonus
        elif self._is_data_transfer_method(method_name):
            score -= 0.2  # Give a slight bonus

        if method_name in self.method_scores:
            method_data = self.method_scores[method_name]

            # Add penalties to methods that require navigation or calibration
            score += method_data['navigate_count'] * 0.1  # Navigation adds cost
            score += method_data['calibrate_count'] * 0.05 # Calibration adds cost

            # Prioritize direct actions over abstract tasks
            score += 0.01 * method_data['navigate_count']
            score += 0.01 * method_data['sample_count']

            score = min(score, 1.0)
        return max(score, 0)

    def _is_data_collection_method(self, method_name):
        return 'get_' in method_name.lower()

    def _is_data_transfer_method(self, method_name):
        return 'send_' in method_name.lower()

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic20(rover_gemini20flash_heuristic_20={self.rover_gemini20flash_heuristic_20})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic20(rover_gemini20flash_heuristic_20={self.rover_gemini20flash_heuristic_20})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 20 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tExploration weight: {self.rover_gemini20flash_heuristic_20}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
