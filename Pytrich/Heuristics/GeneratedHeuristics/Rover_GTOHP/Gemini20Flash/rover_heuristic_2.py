import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic2(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_2: str = "rover_gemini20flash_heuristic_2", navigation_penalty: float = 1.0, sampling_penalty: float = 2.0, communication_penalty: float = 3.0):
        super().__init__(name=rover_gemini20flash_heuristic_2)
        self.navigation_penalty = navigation_penalty
        self.sampling_penalty = sampling_penalty
        self.communication_penalty = communication_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()  # Calculate method characteristics once during initialization
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.  Scores
        relative to a single method call - and a lower score indicates
        a more promising / cheaper to execute method.
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

            # Categorize method based on name
            is_navigation = 'navigate' in method_name.lower()
            is_sampling = 'sample' in method_name.lower() or 'get_soil' in method_name.lower() or 'get_rock' in method_name.lower()
            is_communication = 'communicate' in method_name.lower() or 'send' in method_name.lower()

            # assign scores
            score = 0
            if is_navigation:
                score += self.navigation_penalty
            if is_sampling:
                score += self.sampling_penalty
            if is_communication:
                score += self.communication_penalty

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node: HTNNode):
        """
        Calculates a heuristic value for a given HTN node based on its associated method,
        considering navigation, sampling, and communication penalties.

        The heuristic aims to guide the search by assigning higher costs to methods
        involving actions that are typically more expensive or less desirable in the domain,
        such as long-distance navigation or complex communication protocols.
        """

        if not hasattr(node, 'method') or not node.method:
            return 0  # Initial abstract task network

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)

        # Retrieve precomputed score or default to a base value
        score = self.method_scores.get(method_name, 1) # Default value

        return score

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic2(navigation_penalty={self.navigation_penalty}, sampling_penalty={self.sampling_penalty}, communication_penalty={self.communication_penalty})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic2(navigation_penalty={self.navigation_penalty}, sampling_penalty={self.sampling_penalty}, communication_penalty={self.communication_penalty})"

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic2 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tNavigation penalty: {self.navigation_penalty}\n"
            f"\tSampling penalty: {self.sampling_penalty}\n"
            f"\tCommunication penalty: {self.communication_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
        )
