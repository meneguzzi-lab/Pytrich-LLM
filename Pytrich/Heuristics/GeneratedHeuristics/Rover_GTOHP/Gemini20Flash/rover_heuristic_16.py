import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic16(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_16=1, name="RoverGemini20FlashHeuristic16"):  # Initialize with rover_gemini20flash_heuristic_16
        super().__init__(name=name)
        self.transport_penalty = 1.0  # Cost for each navigation action
        self.sample_penalty = 1.5       # Cost for each sampling action
        self.communicate_penalty = 2.0  # Cost for each communication action
        self.calibrate_penalty = 0.8    # Cost for each calibrate action
        self.nop_penalty = 0.01 #very small
        self.visit_penalty = 0.01 #very small
        self.unvisit_penalty = 0.01 #very small
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """Precompute information about methods"""
        self.method_scores = {}
        for method in self.model.decompositions:
            method_name = method.name
            self.method_scores[method_name] = self._calculate_method_score(method)

    def _calculate_method_score(self, method):
        """Calculate score for a decomposition method based on subtasks"""
        subtasks = method.task_network
        score = 0.0

        if subtasks is not None:
            for subtask in subtasks:
                if isinstance(subtask, Operator):
                    if 'navigate' in subtask.name:
                        score += self.transport_penalty
                    elif 'sample_soil' in subtask.name or 'sample_rock' in subtask.name:
                        score += self.sample_penalty
                    elif 'communicate' in subtask.name:
                        score += self.communicate_penalty
                    elif 'calibrate' in subtask.name:
                        score += self.calibrate_penalty
                    elif 'nop' in subtask.name:
                        score += self.nop_penalty #small
                    elif 'visit' in subtask.name:
                        score += self.visit_penalty #small
                    elif 'unvisit' in subtask.name:
                        score += self.unvisit_penalty #small

        return score

    def _calculate_node_heuristic(self, node):
        """Calculate heuristic value for a node."""
        if node.compound:
            # For abstract nodes, use precomputed method score
            if node.method is not None:
                method_name = node.method.name
                return self.method_scores.get(method_name, 0.0)  # Return 0 if method not found
            else:
                return 0.0  # No method applied yet

        else:
            return 0.0  # Primitive nodes have zero heuristic

    def __call__(self, parent_node, node):
        """Heuristic function called during search."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(transport={self.transport_penalty}, sample={self.sample_penalty}, communicate={self.communicate_penalty}, calibrate={self.calibrate_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (f"{self.name} info:\n"f"\tTransport penalty: {self.transport_penalty}\n"f"\tSample penalty: {self.sample_penalty}\n"f"\tCommunicate penalty: {self.communicate_penalty}\n"f"\tCalibrate penalty: {self.calibrate_penalty}\n"f"\tMethods analyzed: {len(self.method_scores)}\n"f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n")


