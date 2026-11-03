import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic5(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_5=1.0, name="RoverGemini20FlashHeuristic5"):
        super().__init__(name=name)
        self.navigate_penalty = 1.0
        self.sample_penalty = 2.0
        self.communicate_penalty = 3.0
        self.calibrate_penalty = 1.5
        self.drop_penalty = 0.5
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        self.method_scores = {}
        for method in self.model.decompositions:
            method_name = method.name
            score = 0
            if "navigate" in method_name:
                score += self.navigate_penalty
            if "sample" in method_name:
                score += self.sample_penalty
            if "communicate" in method_name:
                score += self.communicate_penalty
            if "calibrate" in method_name:
                score += self.calibrate_penalty
            if "drop" in method_name:
                score += self.drop_penalty
            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            #if len(node.incomplete_tasks) >0: return 1 #isinstance(node, HTNNode) and
            return 0

        method_name = node.method.name
        return self._score_method(method_name)

    def _score_method(self, method_name):
        if method_name in self.method_scores:
            return self.method_scores[method_name]
        else:
            return 0

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(navigate_penalty={self.navigate_penalty}, sample_penalty={self.sample_penalty}, communicate_penalty={self.communicate_penalty})"

    def __str__(self):
        return f"{self.name}(navigate_penalty={self.navigate_penalty}, sample_penalty={self.sample_penalty}, communicate_penalty={self.communicate_penalty})"

    def __output__(self):
        return (f"RoverGemini20FlashHeuristic5 info:\n"
                f"\tName: {self.name}\n"
                f"\tNavigate penalty: {self.navigate_penalty}\n"
                f"\tSample penalty: {self.sample_penalty}\n"
                f"\tCommunicate penalty: {self.communicate_penalty}\n"
                f"\tCalibrate penalty: {self.calibrate_penalty}\n"
                f"\tDrop penalty: {self.drop_penalty}\n"
                f"\tMethods analyzed: {len(self.method_scores)}\n"
                f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n")
