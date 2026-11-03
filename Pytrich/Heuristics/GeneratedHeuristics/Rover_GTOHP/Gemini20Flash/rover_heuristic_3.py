import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic3(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_3="RoverGemini20FlashHeuristic3", navigate_scale=1.0, sample_scale=1.0, communicate_scale=1.0, calibrate_scale=0.5):
        super().__init__(name=rover_gemini20flash_heuristic_3)
        self.navigate_scale = navigate_scale
        self.sample_scale = sample_scale
        self.communicate_scale = communicate_scale
        self.calibrate_scale = calibrate_scale
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        if not self.model:
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            subtask_count = 0
            if hasattr(method, 'task_network'):
              if method.task_network:
                subtask_count = len(method.task_network)
            elif hasattr(method, 'decomposition'):
              if method.decomposition:
                subtask_count = len(method.decomposition)

            self.method_scores[method_name] = {
                'is_navigate': 'navigate' in method_name.lower(),
                'is_sample': 'sample' in method_name.lower(),
                'is_communicate': 'communicate' in method_name.lower(),
                'is_calibrate': 'calibrate' in method_name.lower(),
                'subtask_count': subtask_count
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            if info['is_navigate']:
                score += self.navigate_scale * info['subtask_count'] # Scale by num of subtasks
            elif info['is_sample']:
                score += self.sample_scale * info['subtask_count'] # Scale by num of subtasks
            elif info['is_communicate']:
                score += self.communicate_scale * info['subtask_count'] # Scale by num of subtasks
            elif info['is_calibrate']:
                score += self.calibrate_scale * info['subtask_count'] # Scale by num of subtasks
            else:
                score += info['subtask_count'] * 0.1 # Add a small cost for other tasks

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic3(navigate_scale={self.navigate_scale}, sample_scale={self.sample_scale}, communicate_scale={self.communicate_scale}, calibrate_scale={self.calibrate_scale})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic3(navigate_scale={self.navigate_scale}, sample_scale={self.sample_scale}, communicate_scale={self.communicate_scale}, calibrate_scale={self.calibrate_scale})"

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic3 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tNavigate scale: {self.navigate_scale}\n"
            f"\tSample scale: {self.sample_scale}\n"
            f"\tCommunicate scale: {self.communicate_scale}\n"
            f"\tCalibrate scale: {self.calibrate_scale}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
