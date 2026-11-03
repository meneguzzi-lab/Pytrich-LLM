import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic35(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_35 = 1, name="RoverGemini20FlashHeuristic35"):
        super().__init__(name=name)
        self.navigation_weight = 1
        self.sampling_weight = 2
        self.communication_weight = 3
        self.calibration_weight = 1.5
        self.image_weight = 2.5
        self.method_scores = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.num_waypoints = len([f for f in model.facts if 'waypoint' in f.name])
        self.num_rovers = len([f for f in model.facts if 'rover' in f.name])

        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        self.method_scores = {}
        for method in self.model.decompositions:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = 1


    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given HTN node."""
        score = 0
        task_network = node.task_network

        #Prioritize task types
        for task in task_network:
           task_str = task.name if hasattr(task, 'name') else str(task)
           if 'navigate' in task_str:
              score += self.navigation_weight
           elif 'sample' in task_str:
              score += self.sampling_weight
           elif 'communicate' in task_str:
              score += self.communication_weight
           elif 'calibrate' in task_str:
              score += self.calibration_weight
           elif 'image' in task_str:
              score += self.image_weight

        # Penalize number of tasks, reward available rovers
        score += len(task_network)/ (self.num_rovers + 1)


        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic35(navigation_weight={self.navigation_weight}, sampling_weight={self.sampling_weight}, communication_weight={self.communication_weight})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic35: Weights(Nav={self.navigation_weight}, Samp={self.sampling_weight}, Comm={self.communication_weight})"

    def __output__(self):
        return f"Rover Gemini 2.0 Flash Heuristic 35 info:Name: {self.name}Methods analyzed: {len(self.method_scores)}Navigation weight: {self.navigation_weight}Sampling weight: {self.sampling_weight}Communication weight: {self.communication_weight}Preprocessing time: {self.preprocessing_time:.2f} sNum waypoints: {self.num_waypoints}Num rovers: {self.num_rovers}"
