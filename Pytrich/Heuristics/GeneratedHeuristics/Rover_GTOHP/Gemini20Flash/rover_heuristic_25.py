import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic25(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_25=0.7, name="RoverGemini20FlashHeuristic25"):  # Corrected parameter name and added default value
        super().__init__(name=name)
        self.waypoint_penalty = rover_gemini20flash_heuristic_25
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.num_soil_samples = 0
        self.num_rock_samples = 0
        self.num_image_data = 0

        # Analyze initial task network to count goals
        for task in model.initial_tn:
            task_str = str(task).lower()
            if 'get_soil_data' in task_str:
                self.num_soil_samples += 1
            elif 'get_rock_data' in task_str:
                self.num_rock_samples += 1
            elif 'get_image_data' in task_str:
                self.num_image_data += 1

        self.total_tasks = self.num_soil_samples + self.num_rock_samples + self.num_image_data
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            # This is a base node. Goal already achieved. Returning heuristic cost of 0.
            return 0

        method_name = node.method.name.lower() if hasattr(node.method, 'name') else str(node.method).lower()

        h_value = 0.0

        # Prioritize tasks that collect data
        if 'get_soil_data' in method_name or 'get_rock_data' in method_name or 'get_image_data' in method_name:
            h_value += self.waypoint_penalty  # encourage visiting waypoints.

        # Penalize methods that do not make progress.
        elif 'do_navigate' in method_name or 'communicate' in method_name or 'empty_store' in method_name:
            h_value += 0.01

        elif "communicate_image_data" in method_name or 'do_calibrate' in method_name:
             h_value += 0.01

        elif "send_soil_data" in method_name or 'send_rock_data' in method_name:
            h_value += 0.01
        else:
            h_value += 0.001

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic25(waypoint_penalty={self.waypoint_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (f"Rover Gemini 2.0 Flash Heuristic 25 info:\n"  # Corrected f-string
                f"\tName: {self.name}\n"
                f"\tWaypoint Penalty: {self.waypoint_penalty}\n"
                f"\tEstimated initial soil samples: {self.num_soil_samples}\n"
                f"\tEstimated initial rock samples: {self.num_rock_samples}\n"
                f"\tEstimated initial image data: {self.num_image_data}\n"
                f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n")
