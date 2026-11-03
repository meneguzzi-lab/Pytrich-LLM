import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic36(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_36, navigate_penalty=1.0, sample_penalty=1.5, communicate_penalty=2.0, calibrate_penalty=0.8):
        super().__init__(name="RoverGemini20FlashHeuristic36")
        self.navigate_penalty = navigate_penalty
        self.sample_penalty = sample_penalty
        self.communicate_penalty = communicate_penalty
        self.calibrate_penalty = calibrate_penalty
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        h_value = 0

        # Count the number of each type of task in the subtasks
        if isinstance(node, HTNNode) and node.uncompleted_subtasks:
            for task in node.uncompleted_subtasks:
                task_name = task.name.lower()

                if 'navigate' in task_name:
                    h_value += self.navigate_penalty
                elif 'sample' in task_name:
                    h_value += self.sample_penalty
                elif 'communicate' in task_name:
                    h_value += self.communicate_penalty
                elif 'calibrate' in task_name:
                    h_value += self.calibrate_penalty
                elif 'get_image_data' in task_name:
                    h_value += self.sample_penalty  # Treat image data like sampling
                elif 'get_rock_data' in task_name or 'get_soil_data' in task_name:
                    h_value += self.sample_penalty
                elif 'empty_store' in task_name:
                  h_value += self.navigate_penalty*0.5 #Small Nav Penalty
                elif 'do_navigate1' in task_name or 'do_navigate2' in task_name:
                    h_value += self.navigate_penalty*0.75
                else:
                    h_value += self.navigate_penalty * 0.2 #Add small penalty for every other case

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic36(navigate_penalty={self.navigate_penalty}, sample_penalty={self.sample_penalty}, communicate_penalty={self.communicate_penalty}, calibrate_penalty={self.calibrate_penalty})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic36(navigate_penalty={self.navigate_penalty}, sample_penalty={self.sample_penalty}, communicate_penalty={self.communicate_penalty}, calibrate_penalty={self.calibrate_penalty})"

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic36 info:\n"
            f"\tName: {self.name}\n"
            f"\tNavigate penalty: {self.navigate_penalty}\n"
            f"\tSample penalty: {self.sample_penalty}\n"
            f"\tCommunicate penalty: {self.communicate_penalty}\n"
            f"\tCalibrate penalty: {self.calibrate_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
