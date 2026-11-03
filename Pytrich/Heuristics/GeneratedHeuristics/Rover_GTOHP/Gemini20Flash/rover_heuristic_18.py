import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic18(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_18=1.0, name="RoverGemini20FlashHeuristic18"):
        super().__init__(name=name)
        self.sample_data_penalty = rover_gemini20flash_heuristic_18  
        self.navigate_penalty = rover_gemini20flash_heuristic_18
        self.calibrate_penalty = rover_gemini20flash_heuristic_18
        self.communicate_penalty = rover_gemini20flash_heuristic_18
        self.preprocessed = False

    def initialize(self, model, initial_node):
        self.model = model
        self.initial_state = initial_node.state
        start = time.time()
        self.preprocess(model)
        self.preprocessing_time = time.time() - start
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def preprocess(self, model):
        self.num_soil_samples = 0
        self.num_rock_samples = 0
        self.num_image_data = 0

        # Count initial goal tasks
        for task in self.model.initial_tn:
            task_str = str(task)
            if 'get_soil_data' in task_str:
                self.num_soil_samples += 1
            elif 'get_rock_data' in task_str:
                self.num_rock_samples += 1
            elif 'get_image_data' in task_str:
                self.num_image_data += 1
        self.preprocessed = True


    def _calculate_node_heuristic(self, node: HTNNode):
        """Calculates the heuristic value of a given node."""
        if not self.preprocessed:
            raise Exception("Heuristic not preprocessed. Call preprocess() first.")

        h_value = 0

        # Remaining tasks heuristic
        remaining_soil = self.num_soil_samples
        remaining_rock = self.num_rock_samples
        remaining_image = self.num_image_data

        if node.cg:
            for achieved_task in node.cg:
                achieved_task_str = str(achieved_task)
                if 'communicate_soil_data' in achieved_task_str:
                    remaining_soil = max(0, remaining_soil - 1)
                elif 'communicate_rock_data' in achieved_task_str:
                    remaining_rock = max(0, remaining_rock - 1)
                elif 'communicate_image_data' in achieved_task_str:
                    remaining_image = max(0, remaining_image - 1)

        h_value += remaining_soil * self.sample_data_penalty
        h_value += remaining_rock * self.sample_data_penalty
        h_value += remaining_image * self.sample_data_penalty

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return f"RoverGemini20FlashHeuristic18(sample_data_penalty={self.sample_data_penalty}, navigate_penalty={self.navigate_penalty}, calibrate_penalty={self.calibrate_penalty}, communicate_penalty = {self.communicate_penalty})"

    def __output__(self):
        return self.__str__() + f"Preprocessed in {self.preprocessing_time}"
