import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic22(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_22=1):
        super().__init__()
        self.sample_weight = 2
        self.move_weight = 1
        self.calibrate_weight = 1.5

    def initialize(self, model, initial_node):
        self.model = model
        self.initial_state = initial_node.state
        self.goal_state = model.goals
        self.num_soil_samples = 0
        self.num_rock_samples = 0
        self.num_image_data = 0

        for task in model.initial_tn:
             str_task = str(task.name).lower()
             if 'get_soil_data' in str_task:
                  self.num_soil_samples +=1
             if 'get_rock_data' in str_task:
                  self.num_rock_samples += 1
             if 'get_image_data' in str_task:
                  self.num_image_data += 1

        h_value = self._compute_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _compute_heuristic(self, node):
        # Heuristic value is a weighted sum of subgoals remaining
        soil_heuristic = self.num_soil_samples * self.sample_weight
        rock_heuristic = self.num_rock_samples * self.sample_weight
        image_heuristic = self.num_image_data * self.calibrate_weight

        # Subtract already achieved goals  (THIS IS IMPORTANT)
        # soil sample check
        for f in self.model.facts:
            str_fact = str(f.name).lower()
            if ('communicated_soil_data' in str_fact) and (f.global_id in node.state):
                soil_heuristic -= self.sample_weight
        # rock sample check
        for f in self.model.facts:
            str_fact = str(f.name).lower()
            if ('communicated_rock_data' in str_fact) and (f.global_id in node.state):
                rock_heuristic -= self.sample_weight

        # image data check
        for f in self.model.facts:
            str_fact = str(f.name).lower()
            if ('communicated_image_data' in str_fact) and (f.global_id in node.state):
                image_heuristic -= self.calibrate_weight

        # Ensure non-negative heuristic
        soil_heuristic = max(0, soil_heuristic)
        rock_heuristic = max(0, rock_heuristic)
        image_heuristic = max(0, image_heuristic)

        return soil_heuristic + rock_heuristic + image_heuristic

    def __call__(self, parent_node, node):
        h_value = self._compute_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic22(sample_weight={self.sample_weight}, move_weight={self.move_weight}, calibrate_weight={self.calibrate_weight})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic22(sample_weight={self.sample_weight}, move_weight={self.move_weight}, calibrate_weight={self.calibrate_weight})"


