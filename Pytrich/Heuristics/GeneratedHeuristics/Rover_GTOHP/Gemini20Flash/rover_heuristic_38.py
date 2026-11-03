import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic38(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_38=1.0, name="RoverGemini20FlashHeuristic38"):
        super().__init__(name=name)
        self.move_cost = 1.0
        self.sample_cost = 5.0
        self.communicate_cost = 3.0
        self.calibrate_cost = 2.0
        self.preprocessing_time = 0.0
        self.objective_weight = rover_gemini20flash_heuristic_38

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_state = initial_node.state
        self._precompute_goal_information(model)
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_information(self, model):
        """Analyzes the goal and extracts relevant information for heuristic calculation."""
        self.num_soil_data = 0
        self.num_rock_data = 0
        self.num_image_data = 0

        for goal in model.goals:
            goal_str = str(goal)
            if 'communicated_soil_data' in goal_str:
                self.num_soil_data += 1
            elif 'communicated_rock_data' in goal_str:
                self.num_rock_data += 1
            elif 'communicated_image_data' in goal_str:
                self.num_image_data += 1


    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given HTN node.

        The heuristic estimates the cost to achieve the remaining tasks based on:
            - Number of soil, rock, and image data to be communicated
            - Whether the rover is at the required waypoints
            - The costs associated with different actions (move, sample, communicate)
        """
        
        remaining_soil = self.num_soil_data
        remaining_rock = self.num_rock_data
        remaining_image = self.num_image_data

        # Simple heuristic: reward objectives
        value = self.objective_weight*(remaining_soil + remaining_rock + remaining_image)
        
        return -value


    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)

        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic38(objective_weight={self.objective_weight})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic38(objective_weight={self.objective_weight})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 38 info:\n"
            f"\tName: {self.name}\n"
            f"\tObjective weight: {self.objective_weight}\n"
            f"\tMove cost: {self.move_cost}\n"
            f"\tSample cost: {self.sample_cost}\n"
            f"\tCommunicate cost: {self.communicate_cost}\n"
            f"\tCalibration cost: {self.calibrate_cost}\n"
            f"\tNumber of Soil Data: {self.num_soil_data}\n"
            f"\tNumber of Rock Data: {self.num_rock_data}\n"
            f"\tNumber of Image Data: {self.num_image_data}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
