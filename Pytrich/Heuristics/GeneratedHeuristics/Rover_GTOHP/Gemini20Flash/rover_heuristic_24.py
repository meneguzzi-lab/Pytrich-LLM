import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic24(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_24=1.0):
        super().__init__()
        self.at_soil_sample_cost = 1.0  
        self.at_rock_sample_cost = 1.0
        self.visible_cost = 0.5
        self.do_navigate1_cost = 2.0
        self.do_navigate2_cost = 1.0
        self.get_image_data_cost = 3.0
        self.at_lander_bonus = -1.0
        self.communicated_data_bonus = -2.0
        self.high_available_rover_cost = 0.1

    def initialize(self, model, initial_node):
        self.model = model
        self.goal = model.goals
        h_value = self.calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def calculate_heuristic(self, node: HTNNode):
        h_value = 0.0

        # Prioritize tasks that get soil data, rock data and image data
        for task in node.incomplete_tasks:
            task_str = str(task).lower()
            if 'get_soil_data' in task_str:
                h_value += self.at_soil_sample_cost
            elif 'get_rock_data' in task_str:
                h_value += self.at_rock_sample_cost
            elif 'get_image_data' in task_str:
                h_value += self.get_image_data_cost
            elif 'do_navigate1' in task_str:
                h_value += self.do_navigate1_cost  # Encourage navigation tasks
            elif 'do_navigate2' in task_str:
                h_value += self.do_navigate2_cost

        # Communicate data is benificial
        for f in self.goal:
            fact_str = str(f).lower()
            if 'communicated_soil_data' in fact_str or 'communicated_rock_data' in fact_str or 'communicated_image_data' in fact_str:
                h_value += self.communicated_data_bonus  # Apply the bonus for communicated data
        
        # If rover is at lander reduce heuristic (goal oriented)
        for f in node.state:
            fact_str = str(f).lower()
            if 'at_lander' in fact_str: #and 'rover' in fact_str:
                 h_value += self.at_lander_bonus
            if 'available rover' in fact_str: #and 'rover' in fact_str:
                 h_value += self.high_available_rover_cost
            if 'visible' in fact_str:
                h_value += self.visible_cost

        return h_value

    def __call__(self, parent_node, node):
        h_value = self.calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return "Rover Gemini 2.0 Flash Heuristic v24"
