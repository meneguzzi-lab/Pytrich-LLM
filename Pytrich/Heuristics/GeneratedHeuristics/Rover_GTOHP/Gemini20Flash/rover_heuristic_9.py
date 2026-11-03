import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic9(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_9=1.0):
        super().__init__(name='RoverGemini20FlashHeuristic9')
        self.soil_penalty = 1.0
        self.rock_penalty = 1.0
        self.image_penalty = 1.0
        self.calibrate_penalty = 0.5
        self.navigate_penalty = 0.2
        self.communicate_penalty = 0.3
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_counts()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_counts(self):
        self.num_soil_goals = 0
        self.num_rock_goals = 0
        self.num_image_goals = 0

        for goal in self.model.goals:
            goal_str = str(goal)
            if 'communicated_soil_data' in goal_str:
                self.num_soil_goals += 1
            elif 'communicated_rock_data' in goal_str:
                self.num_rock_goals += 1
            elif 'communicated_image_data' in goal_str:
                self.num_image_goals += 1

    def _calculate_node_heuristic(self, node):
        h_value = 0
        #Prioritize uncompleted tasks
        h_value += self.soil_penalty * self.num_soil_goals
        h_value += self.rock_penalty * self.num_rock_goals
        h_value += self.image_penalty * self.num_image_goals
        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"Rover Gemini Flash 2.0 Heuristic 9 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tSoil penalty: {self.soil_penalty}\n"
            f"\tRock penalty: {self.rock_penalty}\n"
            f"\tImage penalty: {self.image_penalty}\n"
        )

    def __str__(self):
        return self.__output__()

    def __repr__(self):
        return self.__output__()