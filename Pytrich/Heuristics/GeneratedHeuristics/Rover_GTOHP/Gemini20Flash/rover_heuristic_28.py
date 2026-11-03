import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic28(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_28: str = "RoverGemini20FlashHeuristic28"):
        super().__init__(name=rover_gemini20flash_heuristic_28)
        self.store_capacity = 1  
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_state = initial_node.state
        self._precompute_goal_information()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_information(self):
        self.soil_samples_needed = set()
        self.rock_samples_needed = set()
        self.image_data_needed = set()

        for goal in self.model.goals:
            goal_str = str(goal)
            if 'communicated_soil_data' in goal_str:
                waypoint = goal_str.split()[1]
                self.soil_samples_needed.add(waypoint)
            elif 'communicated_rock_data' in goal_str:
                waypoint = goal_str.split()[1]
                self.rock_samples_needed.add(waypoint)
            elif 'communicated_image_data' in goal_str:
                objective_mode = (goal_str.split()[1], goal_str.split()[2])
                self.image_data_needed.add(objective_mode)

    def _calculate_node_heuristic(self, node):
        h_value = 0

        # Penalize unachieved goals
        h_value += len(self.soil_samples_needed - self._get_achieved_soil_samples(node.state)) * 5
        h_value += len(self.rock_samples_needed - self._get_achieved_rock_samples(node.state)) * 7
        h_value += len(self.image_data_needed - self._get_achieved_image_data(node.state)) * 10

        if hasattr(node, 'method') and node.method:
            method_name = node.method.name
            if 'navigate' in method_name.lower():
                h_value += 1  # Small penalty for each navigation step

        return h_value

    def _get_achieved_soil_samples(self, state):
        achieved = set()
        for fact_id in range(state.bit_length()):
            if state & (1 << fact_id):
                fact_name = self.model.get_component(fact_id).name
                if 'communicated_soil_data' in fact_name:
                    achieved.add(fact_name.split()[1])
        return achieved

    def _get_achieved_rock_samples(self, state):
        achieved = set()
        for fact_id in range(state.bit_length()):
            if state & (1 << fact_id):
                fact_name = self.model.get_component(fact_id).name
                if 'communicated_rock_data' in fact_name:
                    achieved.add(fact_name.split()[1])
        return achieved

    def _get_achieved_image_data(self, state):
        achieved = set()
        for fact_id in range(state.bit_length()):
            if state & (1 << fact_id):
                fact_name = self.model.get_component(fact_id).name
                if 'communicated_image_data' in fact_name:
                    parts = fact_name.split()
                    achieved.add((parts[1], parts[2]))
        return achieved

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic28(store_capacity={self.store_capacity})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic28(store_capacity={self.store_capacity})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 28 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tStore capacity: {self.store_capacity}\n"
            f"\tInitial State: {self.initial_state}\n"
            f"\tSoil Samples Needed: {self.soil_samples_needed}\n"
            f"\tRock Samples Needed: {self.rock_samples_needed}\n"
            f"\tImage Data Needed: {self.image_data_needed}\n"
        )
