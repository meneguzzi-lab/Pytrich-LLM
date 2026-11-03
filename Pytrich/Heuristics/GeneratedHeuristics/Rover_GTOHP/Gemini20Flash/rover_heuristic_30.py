import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic30(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_30=1.0, name="RoverGemini20FlashHeuristic30"):
        super().__init__(name=name)
        self.reach_lander_cost = rover_gemini20flash_heuristic_30  # Cost for reaching the lander
        self.sample_cost = 1.0  # Cost for each sample (soil or rock)
        self.image_cost = 1.5  # Cost for taking an image
        self.calibrate_cost = 0.8  # Cost for calibrating a camera
        self.preprocessing_time = 0
        self.at_lander_predicate = None
        self.communicated_soil_data_predicate = None
        self.communicated_rock_data_predicate = None
        self.communicated_image_data_predicate = None

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        # Find relevant predicates
        for fact in model.facts:
            if fact.name == 'at_lander':
                self.at_lander_predicate = fact
            elif fact.name == 'communicated_soil_data':
                self.communicated_soil_data_predicate = fact
            elif fact.name == 'communicated_rock_data':
                self.communicated_rock_data_predicate = fact
            elif fact.name == 'communicated_image_data':
                self.communicated_image_data_predicate = fact
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node: HTNNode):
        """
        Calculates the heuristic value for a given node by estimating the cost to achieve
        the remaining tasks. Considers sampling, imaging and reaching the lander. Small float numbers ensure admissibility.
        """
        h_value = 0.0

        # Estimate cost of remaining tasks, goal facts
        if self.communicated_soil_data_predicate:
            h_value += sum(1 for goal in self.model.goals if goal.name == self.communicated_soil_data_predicate.name) * self.sample_cost
        if self.communicated_rock_data_predicate:
            h_value += sum(1 for goal in self.model.goals if goal.name == self.communicated_rock_data_predicate.name) * self.sample_cost
        if self.communicated_image_data_predicate:
            h_value += sum(1 for goal in self.model.goals if goal.name == self.communicated_image_data_predicate.name) * self.image_cost

        # Add a heuristic for needing to reach the lander.  This assumes that at_lander must always be called for success.
        if self.at_lander_predicate: #Assumes goal reached if all samples taken and at lander.
             h_value +=  self.reach_lander_cost if self.at_lander_predicate not in node.state else 0.0

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic30(reach_lander_cost={self.reach_lander_cost}, sample_cost={self.sample_cost}, image_cost={self.image_cost})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic30(reach_lander_cost={self.reach_lander_cost}, sample_cost={self.sample_cost}, image_cost={self.image_cost})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 30 info:\n"
            f"\tName: {self.name}\n"
            f"\tReach Lander cost: {self.reach_lander_cost}\n"
            f"\tSample cost: {self.sample_cost}\n"
            f"\tImage cost: {self.image_cost}\n"
            f"\tCalibrate cost: {self.calibrate_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f}s\n"
        )
