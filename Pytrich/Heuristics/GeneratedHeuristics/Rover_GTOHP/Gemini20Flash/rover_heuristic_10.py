import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic10(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_10=1.0, name="RoverGemini20FlashHeuristic10"):
        super().__init__(name=name)
        self.rover_gemini20flash_heuristic_10 = rover_gemini20flash_heuristic_10
        self.at_soil_sample_cost = 1.0
        self.at_rock_sample_cost = 1.0
        self.at_image_cost = 1.0
        self.preprocessing_time = 0


    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self.calculate_heuristic(initial_node)

        return super().initialize(model, h_value)

    def calculate_heuristic(self, node: HTNNode):
        """Calculates a heuristic estimate of the cost to reach the goal from the given node.

        This heuristic considers the number of uncommunicated soil, rock, and image data, and
        estimates the cost based on a predefined weight for each type of data.
        """
        state_bitwise = node.state
        uncommunicated_soil = 0
        uncommunicated_rock = 0
        uncommunicated_image = 0

        # Count uncommunicated data
        for fact in self.model.facts:
            if 'communicated_' not in fact.name:
                continue

            fact_index = fact.global_id

            if (state_bitwise & (1 << fact_index)) == 0:
                if 'communicated_soil_data' in fact.name:
                    uncommunicated_soil += 1
                elif 'communicated_rock_data' in fact.name:
                    uncommunicated_rock += 1
                elif 'communicated_image_data' in fact.name:
                    uncommunicated_image += 1

        # Estimate cost based on the remaining uncommunicated data
        h_value = (
            uncommunicated_soil * self.at_soil_sample_cost +  #Cost for soil
            uncommunicated_rock * self.at_rock_sample_cost +  #Cost for rock
            uncommunicated_image * self.at_image_cost  #Cost for images
        ) * self.rover_gemini20flash_heuristic_10

        return h_value


    def __call__(self, parent_node, node):
        """Wrapper function for calculating the heuristic value.

        Ensures that the heuristic value is calculated and returned for a given node.
        """
        h_value = self.calculate_heuristic(node)
        super().update_info(h_value)
        return h_value


    def __repr__(self):
        return f"RoverGemini20FlashHeuristic10(soil={self.at_soil_sample_cost}, rock={self.at_rock_sample_cost}, image={self.at_image_cost})"


    def __str__(self):
        return f"RoverGemini20FlashHeuristic10(soil={self.at_soil_sample_cost}, rock={self.at_rock_sample_cost}, image={self.at_image_cost})"


    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 10 info:\n"
            f"Soil Sample Cost: {self.at_soil_sample_cost}\n"
            f"Rock Sample Cost: {self.at_rock_sample_cost}\n"
            f"Image Cost: {self.at_image_cost}\n"
            f"Preprocessing time: {self.preprocessing_time:.2f} s\n"
        )
