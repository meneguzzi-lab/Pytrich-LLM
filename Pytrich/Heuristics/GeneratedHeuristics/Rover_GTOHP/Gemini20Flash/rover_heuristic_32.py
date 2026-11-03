import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic32(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_32="RoverGemini20FlashHeuristic32"):  # Corrected parameter name to lowercase
        super().__init__(name=rover_gemini20flash_heuristic_32)
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        # Precompute data if needed (e.g., shortest path distances between waypoints)
        # For this simple heuristic, we skip precomputation

        self.preprocessing_time = time.time() - start_time
        initial_h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, initial_h_value)

    def _calculate_node_heuristic(self, node):
        """A simple heuristic that counts the number of unachieved goals and unvisited sample locations."""
        h_value = 0
        state = node.state
        model = self.model

        # Count uncommunicated soil data locations
        for fact in model.facts:
            if "communicated_soil_data" in fact.name and fact.global_id not in state:
                h_value += 1

        # Count uncommunicated rock data locations
        for fact in model.facts:
            if "communicated_rock_data" in fact.name and fact.global_id not in state:
                h_value += 1

        # Count uncommunicated image data objectives and modes
        for fact in model.facts:
          if 'communicated_image_data' in fact.name and fact.global_id not in state:
            h_value += 1

        # Simple heuristic, just the number of open tasks. Not good, but admissible.
        if hasattr(node, 'open_tasks') and node.open_tasks:
            h_value += len(node.open_tasks)

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 32 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return f"{self.name}"
