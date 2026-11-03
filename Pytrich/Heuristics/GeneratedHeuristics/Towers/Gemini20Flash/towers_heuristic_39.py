import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic39(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_39=1, name="TowersGemini20FlashHeuristic39"):
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_39 = towers_gemini20flash_heuristic_39
        self.preprocessing_time = 0
        self.goal_positions = {}

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_goal_positions()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_goal_positions(self):
        """Precomputes the goal positions of each ring."""
        self.goal_positions = {}
        for fact in self.model.facts:
            if fact.name.startswith('goal_on'):
                parts = fact.name.split()
                ring = parts[1]
                location = parts[2]
                self.goal_positions[ring] = location

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        state = node.state
        h_value = 0

        # Iterate through each ring and calculate its heuristic value
        for fact in self.model.facts:
            if fact.name.startswith('on'):
                parts = fact.name.split()
                ring = parts[1]
                current_location = parts[2]
                if ring in self.goal_positions:
                    goal_location = self.goal_positions[ring]
                    if current_location != goal_location:
                        h_value += 1  # Increment if the ring is not in its goal location

        return h_value * self.towers_gemini20flash_heuristic_39 # Scale heuristic by a constant

    def __call__(self, parent_node, node):
        """Calculates the heuristic value for a given node and updates heuristic info."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic39(towers_gemini20flash_heuristic_39={self.towers_gemini20flash_heuristic_39})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic39(towers_gemini20flash_heuristic_39={self.towers_gemini20flash_heuristic_39})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 39 info:\n"
            f"\tName: {self.name}\n"
            f"\tTowers Heuristic 39: {self.towers_gemini20flash_heuristic_39}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
