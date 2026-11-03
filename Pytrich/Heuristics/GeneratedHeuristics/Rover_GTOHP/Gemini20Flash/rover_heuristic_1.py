import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic1(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_1="RoverGemini20FlashHeuristic1"):  # Keep parameter name all lowercase
        super().__init__(name=rover_gemini20flash_heuristic_1)
        self.preprocessing_time = 0
        self.goal_facts = set()

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_facts = model.goals  # Store goal facts for efficient access
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node: HTNNode):
        """Calculates the heuristic value for a given node.

        This heuristic estimates the cost to achieve the goal state from the current node's state.
        It considers:
            1. Number of unachieved goal facts.
            2. A bonus for methods that directly contribute to data collection (soil, rock, image).
            3. A larger penalty for methods that involve navigation, as navigation is often a bottleneck.
        It tries to keep the heuristic admissible and informed.
        """

        h_value = 0
        state = node.state

        # 1. Count unachieved goal facts
        unachieved_goals = self.goal_facts - state
        h_value += len(unachieved_goals)  # Each unachieved goal adds to the heuristic

        # Adjust the heuristic based on the method used to reach this node
        if hasattr(node, 'method') and node.method:
            method_name = node.method.name.lower()

            # 2. Add bonus for data collection
            if 'get_soil_data' in method_name or 'get_rock_data' in method_name or 'get_image_data' in method_name:
                h_value -= 0.1 # Slighty promote those nodes


            # 3. Penalize navigation methods, navigation is a frequent action so we penalize heavily for it to make the planner choose less nav heavy plans
            if 'navigate' in method_name:
                h_value += 0.4 # Heavy penalize those nodes

        # Ensure heuristic is non-negative
        return max(0, h_value)


    def __call__(self, parent_node, node):
        """Heuristic evaluation function called by the planner.

        Calculates and returns the heuristic value for a given search node.
        """
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic1 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f}s\n"
            f"\tGoal facts count: {len(self.goal_facts)}\n"
        )

    def __str__(self):
         return f"RoverGemini20FlashHeuristic1()"
