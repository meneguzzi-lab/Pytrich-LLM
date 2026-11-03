import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic27(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_27=1, name="RobotGemini20FlashHeuristic27"):
        super().__init__(name=name)
        self.name = name
        self.robot_gemini20flash_heuristic_27 = robot_gemini20flash_heuristic_27

        self.object_locations = {}

    def initialize(self, model, initial_node):
        self.model = model
        self.preprocessing_time = time.time()

        # Precompute object locations from initial state for heuristic calculations
        self._precompute_object_locations()

        self.preprocessing_time = time.time() - self.preprocessing_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_object_locations(self):
        """Extracts the initial locations of objects from the initial state."""
        for fact in self.model.facts:
            fact_str = str(fact)
            if fact_str.startswith('(in '):
                parts = fact_str.split()
                obj = parts[1]
                loc = parts[2]
                self.object_locations[obj] = loc
            elif fact_str.startswith('(rloc '):
                parts = fact_str.split()
                loc = parts[1]
                self.robot_location = loc
            elif fact_str.startswith('(goal_in '):
                parts = fact_str.split()
                obj = parts[1]
                loc = parts[2]
                self.goal_locations = {obj:loc}


    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given HTN node.

        This heuristic estimates the cost based on the number of packages that are not
        at their goal locations and the estimated cost to move them.

        The components considered are:
            1. Misplaced Packages: Number of packages not in their goal location.
            2. Movement Cost: A simple estimate based on whether the robot and package are in the same location
               and whether doors need to be opened (not yet implemented efficiently).
        """
        misplaced_packages = 0
        movement_cost = 0

        # Iterate through each goal condition in the problem definition
        for obj, goal_loc in self.goal_locations.items():

            if obj in self.object_locations:
                current_loc = self.object_locations[obj]  # Retrieve the object's current location
            else:
                current_loc = None # Object location unknown

            # Check if the package is not at its goal location
            if current_loc != goal_loc:
                misplaced_packages += 1

                # Estimate movement cost: simple check if the robot is in the same location as the package.
                if self.robot_location != current_loc:
                    movement_cost += 1 # Robot needs to move to the package
                movement_cost += 1 # Package needs to be moved to goal.

        # Base cost on number of misplaced packages,  and a small movement cost estimate
        total_cost = misplaced_packages * self.robot_gemini20flash_heuristic_27 + movement_cost * 0.5

        return total_cost

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the current node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(robot_gemini20flash_heuristic_27={self.robot_gemini20flash_heuristic_27})"

    def __str__(self):
        return f"{self.name}(robot_gemini20flash_heuristic_27={self.robot_gemini20flash_heuristic_27})"

    def __output__(self):
        return (
            f"{self.name} info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tRobot Heuristic 27 value: {self.robot_gemini20flash_heuristic_27}\n"
        )
