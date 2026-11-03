import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic39(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_39=1.0, name="RobotGemini20FlashHeuristic39"):
        super().__init__(name=name)
        self.reach_goal_cost = robot_gemini20flash_heuristic_39  # Cost for each package that needs to reach its goal
        self.move_cost = 0.1  # Cost for each move action
        self.open_door_cost = 0.05  # Cost for opening a door
        self.pickup_cost = 0.07  # Cost for picking up a package
        self.putdown_cost = 0.06  # Cost for putting down a package
        self.preprocessed_data = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._preprocess_domain_data()
        self.preprocessing_time = time.time() - start_time

        initial_h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _preprocess_domain_data(self):
        """
        Preprocesses the domain data to extract useful information for the heuristic calculation.
        For instance, creates a mapping of goal objects to their goal locations.
        """
        self.goal_locations = {}
        for fact in self.model.facts:
            if fact.name.startswith("goal_in"):
                parts = fact.name.split()
                package = parts[1]
                location = parts[2]
                self.goal_locations[package] = location

    def _calculate_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        """
        h_value = 0

        # Iterate through the facts in the current state to identify packages that are not at their goal locations
        packages_at_goal = set()
        for fact in self.model.facts:
          if fact.name.startswith("in"):
            parts = fact.name.split()
            package = parts[1]
            location = parts[2]
            if package in self.goal_locations and self.goal_locations[package] == location:
              packages_at_goal.add(package)

        # Count the number of packages that are NOT at their goal locations and add the reach_goal_cost for each
        num_packages = 0
        for fact in self.model.facts:
          if fact.name.startswith("in"):
            num_packages +=1

        packages_not_at_goal = num_packages - len(packages_at_goal)
        h_value += packages_not_at_goal * self.reach_goal_cost

        # Penalize based on the number of moves, open door, pickup, putdown.
        if hasattr(node, 'applied_op') and node.applied_op is not None:
            op = node.applied_op

            if op.name == 'move':
                h_value += self.move_cost
            elif op.name == 'open':
                h_value += self.open_door_cost
            elif op.name == 'pickup':
                h_value += self.pickup_cost
            elif op.name == 'putdown':
                h_value += self.putdown_cost

        return h_value

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the current node.
        """
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic39(reach_goal_cost={self.reach_goal_cost}, move_cost={self.move_cost})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic39(reach_goal_cost={self.reach_goal_cost}, move_cost={self.move_cost})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 39 info:\n"
            f"\tName: {self.name}\n"
            f"\tReach Goal Cost: {self.reach_goal_cost}\n"
            f"\tMove Cost: {self.move_cost}\n"
            f"\tOpen Door Cost: {self.open_door_cost}\n"
            f"\tPickup Cost: {self.pickup_cost}\n"
            f"\tPutdown Cost: {self.putdown_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
