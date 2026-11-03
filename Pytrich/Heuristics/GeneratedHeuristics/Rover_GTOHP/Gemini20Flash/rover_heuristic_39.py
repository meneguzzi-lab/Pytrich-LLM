import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic39(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_39=1.0, name="RoverGemini20FlashHeuristic39"):
        super().__init__(name=name)
        self.goal_proximity_weight = 0.6
        self.communication_weight = 0.4
        self.distance_weight = 0.1
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.goal_facts = model.goals  # Store goal facts for heuristic calculation
        self.initial_state = model.initial_state
        self.waypoint_objects = [obj for obj in model.facts if 'waypoint' in obj.name]
        self.lander_locations = [obj for obj in model.facts if 'at_lander' in obj.name]


        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node."""
        # If it's a relaxed plan, return small heuristic, for A*
        if hasattr(node, 'relaxed') and node.relaxed:
            return 0.001

        if not self.goal_facts:
            return 0

        # Estimate distance to goal
        goal_proximity = self._estimate_goal_proximity(node)

        # Communication is a bottleneck, give it a weigh
        communication_need = self._estimate_communication_effort(node)

        # Distance to the closest way point
        distance_estimate = self._estimate_min_distance(node)

        # Combine the hueristics, weightning for performance and accuracy
        heuristic_value = (
            self.goal_proximity_weight * goal_proximity
            + self.communication_weight * communication_need
            + self.distance_weight * distance_estimate
        )

        # Keep admissible by returning a small value. A* is only optimal with admissible heuristics
        return heuristic_value / 100

    def _estimate_goal_proximity(self, node):
        """Estimates how close the current state is to achieving the goal based on facts."""
        state = node.state
        achieved_goals = len(self.goal_facts.intersection(state))
        remaining_goals = len(self.goal_facts) - achieved_goals
        return remaining_goals

    def _estimate_communication_effort(self, node):
        """Estimates effort required for communication based on remaining data to send."""
        #Assumes that the communicate action is the only goal fact for communicate
        state = node.state
        communicated_data = sum(1 for fact in self.goal_facts if 'communicated' in fact.name)
        remaining_data = len([fact for fact in self.goal_facts if 'communicated' in fact.name]) - communicated_data
        return remaining_data

    def _estimate_min_distance(self, node):
        """Estimates minimum travel distance based on the agent's current location."""
        #Assumes that communicate action requires that all the rover's are in one location
        at_facts = [fact for fact in node.state if 'at rover' in fact.name]
        if not at_facts:
            return 10
        total_distance = 0
        at_location = at_facts[0].name.split()[-1]
        for fact in self.goal_facts:
            if 'communicated' in fact.name:
                communicated_location = fact.name.split()[2]
                if at_location != communicated_location:
                    total_distance = total_distance + 1

        return total_distance

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic39()"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic39 goal_proximity_weight= {self.goal_proximity_weight} ,communication_weight= {self.communication_weight},distance_weight= {self.distance_weight}) "

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic39 info:\n"
            f"\tGoal proximity weight: {self.goal_proximity_weight}\n"
            f"\tCommunication weight: {self.communication_weight}\n"
            f"\tDistance weight: {self.distance_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
