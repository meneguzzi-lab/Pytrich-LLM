import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic34(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_34=1.0):
        super().__init__()
        self.traverse_cost = 1.0
        self.sample_cost = 2.0
        self.communicate_cost = 3.0
        self.calibrate_cost = 1.5
        self.image_cost = 2.5
        self.distance_weight = rover_gemini20flash_heuristic_34
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_state = initial_node.state
        self.goal_facts = model.goals
        self.land_waypoints = []
        self.rock_waypoints = []
        self.soil_waypoints = []

        for fact in model.facts:
            if 'at_lander' in fact.name:
                self.land_waypoints.append(fact)
            if 'at_rock_sample' in fact.name:
                self.rock_waypoints.append(fact)
            if 'at_soil_sample' in fact.name:
                self.soil_waypoints.append(fact)
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        # h_value = sum([self.mrh.get(goal, 1) for goal in self.goal_facts if goal not in node.state])
        remaining_tasks = 0
        for task in node.uncompleted_tasks:
             remaining_tasks += 1

        num_goals = len(self.goal_facts)

        goals_in_state = 0
        for goal in self.goal_facts:
            if goal in node.state:
                goals_in_state += 1
        distance = self.get_closest_rock_sample(node.state) * self.distance_weight
        return (num_goals - goals_in_state) + distance + remaining_tasks

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f"""RoverGemini20FlashHeuristic34 info:\n	Traverse cost: {self.traverse_cost}\n	Sample cost: {self.sample_cost}\n	Communicate cost: {self.communicate_cost}\n	Calibrate cost: {self.calibrate_cost}\n	Image cost: {self.image_cost}\n	Preprocessing time: {self.preprocessing_time:.2f}s"""

    def get_closest_rock_sample(self, state):
        closest_distance = float('inf')
        for fact in self.model.facts:
            if 'at rover0' in fact.name:
                rover_location = fact

        for rock_sample in self.rock_waypoints:
            dist = self.manhattan_distance(rover_location.name, rock_sample.name)
            if dist < closest_distance:
                closest_distance = dist

        return closest_distance

    def manhattan_distance(self, pos1, pos2):
        x1, y1 = self.extract_coordinates(pos1)
        x2, y2 = self.extract_coordinates(pos2)
        return abs(x1 - x2) + abs(y1 - y2)

    def extract_coordinates(self, position_string):
        parts = position_string.split()
        waypoint = parts[2]
        x = int(waypoint[9:])
        y = 0
        return x, y