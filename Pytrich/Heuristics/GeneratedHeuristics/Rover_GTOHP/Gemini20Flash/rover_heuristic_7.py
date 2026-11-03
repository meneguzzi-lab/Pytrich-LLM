import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic7(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_7: str = "RoverGemini20FlashHeuristic7"):
        super().__init__(name=rover_gemini20flash_heuristic_7)
        self.distance_weight = 1.0
        self.sample_weight = 2.0
        self.image_weight = 3.0
        self.calibrate_weight = 1.5
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_state = initial_node.state
        self.goal = model.goals
        self._precompute_waypoint_information(model)
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_initial_heuristic(model, initial_node)
        return super().initialize(model, h_value)

    def _precompute_waypoint_information(self, model):
        self.waypoint_soil_samples = set()
        self.waypoint_rock_samples = set()
        self.objective_waypoints = {}

        for fact in model.facts:
            if 'at_soil_sample' in fact.name:
                waypoint = fact.name.split()[1]
                self.waypoint_soil_samples.add(waypoint)
            elif 'at_rock_sample' in fact.name:
                waypoint = fact.name.split()[1]
                self.waypoint_rock_samples.add(waypoint)
            elif 'visible_from' in fact.name:
                parts = fact.name.split()
                objective = parts[1]
                waypoint = parts[2]
                if objective not in self.objective_waypoints:
                    self.objective_waypoints[objective] = set()
                self.objective_waypoints[objective].add(waypoint)

    def _calculate_initial_heuristic(self, model, initial_node):

        num_soil_tasks = 0
        num_rock_tasks = 0
        num_image_tasks = 0

        for task in model.initial_tn:
             if 'get_soil_data' in task.name:
                 num_soil_tasks += 1
             elif 'get_rock_data' in task.name:
                 num_rock_tasks += 1
             elif 'get_image_data' in task.name:
                 num_image_tasks += 1
        # h_value =  (num_soil_tasks + num_rock_tasks + num_image_tasks)*0.1
        h_value = (num_soil_tasks * self.sample_weight + 
                  num_rock_tasks * self.sample_weight + 
                  num_image_tasks * self.image_weight) * 0.01

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def _calculate_node_heuristic(self, node):
        """
        A heuristic that estimates the cost of a node based on the number of tasks that need to be done.
        """

        if not hasattr(node, 'method') or not node.method:
            return 0

        score = 0
        return score



    def __repr__(self):
        return f"RoverGemini20FlashHeuristic7()"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic7()"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 7 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tDistance weight: {self.distance_weight}\n"
            f"\tSample weight: {self.sample_weight}\n"
            f"\tImage weight: {self.image_weight}\n"
            f"\tCalibrate weight: {self.calibrate_weight}\n"
        )
