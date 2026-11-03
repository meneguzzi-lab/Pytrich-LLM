import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic33(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_33=1.0):
        super().__init__(name="RoverGemini20FlashHeuristic33")
        self.travel_weight = rover_gemini20flash_heuristic_33  
        self.sample_weight = rover_gemini20flash_heuristic_33
        self.communicate_weight = rover_gemini20flash_heuristic_33
        self.image_weight = rover_gemini20flash_heuristic_33
        self.calibrate_weight = rover_gemini20flash_heuristic_33
        self.preprocessed = False
        self.waypoint_soil_rock = {}
        self.waypoint_rock = {}
        self.waypoint_image = {}
        self.preprocessing_time = 0
        self.lander_waypoints = set()

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_initial_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _preprocess(self):
        """Precomputes domain-specific information."""
        if self.preprocessed:
            return

        self.waypoint_soil_rock = {}
        self.waypoint_rock = {}
        self.waypoint_image = {}
        self.lander_waypoints = set()

        for fact in self.model.facts:
            if 'at_soil_sample' in fact.name:
                waypoint = fact.name.split()[1]
                if waypoint not in self.waypoint_soil_rock:
                    self.waypoint_soil_rock[waypoint] = 0
                self.waypoint_soil_rock[waypoint] += 1
            if 'at_rock_sample' in fact.name:
                waypoint = fact.name.split()[1]
                if waypoint not in self.waypoint_soil_rock:
                    self.waypoint_soil_rock[waypoint] = 0
                if waypoint not in self.waypoint_rock:
                     self.waypoint_rock[waypoint] = 0
                self.waypoint_rock[waypoint] +=1
            if 'at_lander' in fact.name:
                waypoint = fact.name.split()[2]
                self.lander_waypoints.add(waypoint)
        self.preprocessed = True

    def _calculate_initial_heuristic(self, node):
        """Calculates the initial heuristic value based on the initial task network."""
        h_value = 0
        for task in node.task_network:
            task_name = task.name.lower()
            if 'get_soil_data' in task_name or 'get_rock_data' in task_name:
                h_value += self.sample_weight
            elif 'send_soil_data' in task_name or 'send_rock_data' in task_name:  
                h_value += self.communicate_weight
            elif 'get_image_data' in task_name:
                 h_value += self.image_weight
            elif 'do_calibrate' in task_name:
                 h_value += self.calibrate_weight
            elif 'do_navigate' in task_name:
                h_value += self.travel_weight
        return h_value

    def __call__(self, parent_node, node):
        """Calculates the heuristic value of a given node."""
        if not hasattr(node, 'task_network'):
            return 0

        h_value = self._calculate_initial_heuristic(node)  

        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f"\nRover Gemini 2.0 Flash Heuristic 33 Info:\n" \
               f"\tTravel Weight: {self.travel_weight}\n" \
               f"\tSample Weight: {self.sample_weight}\n" \
               f"\tCommunicate Weight: {self.communicate_weight}\n" \
               f"\tImage Weight: {self.image_weight}\n" \
               f"\tCalibrate Weight: {self.calibrate_weight}\n" \
               f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"

    def __repr__(self):
         return f'RoverGemini20FlashHeuristic33(rover_gemini20flash_heuristic_33={self.travel_weight})'

    def __str__(self):
        return self.__repr__()
