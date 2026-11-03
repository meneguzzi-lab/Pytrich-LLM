import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic21(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_21=1.0, name="RoverGemini20FlashHeuristic21"):  
        super().__init__(name=name)
        self.task_costs = {}
        self.waypoint_connections = {}
        self.lander_locations = set()
        self.soil_locations = set()
        self.rock_locations = set()
        self.objective_locations = set()
        self.rover_gemini20flash_heuristic_21 = rover_gemini20flash_heuristic_21

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        self._precompute_task_costs()
        self._extract_location_data()

        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic_value(initial_node)

        return super().initialize(model, h_value)

    def _precompute_task_costs(self):
        """Estimates task costs based on domain analysis."""
        #Simplified cost estimations - could be more sophisticated based on subtasks
        self.task_costs = {
            "get_soil_data": 5,
            "send_soil_data": 7,
            "get_rock_data": 6,
            "send_rock_data": 8,
            "get_image_data": 9,
            "send_image_data": 10,
            "do_calibrate": 4,
            "do_navigate1": 3,
            "do_navigate2": 4,
            "empty_store": 2
        }

    def _extract_location_data(self):
        """Extracts waypoint connections and lander locations from initial state."""
        for fact in self.model.facts:
            fact_str = str(fact)
            if 'at_lander' in fact_str:
                parts = fact_str.split()
                if len(parts) > 2:
                    self.lander_locations.add(parts[-1])
            elif 'at_soil_sample' in fact_str:
                parts = fact_str.split()
                if len(parts) > 1:
                    self.soil_locations.add(parts[-1])

            elif 'at_rock_sample' in fact_str:
                parts = fact_str.split()
                if len(parts) > 1:
                    self.rock_locations.add(parts[-1])


    def _calculate_heuristic_value(self, node: HTNNode):
        """Calculates heuristic value for a given HTN node.
        """
        h_value = 0
        for task in node.incomplete_tasks:
            task_name = task.name
            if task_name in self.task_costs:
                h_value += self.task_costs[task_name] * self.rover_gemini20flash_heuristic_21
            else:
                h_value += 3 * self.rover_gemini20flash_heuristic_21 #default cost

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic_value(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic21 info:\n"
            f"    Name: {self.name}\n"
            f"    Heuristic Value: {self.rover_gemini20flash_heuristic_21}\n"
            f"    Num Get Soil Tasks: {len(self.soil_locations)}\n"
            f"    Num Get Rock Tasks: {len(self.rock_locations)}\n"
            f"    Lander location: {len(self.lander_locations)}\n"
            f"    Preprocessing time: {self.preprocessing_time:.4f}s\n"
        )

    def __str__(self):
         return f"RoverGemini20FlashHeuristic21:  {self.name} {self.rover_gemini20flash_heuristic_21}"

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic21:  {self.name} {self.rover_gemini20flash_heuristic_21}"
