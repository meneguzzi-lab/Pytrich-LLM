import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic11(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_11=1, name="RoverGemini20FlashHeuristic11"):
        super().__init__(name=name)
        self.task_costs = {}
        self.waypoint_proximity_bonus = rover_gemini20flash_heuristic_11
        self.lander_waypoint = None

    def initialize(self, model, initial_node):
        self.model = model
        self._precompute_task_costs()
        self._find_lander_waypoint()

        initial_h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _precompute_task_costs(self):
        # Estimate costs for achieving each task type (soil, rock, image) based on domain actions/methods
        self.task_costs = {
            "get_soil_data": 5,  # Includes navigate, sample, send
            "get_rock_data": 5,  # Includes navigate, sample, send
            "get_image_data": 7   # Includes navigate, calibrate, take_image, send
        }

    def _find_lander_waypoint(self):
        # Identify the waypoint where the lander is located
        for fact in self.model.initial_state:
            fact_obj = self.model.get_component(fact)
            if "at_lander" in fact_obj.name:
                parts = fact_obj.name.split()
                self.lander_waypoint = parts[-1]
                return
        self.lander_waypoint = None

    def _calculate_node_heuristic(self, node):
        # Domain-dependent heuristic calculation
        h_value = 0

        # Add up the costs for all tasks on the node.uncompleted_task list
        if hasattr(node, 'uncompleted_tasks') and node.uncompleted_tasks:
            for task in node.uncompleted_tasks:
                task_name = task.name if hasattr(task, 'name') else str(task)
                if task_name in self.task_costs:
                    h_value += self.task_costs[task_name]
                else:
                    h_value += 3  # Default cost for unknown tasks

        # Prioritize tasks near the lander for efficient communication
        if self.lander_waypoint is not None:
            h_value -= self.waypoint_proximity_bonus  # Bonus for proximity

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic11(waypoint_proximity_bonus={self.waypoint_proximity_bonus})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic11(waypoint_proximity_bonus={self.waypoint_proximity_bonus})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 11 info:\n"
            f"\tName: {self.name}\n"
            f"\tTask costs: {self.task_costs}\n"
            f"\tWaypoint proximity bonus: {self.waypoint_proximity_bonus}\n"
            f"\tLander waypoint: {self.lander_waypoint}\n"
        )
