import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic8(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_8="RoverGemini20FlashHeuristic8"):  # Ensure parameter is lowercase
        super().__init__(name=rover_gemini20flash_heuristic_8)
        self.waypoint_connections = {}
        self.min_dist_lander = float('inf')
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_node = initial_node
        self.calculate_waypoint_connections()
        self.find_min_dist_to_lander()
        self.preprocessing_time = time.time() - start_time
        initial_h_value = self.heuristic_function(initial_node)

        return super().initialize(model, initial_h_value)

    def calculate_waypoint_connections(self):
        self.waypoint_connections = {}
        for fact in self.model.facts:
            if fact.name.startswith("visible "):
                parts = fact.name.split()
                wp1 = parts[1]
                wp2 = parts[2]
                if wp1 not in self.waypoint_connections:
                    self.waypoint_connections[wp1] = []
                self.waypoint_connections[wp1].append(wp2)

    def find_min_dist_to_lander(self):
        self.min_dist_lander = float('inf')
        initial_location = None
        lander_location = None
        for fact in self.model.facts:
            if fact.name.startswith("at rover"): #Use to find inital location 
              if self.initial_node.state & fact.global_id:
                initial_location = fact.name.split()[2]
                break

        for fact in self.model.facts:
            if fact.name.startswith("at_lander general "):
                lander_location = fact.name.split()[2]
                break

        if initial_location and lander_location:
            self.min_dist_lander = self.calculate_distance(initial_location, lander_location)

    def calculate_distance(self, start, end, visited=None):
        if start == end:
            return 0
        if visited is None:
            visited = set()
        visited.add(start)
        queue = [(start, 0)]
        while queue:
            node, dist = queue.pop(0)
            if node == end:
                return dist
            if node in self.waypoint_connections:
                for neighbor in self.waypoint_connections[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        return float('inf')

    def heuristic_function(self, node):
        """
        Heuristic based on:
          - Number of remaining tasks: count of subtasks in the task network.
          - Distance to lander: manhattan distance to the lander.
          - Number of soil, rock and image data that is not communicated yet
        """
        h_value = 0

        if hasattr(node, 'uncompleted_tasks'):
            h_value += len(node.uncompleted_tasks) * 0.1
        elif hasattr(node, 'tasks'):
            h_value += len(node.tasks) * 0.1

        h_value += self.min_dist_lander * 0.05

        # Count communicated data
        communicated_soil = 0
        communicated_rock = 0
        communicated_image = 0

        for fact in self.model.facts:
            if fact.name.startswith("communicated_soil_data") and fact.global_id in node.state:
                communicated_soil += 1
            if fact.name.startswith("communicated_rock_data") and fact.global_id in node.state:
                communicated_rock += 1
            if fact.name.startswith("communicated_image_data") and fact.global_id in node.state:
                communicated_image += 1

        total_soil_tasks = 0
        total_rock_tasks = 0
        total_image_tasks = 0

        #Count all tasks
        for task in self.initial_node.uncompleted_tasks:
            task_string = str(task).lower()
            if 'get_soil_data' in task_string:
                total_soil_tasks += 1
            if 'get_rock_data' in task_string:
                total_rock_tasks += 1
            if 'get_image_data' in task_string:
                total_image_tasks += 1

        uncommunicated_soil = max(0, total_soil_tasks - communicated_soil)  # Ensure the number is not negative
        uncommunicated_rock = max(0, total_rock_tasks - communicated_rock)
        uncommunicated_image = max(0, total_image_tasks - communicated_image)

        # Penalize uncommunicated data
        h_value += (uncommunicated_soil + uncommunicated_rock + uncommunicated_image) * 0.02

        return h_value

    def __call__(self, parent_node, node):
        h_value = self.heuristic_function(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f"RoverGemini20FlashHeuristic8: \n waypoints: {len(self.waypoint_connections)}"
