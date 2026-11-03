import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic6(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_6="RoverGemini20FlashHeuristic6"):  
        super().__init__(name=rover_gemini20flash_heuristic_6)
        self.min_distance = 1000
        self.visited = set()
        self.soil_locations = set()
        self.rock_locations = set()
        self.image_locations = set()
        self.objective_locations = {}
        self.lander_location = None
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.visited = set()
        self.soil_locations = set()
        self.rock_locations = set()
        self.image_locations = set()
        self.objective_locations = {}
        self.lander_location = None

        for fact in model.facts:
            if 'soil' in fact.name:
                self.soil_locations.add(fact.name.split()[-1])
            if 'rock' in fact.name:
                self.rock_locations.add(fact.name.split()[-1])

            if 'visible_from' in fact.name:
                objective = fact.name.split()[1]
                waypoint = fact.name.split()[-1]
                if objective not in self.objective_locations:
                    self.objective_locations[objective] = set()
                self.objective_locations[objective].add(waypoint)
            if 'at_lander' in fact.name:
                self.lander_location = fact.name.split()[-1]


        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        A heuristic which counts the number of unvisited soil and rock sample locations
        and number of objectives that still need imaging, in addition to the distance to the lander.
        """
        soil_remaining = len(self.soil_locations)
        rock_remaining = len(self.rock_locations)
        image_remaining = len(self.objective_locations)

        current_location = None
        for fact in node.state:
            fact_obj = self.model.get_component(fact)
            if 'at rover' in fact_obj.name:
                current_location = fact_obj.name.split()[-1] 
                break  # Assuming only one rover
        
        # Estimate the Manhattan distance from the current location to the lander
        lander_distance = self._manhattan_distance(current_location, self.lander_location) if current_location and self.lander_location else 0

        # This heuristic is admissible because each goal contributes at least one unit cost
        # and the Manhattan distance is an underestimate of the travel cost.
        return (soil_remaining + rock_remaining + image_remaining + lander_distance)

    def _manhattan_distance(self, pos1, pos2):
        if not pos1 or not pos2:
            return self.min_distance
        try:
            x1 = int(pos1.split('y')[0].replace('px', ''))
            y1 = int(pos1.split('y')[1])
            x2 = int(pos2.split('y')[0].replace('px', ''))
            y2 = int(pos2.split('y')[1])
            return abs(x1 - x2) + abs(y1 - y2)
        except:
            return self.min_distance

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic6()"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic6()"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 6 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tSoil Locations: {len(self.soil_locations)}\n"
            f"\tRock Locations: {len(self.rock_locations)}\n"
            f"\tImage Locations: {len(self.objective_locations)}\n"
            f"\tLander Location: {self.lander_location}\n"
        )

