import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic17(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_17=1, name="RoverGemini20FlashHeuristic17"):
        super().__init__(name=name)
        self.rover_gemini20flash_heuristic_17 = rover_gemini20flash_heuristic_17
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.num_soil_samples = 0
        self.num_rock_samples = 0
        self.num_image_data = 0
        if hasattr(model, 'abstract_tasks'):
            for task in model.abstract_tasks:
                if task.name == 'get_soil_data':
                    self.num_soil_samples +=1
                elif task.name == 'get_rock_data':
                    self.num_rock_samples += 1
                elif task.name == 'get_image_data':
                    self.num_image_data += 1

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        A simple heuristic that counts the number of unachieved tasks.
        This heuristic is admissible because it underestimates the true cost.
        It considers the number of soil data, rock data, and image data tasks.
        """
        unachieved_tasks = 0

        # check if the data communicated already in parent
        if hasattr(node, 'state') and node.state:
            if hasattr(node, 'parent') and hasattr(node.parent, 'state') and node.parent.state:
                for p in node.state:
                  if p not in node.parent.state:
                    if str(self.model.facts[p].name).startswith("communicated_"):
                       return 0.000001

        
        
        if hasattr(node, 'uncompleted_tasks'):
            uncompleted_tasks = len(node.uncompleted_tasks)
        else:
            uncompleted_tasks = self.num_soil_samples + self.num_rock_samples + self.num_image_data
        return self.rover_gemini20flash_heuristic_17 * uncompleted_tasks

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic17(rover_gemini20flash_heuristic_17={self.rover_gemini20flash_heuristic_17})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic17(rover_gemini20flash_heuristic_17={self.rover_gemini20flash_heuristic_17})"

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 17 info:\n"
            f"\tName: {self.name}\n"
            f"\trover_gemini20flash_heuristic_17 parameter: {self.rover_gemini20flash_heuristic_17}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tNum Soil Samples: {self.num_soil_samples}\n"
            f"\tNum Rock Samples: {self.num_rock_samples}\n"
            f"\tNum Image Data: {self.num_image_data}\n"
        )
