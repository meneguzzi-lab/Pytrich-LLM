import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic14(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_14=1):
        super().__init__()
        self.distances = dict()
        self.available_rovers = set()

    def initialize(self, model, initial_node):
        start = time.time()
        self.model = model
        self.is_relaxed = True

        self.general_init = None
        for fact in self.model.facts:
            if fact.name == 'general':
                self.general_init = fact
                break

        self.available_rovers = {fact for fact in self.model.facts if fact.name == 'available'}
        self.soil_samples = {fact for fact in self.model.facts if fact.name == 'at_soil_sample'}
        self.rock_samples = {fact for fact in self.model.facts if fact.name == 'at_rock_sample'}
        self.image_objectives = {fact for fact in self.model.facts if fact.name == 'visible_from'}

        self.preprocess_time = time.time() - start
        h_value = self.calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def calculate_heuristic(self, node):

        # Initialize counts for each type of task
        soil_tasks = 0
        rock_tasks = 0
        image_tasks = 0
        navigate_tasks = 0

        # Iterate through the task network and count tasks
        for task in node.uncompleted_tasks:
            task_name = task.name if hasattr(task, 'name') else str(task)

            if "get_soil_data" in task_name:
                soil_tasks += 1
            elif "get_rock_data" in task_name:
                rock_tasks += 1
            elif "get_image_data" in task_name:
                image_tasks += 1
            elif "do_navigate1" in task_name or "do_navigate2" in task_name:
                navigate_tasks += 1

        # Prioritize fulfilling soil and rock data collection since it depends on stores available
        heuristic_value = (soil_tasks * 3 + rock_tasks * 3+image_tasks * 2 + navigate_tasks)
        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self.calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f'Rover Gemini 2.0 Flash Heuristic 14: Soil {len(self.soil_samples)} Rock {len(self.rock_samples)} Img {len(self.image_objectives)} Init{self.general_init}\n'
