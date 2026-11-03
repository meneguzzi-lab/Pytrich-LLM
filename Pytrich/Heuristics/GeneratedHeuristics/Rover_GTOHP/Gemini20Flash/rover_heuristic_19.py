import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic19(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_19=1.0):
        super().__init__(name='RoverGemini20FlashHeuristic19')
        self.h_weight = rover_gemini20flash_heuristic_19
        self.preprocessing_time = 0
        self.sample_tasks = ['get_soil_data', 'get_rock_data']
        self.image_tasks = ['get_image_data']

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.initial_state = initial_node.state
        self.goal = model.goals
        self._precompute_location_data()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_location_data(self):
        self.soil_locations = set()
        self.rock_locations = set()
        self.objective_locations = set()
        for fact in self.model.facts:
            if 'at_soil_sample' in fact.name:
                self.soil_locations.add(fact.name.split()[1])
            if 'at_rock_sample' in fact.name:
                self.rock_locations.add(fact.name.split()[1])
            if 'visible_from' in fact.name:
                self.objective_locations.add(fact.name.split()[1])


    def _is_sample_task(self, task_name):
        for sample_task in self.sample_tasks:
            if sample_task in task_name:
                return True
        return False

    def _is_image_task(self, task_name):
        for image_task in self.image_tasks:
            if image_task in task_name:
                return True
        return False

    def _calculate_node_heuristic(self, node):
        """
        Heuristic that prioritizes:
            - Satisfying communicated data goal.
            - Visiting waypoints with rock and soil samples
            - Capturing images
        """

        h_value = 0.0

        # Count remaining tasks
        remaining_tasks = []
        if hasattr(node, 'task_network'):
            remaining_tasks = node.task_network
        elif hasattr(node, 'subtasks'):
            remaining_tasks = node.subtasks

        num_remaining_tasks = len(remaining_tasks)

        h_value += num_remaining_tasks * 0.1

        # Goal proximity - how many communicated goals are left
        goals_not_achieved = self.goal - node.state
        h_value += len(goals_not_achieved) * 0.15
        
        for task in remaining_tasks:
            task_name = str(task)
            if self._is_sample_task(task_name):
                h_value += 0.3

            if self._is_image_task(task_name):
                h_value += 0.4

        return h_value * self.h_weight

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return f'{self.name} h_weight = {self.h_weight}  PREP TIME{self.preprocessing_time}'

    def __str__(self):
        return self.__output__()

    def __repr__(self):
        return self.__output__()