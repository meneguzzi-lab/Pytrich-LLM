import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic15(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_15=1.0):
        super().__init__()
        self.sample_data_reward = rover_gemini20flash_heuristic_15
        self.move_cost = 1.0
        self.calibrate_cost = 1.2
        self.communicate_cost = 0.9
        self.store_penalty = 0.5
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.num_soil_samples = 0
        self.num_rock_samples = 0
        self.num_image_tasks = 0

        for fact in model.facts:
            if 'at_soil_sample' in fact.name:
                self.num_soil_samples += 1
            if 'at_rock_sample' in fact.name:
                self.num_rock_samples += 1

        if hasattr(model, 'abstract_tasks'):
            for task in model.abstract_tasks:
                 if 'get_image_data' in task.name:
                      self.num_image_tasks+=1
                      
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        """Calculates the heuristic value for a given node.
        This heuristic estimates the cost to complete the remaining tasks,
        giving higher rewards for tasks that gather sample data and estimates
        lower costs for doing nothing.
        """
        h_value = 0.0

        if hasattr(node, 'uncompleted_tasks') and node.uncompleted_tasks:
            for task in node.uncompleted_tasks:
                task_name = task.name

                if 'get_soil_data' in task_name or 'get_rock_data' in task_name:
                    h_value -= self.sample_data_reward #Give reward for achieving the collection of sample
                elif 'get_image_data' in task_name:
                    h_value -= (self.sample_data_reward / 2)
                elif 'do_navigate1' in task_name or 'do_navigate2' in task_name:
                     h_value += self.move_cost   # Flat Move cost
                elif  'empty_store' in task_name: #Give penalty for not using the store
                     h_value += self.store_penalty  

                # Sending and calibrate should have no cost
                elif 'send_soil_data' in task_name or 'send_rock_data' in task_name or  'send_image_data' in task_name:
                    h_value += self.communicate_cost
                elif 'do_calibrate' in task_name: #Calibrating has some penalty cost
                   h_value += self.calibrate_cost

                else:
                    pass  # Ignore other task types (nop, visit, unvisit)

        return h_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic15(sample_data_reward={self.sample_data_reward}, move_cost={self.move_cost})"

    def __str__(self):
        return f"RoverGemini20FlashHeuristic15(sample_data_reward={self.sample_data_reward}, move_cost={self.move_cost})"

    def __output__(self):
        return (f"Rover Gemini 2.0 Flash Heuristic 15 info:\n"
                f"\tName: {self.name}\n"
                f"\tSample Data Reward: {self.sample_data_reward}\n"
                f"\tMove Cost: {self.move_cost}\n"
                f"\tCalibrate Cost: {self.calibrate_cost}\n"
                f"\tCommunicate Cost: {self.communicate_cost}\n"
                f"\tStore Penalty: {self.store_penalty}\n"
                f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
                f"\tSoil Samples: {self.num_soil_samples}\n"
                f"\tRock Samples: {self.num_rock_samples}\n"
                f"\tImage Tasks: {self.num_image_tasks}\n")