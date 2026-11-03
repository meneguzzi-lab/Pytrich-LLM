import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic13(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_13="RoverGemini20FlashHeuristic13"):  # Corrected parameter name
        super().__init__(name=rover_gemini20flash_heuristic_13)
        self.distance_weight = 1.0
        self.task_weight = 0.5
        self.preprocessed = False

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.soil_samples = set()
        self.rock_samples = set()
        self.image_objectives = set()
        self._preprocess_problem()
        self.preprocessing_time = time.time() - start_time
        initial_h_value = self._calculate_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _preprocess_problem(self):
        """Extract soil sample, rock sample, and objective locations from the initial state."""
        if self.preprocessed:
            return

        for fact in self.model.facts:
            fact_name = fact.name
            if 'at_soil_sample' in fact_name:
                try:
                    waypoint = fact_name.split()[1]
                    self.soil_samples.add(waypoint)
                except:
                    pass
            elif 'at_rock_sample' in fact_name:
                try:
                    waypoint = fact_name.split()[1]
                    self.rock_samples.add(waypoint)
                except:
                    pass


        self.preprocessed = True


    def _calculate_heuristic(self, node):
        """Calculate the heuristic value for a given HTN node."""
        if not hasattr(node, 'task_network') or node.task_network is None:
            return 0

        task_network = node.task_network
        if not task_network:
            return 0

        total_cost = 0.0
        soil_tasks_left = 0
        rock_tasks_left = 0
        image_tasks_left = 0

        for task in task_network:
            task_str = str(task)
            if 'get_soil_data' in task_str:
                soil_tasks_left += 1
            elif 'get_rock_data' in task_str:
                rock_tasks_left += 1
            elif 'get_image_data' in task_str:
                image_tasks_left += 1

        total_cost += soil_tasks_left * self.task_weight
        total_cost += rock_tasks_left * self.task_weight
        total_cost += image_tasks_left * self.task_weight

        return total_cost

    def __call__(self, parent_node, node):
        """Heuristic function called during search to evaluate a node."""
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(task_weight={self.task_weight}, distance_weight={self.distance_weight})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """Provides a string representation of the heuristic's configuration and state."""
        return (f"{self.name} Information:\n"f"  Task weight: {self.task_weight}\n"f"  Distance weight: {self.distance_weight}\n"f"  Preprocessing Time: {self.preprocessing_time:.4f}s\n"f"  # Soil Samples: {len(self.soil_samples)}\n"f"  # Rock Samples: {len(self.rock_samples)}\n"f"  # Objectives: {len(self.image_objectives)}")
