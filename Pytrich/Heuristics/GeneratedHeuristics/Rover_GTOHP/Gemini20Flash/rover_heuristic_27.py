import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini20FlashHeuristic27(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_27=1.0, name="RoverGemini20FlashHeuristic27"):
        super().__init__(name=name)
        self.base_penalty = rover_gemini20flash_heuristic_27
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.facts = model.facts
        self.at_fact = next((fact for fact in self.facts if fact.name == 'at'), None)
        self.available_fact = next((fact for fact in self.facts if fact.name == 'available'), None)
        self.have_soil_analysis_fact = next((fact for fact in self.facts if fact.name == 'have_soil_analysis'), None)
        self.have_rock_analysis_fact = next((fact for fact in self.facts if fact.name == 'have_rock_analysis'), None)
        self.have_image_fact = next((fact for fact in self.facts if fact.name == 'have_image'), None)
        self.communicated_soil_data_fact = next((fact for fact in self.facts if fact.name == 'communicated_soil_data'), None)
        self.communicated_rock_data_fact = next((fact for fact in self.facts if fact.name == 'communicated_rock_data'), None)
        self.communicated_image_data_fact = next((fact for fact in self.facts if fact.name == 'communicated_image_data'), None)

        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """Estimates the heuristic value of an HTN node."""
        if not hasattr(node, 'uncompleted_tasks') or not node.uncompleted_tasks:
            return 0.0

        num_uncompleted_tasks = len(node.uncompleted_tasks)
        return num_uncompleted_tasks * self.base_penalty

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"Rover Gemini 2.0 Flash Heuristic 27 info:\n"
            f"\tName: {self.name}\n"
            f"\tBase penalty: {self.base_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
        )

    def __repr__(self):
        return f"{self.name}(base_penalty={self.base_penalty})"

    def __str__(self):
        return f"{self.name}(base_penalty={self.base_penalty})"
