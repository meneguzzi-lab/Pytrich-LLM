import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic18(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_18 ="TowersGemini20FlashHeuristic18", misplaced_rings_weight=1.0, abstract_task_penalty=0.1):
        super().__init__(name=towers_gemini20flash_heuristic_18)
        self.misplaced_rings_weight = misplaced_rings_weight
        self.abstract_task_penalty = abstract_task_penalty
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _calculate_heuristic(self, state):
        misplaced_rings = 0
        goal_rings = {}
        current_rings = {}

        # Collect goal ring positions
        for fact in self.model.facts:
            if 'goal_on' in fact.name:
                parts = fact.name.split()
                ring = parts[1]
                goal_rings[ring] = parts[2]
        
        #Collect current ring positions from the state
        state_facts = self.model.state_explicit_repr(state)
        for fact_name in state_facts:
            if 'on' in fact_name and 'goal_on' not in fact_name:
                 parts = fact_name.split()
                 ring = parts[1]
                 current_rings[ring] = parts[2]

        for ring, target_location in goal_rings.items():
            if ring not in current_rings or current_rings[ring] != target_location:
                misplaced_rings += 1

        heuristic_value = self.misplaced_rings_weight * misplaced_rings

        return heuristic_value

    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic18(misplaced_rings_weight={self.misplaced_rings_weight}, abstract_task_penalty={self.abstract_task_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic18(misplaced_rings_weight={self.misplaced_rings_weight}, abstract_task_penalty={self.abstract_task_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 18 info:\n"
            f"\tName: {self.name}\n"
            f"\tMisplaced Rings Weight: {self.misplaced_rings_weight}\n"
            f"\tAbstract Task Penalty: {self.abstract_task_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
