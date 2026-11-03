import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini20FlashHeuristic27(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_27=1.0, name="TowersGemini20FlashHeuristic27"):
        super().__init__(name=name)
        self.towers_gemini20flash_heuristic_27 = towers_gemini20flash_heuristic_27
        self.preprocessing_time = 0
        self.goal_predicate = "on"

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_heuristic(self, node):
        state = node.state
        goal_count = 0
        rings_to_move = 0
        heuristic_value = 0
        
        for fact in self.model.facts:
          if self.goal_predicate in fact.name and fact.name in self.model.state_explicit_repr(state):
            goal_count += 1

        rings_to_move = len([ring for ring in self.model.facts if 'r' in ring.name and 'RING' in ring.name])
        heuristic_value = rings_to_move - goal_count # Estimation of steps required to move rings to the required positons


        return heuristic_value * self.towers_gemini20flash_heuristic_27


    def __call__(self, parent_node, node):
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic27(weight={self.towers_gemini20flash_heuristic_27})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic27(weight={self.towers_gemini20flash_heuristic_27})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 27 info:\n"
            f"\tName: {self.name}\n"
            f"\tHeuristic weight: {self.towers_gemini20flash_heuristic_27}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
