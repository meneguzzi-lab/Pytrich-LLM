import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini20FlashHeuristic15(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_15=1, name="BarmanGemini20FlashHeuristic15"):
        super().__init__(name=name)
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self.facts_dict = {fact.name: fact for fact in model.facts}
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)

        # Initialize heuristic value
        h = 0

        # Penalize methods that don't directly contribute to achieving cocktails.
        if 'MakeAndPourCocktail' not in method_name and 'AchieveContainsShotCocktail' not in method_name:
            h += 0.1

        # Encourage cleaning steps if things are dirty.
        if 'Clean' in method_name:
            h -= 0.05  # slight reward for cleaning actions

        # Penalize no-op methods less, as they might indicate a solved sub-problem
        if 'Null' in method_name:
            h -= 0.02

        # If the method involves pouring, especially into the final shot, prioritize it
        if 'pour-shaker-to-shot' in method_name or 'pour-shot-to-clean-shaker' in method_name:
            h -= 0.1  # slight reward for pouring towards goal


        return max(h, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic15()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic15()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 15 info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
