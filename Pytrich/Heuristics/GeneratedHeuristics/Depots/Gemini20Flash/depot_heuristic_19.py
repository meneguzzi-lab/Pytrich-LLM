import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic19(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_19=1, name="DepotGemini20FlashHeuristic19"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        self.method_info = {}
        for method in self.model.decompositions:
            self.method_info[method.name] = {
                'num_subtasks': len(method.task_network)
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name
        score = 0

        if method_name in self.method_info:
            score += self.method_info[method_name]['num_subtasks']

        # Additional domain-specific heuristics can be added here
        # For example, penalize methods that involve moving trucks/hoists
        # based on estimated distances.

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}()"

    def __str__(self):
        return f"{self.name}()"

    def __output__(self):
        return (
            f"{self.name} info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tMethods analyzed: {len(self.method_info)}\n"
        )
