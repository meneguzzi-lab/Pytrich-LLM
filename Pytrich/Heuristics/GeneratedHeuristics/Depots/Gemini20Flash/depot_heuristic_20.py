import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini20FlashHeuristic20(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_20=1.0, name="DepotGemini20FlashHeuristic20"):
        super().__init__(name=name)
        self.depot_gemini20flash_heuristic_20 = depot_gemini20flash_heuristic_20
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        return super().initialize(model, self._calculate_heuristic(initial_node))

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
          methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = {}

    def _calculate_heuristic(self, node):
        """
        Calculate heuristic value for a given node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Score a method based on its name and subtasks.
        """
        method_name = method.name if hasattr(method, 'name') else str(method)
        score = 1.0

        # Prioritize methods that directly involve moving or lifting
        if 'move' in method_name.lower() or 'lift' in method_name.lower():
            score -= 0.2

        # Penalize methods that involve complex sequences of actions
        if 'complex' in method_name.lower():
            score += 0.3

        # Base case methods should have lower cost.
        if 'base' in method_name.lower():
           score -= 0.5

        # Reward methods that achieve top-level tasks directly
        if 'achieve' in method_name.lower():
            score -=0.1

        return score

    def __call__(self, parent_node, node):
        """
        Calculate and return the heuristic value for the given node.
        """
        h_value = self._calculate_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic20(depot_gemini20flash_heuristic_20={self.depot_gemini20flash_heuristic_20})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic20(depot_gemini20flash_heuristic_20={self.depot_gemini20flash_heuristic_20})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 20 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
