import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic23(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_23=1, name="depot_gemini20flash_heuristic_23"):
        super().__init__(name=name)
        self.name = name
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            self.method_scores[method_name] = {
                'is_load_truck': 'load-truck' in method_name.lower(),
                'is_unload_truck': 'unload-truck' in method_name.lower(),
                'is_move_truck': 'move-truck' in method_name.lower(),
                'is_lift': 'lift' in method_name.lower(),
                'is_drop': 'drop' in method_name.lower(),
                'is_achieve_goal': 'achieve_goal' in method_name.lower()
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            if info['is_load_truck']:
                score += 1
            if info['is_unload_truck']:
                score += 1
            if info['is_move_truck']:
                score += 2
            if info['is_lift']:
                score += 1
            if info['is_drop']:
                score += 1
            if info['is_achieve_goal']:
                score -= 3 # Reward for achieving a goal.
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
            f"Depot Gemini 2.0 Flash Heuristic 23 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
