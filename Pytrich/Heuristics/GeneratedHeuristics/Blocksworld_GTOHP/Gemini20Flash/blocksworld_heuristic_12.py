import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BlocksworldGemini20FlashHeuristic12(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_12: str = "BlocksworldGemini20FlashHeuristic12"):
        super().__init__(name=blocksworld_gemini20flash_heuristic_12)
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
            subtask_count = 0
            if hasattr(method, 'subtasks'):
                subtask_count = len(method.subtasks)
            elif hasattr(method, 'task_network'):
                subtask_count = len(method.task_network)
            elif hasattr(method, 'decomposition'):
                 subtask_count = len(method.decomposition)
                
            self.method_scores[method_name] = {
                'subtask_count': subtask_count,
                'is_put_on': 'do_put_on' in method_name.lower()
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method)

    def _score_method(self, method):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['subtask_count'] * 0.1

            if info['is_put_on']:
                score += 0.5  # Reward methods that directly achieve put_on goals

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic12()"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic12()"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 12 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
