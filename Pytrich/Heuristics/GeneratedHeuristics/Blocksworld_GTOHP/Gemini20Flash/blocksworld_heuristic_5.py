import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from typing import List, Union


class BlocksworldGemini20FlashHeuristic5(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_5:str = "BlocksworldGemini20FlashHeuristic5"): #Naming the parameter blocksworld_gemini20flash_heuristic_5 according to instructions
        super().__init__(name=blocksworld_gemini20flash_heuristic_5)
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
        elif hasattr(self.model, 'abstract_tasks'):  # Corrected attribute name
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            
            subtask_count = 0
            if hasattr(method, 'task_network'):
                subtask_count = len(method.task_network)
            elif hasattr(method, 'decomposition'):
                subtask_count = len(method.decomposition)
            elif hasattr(method, 'subtasks'):
                subtask_count = len(method.subtasks)
                

            self.method_scores[method_name] = {
                'subtask_count': subtask_count,
                'is_put_on_method': self._is_put_on_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'complexity_score': subtask_count # Consider subtask count as general complexity
            }

    def _is_put_on_method(self, method_name):
        return 'do_put_on' in method_name.lower()

    def _is_base_case_method(self, method_name):
        return False #No obvious base case in blocksworld

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['subtask_count'] * 0.1 #Slight penalty for more subtasks

            if info['is_put_on_method']:
                score -= 0.2  # Reward for methods directly achieving put_on goals

            if info['is_base_case']:
                score -= 0.1 #Small reward for reaching base case.

        # Heuristic refinement: Give a slight advantage to task decompositions.
        if hasattr(method, 'task_network'):
            score -= 0.05 # Slightly prefer task decomposition.
        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 1.0) # Cap the heuristic value
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
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )

