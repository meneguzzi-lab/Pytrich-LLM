import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic27(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_27=1.0, name="BarmanGemini20FlashHeuristic27"):
        super().__init__(name=name)
        self.clean_shot_weight = barman_gemini20flash_heuristic_27
        self.pour_weight = 1.0
        self.holding_weight = 0.5
        self.handempty_weight = 0.2
        self.ontable_weight = 0.1
        self.contains_weight = 0.3
        self.preprocessing_time = 0
        self.method_scores = {}

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
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            clean_count = 0
            pour_count = 0
            holding_count = 0
            handempty_count = 0
            ontable_count = 0
            contains_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'holding' in subtask_lower:
                        holding_count += 1
                    if 'handempty' in subtask_lower:
                        handempty_count += 1
                    if 'ontable' in subtask_lower:
                        ontable_count += 1
                    if 'contains' in subtask_lower:
                        contains_count += 1

            self.method_scores[method_name] = {
                'clean_count': clean_count,
                'pour_count': pour_count,
                'holding_count': holding_count,
                'handempty_count': handempty_count,
                'ontable_count': ontable_count,
                'contains_count': contains_count
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

            score += info['clean_count'] * self.clean_shot_weight
            score += info['pour_count'] * self.pour_weight
            score += info['holding_count'] * self.holding_weight
            score += info['handempty_count'] * self.handempty_weight
            score += info['ontable_count'] * self.ontable_weight
            score += info['contains_count'] * self.contains_weight

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic27(clean_shot_weight={self.clean_shot_weight})"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic27(clean_shot_weight={self.clean_shot_weight})"
