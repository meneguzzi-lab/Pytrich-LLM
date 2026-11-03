import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RoverGemini20FlashHeuristic37(Heuristic):
    def __init__(self, rover_gemini20flash_heuristic_37=0, navigate_weight=1.0, sample_weight=2.0, communicate_weight=3.0, calibrate_weight=1.5, name="RoverGemini20FlashHeuristic37"):
        super().__init__(name=name)
        self.navigate_weight = navigate_weight
        self.sample_weight = sample_weight
        self.communicate_weight = communicate_weight
        self.calibrate_weight = calibrate_weight
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
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            navigate_count = 0
            sample_count = 0
            communicate_count = 0
            calibrate_count = 0

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
                    if 'navigate' in subtask_lower:
                        navigate_count += 1
                    elif 'sample' in subtask_lower:
                        sample_count += 1
                    elif 'communicate' in subtask_lower or 'send' in subtask_lower:
                        communicate_count += 1
                    elif 'calibrate' in subtask_lower:
                        calibrate_count += 1

            self.method_scores[method_name] = {
                'navigate_count': navigate_count,
                'sample_count': sample_count,
                'communicate_count': communicate_count,
                'calibrate_count': calibrate_count,
                'complexity_score': (navigate_count * self.navigate_weight +
                                     sample_count * self.sample_weight +
                                     communicate_count * self.communicate_weight +
                                     calibrate_count * self.calibrate_weight)
            }

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        method_name = method.name if hasattr(method, 'name') else str(method)
        score = 0

        if method_name in self.method_scores:
            score = self.method_scores[method_name]['complexity_score']

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini20FlashHeuristic37(navigate_weight={self.navigate_weight}, sample_weight={self.sample_weight}, communicate_weight={self.communicate_weight}, calibrate_weight={self.calibrate_weight})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"RoverGemini20FlashHeuristic37 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tNavigate weight: {self.navigate_weight}\n"
            f"\tSample weight: {self.sample_weight}\n"
            f"\tCommunicate weight: {self.communicate_weight}\n"
            f"\tCalibrate weight: {self.calibrate_weight}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
