import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from typing import List, Union


class DepotGemini20FlashHeuristic15(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_15=1.0, name="DepotGemini20FlashHeuristic15"):
        super().__init__(name=name)
        self.depot_gemini20flash_heuristic_15 = depot_gemini20flash_heuristic_15
        self.object_locations = {}
        self.trucks = []
        self.hoists = []
        self.depots = []
        self.distributors = []
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

        # Try different possible attributes for methods
        methods = []

        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of operations in subtasks
            load_count = 0
            unload_count = 0
            move_count = 0
            lift_count = 0

            # Get subtasks from method
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

                    # Count different operation types
                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1
                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'lift' in subtask_lower:
                        lift_count += 1

            self.method_scores[method_name] = {
                'load_count': load_count,
                'unload_count': unload_count,
                'move_count': move_count,
                'lift_count': lift_count,
                'is_move_method': self._is_move_method(method_name),
                'is_load_method': self._is_load_method(method_name),
            }

    def _is_move_method(self, method_name):
        return 'move' in method_name.lower()

    def _is_load_method(self, method_name):
        return 'load' in method_name.lower() or 'unload' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['load_count'] * 0.2
            score += info['unload_count'] * 0.2
            score += info['move_count'] * 0.3
            score += info['lift_count'] * 0.1

            if info['is_move_method']:
                score += 0.2
            if info['is_load_method']:
                score += 0.1

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 10.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic15(depot_gemini20flash_heuristic_15={self.depot_gemini20flash_heuristic_15})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic15(depot_gemini20flash_heuristic_15={self.depot_gemini20flash_heuristic_15})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 15 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tDepot heuristic 15: {self.depot_gemini20flash_heuristic_15}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
