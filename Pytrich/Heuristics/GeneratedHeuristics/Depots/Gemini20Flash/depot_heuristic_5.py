import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for depot domain
class DepotGemini20FlashHeuristic5(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_5=1, name="depot_gemini20flash_heuristic_5"):
        super().__init__(name=name)
        self.truck_penalty = 5
        self.crane_penalty = 3
        self.hoist_penalty = 2
        self.move_penalty = 1
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

            move_count = 0
            load_count = 0
            unload_count = 0
            lift_count = 0
            subtask_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1
                    if any(lift_term in subtask_lower for lift_term in ['lift', 'drop']):
                        lift_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'load_count': load_count,
                'unload_count': unload_count,
                'lift_count': lift_count,
                'subtask_count': subtask_count,
                'is_achieve_method': self._is_achieve_method(method_name),
                'is_transport_method': self._is_transport_method(method_name),
                'complexity_score': move_count + load_count + unload_count + lift_count
            }

    def _is_achieve_method(self, method_name):
        return 'achieve' in method_name.lower()

    def _is_transport_method(self, method_name):
        transport_indicators = ['load', 'unload', 'move', 'lift', 'drop']
        return any(indicator in method_name.lower() for indicator in transport_indicators)

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

            score += info['move_count'] * self.move_penalty
            score += info['load_count'] * self.crane_penalty  # Loading is crane-related
            score += info['unload_count'] * self.hoist_penalty # Unloading is hoist-related
            score += info['lift_count'] * self.truck_penalty  # Lifting is often with a truck
            score += info['subtask_count'] * 0.5 # Add small penalty for the number of subtasks
        else:
          score = 1 # give default score to anyhting not yet precomputed.

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic5(truck={self.truck_penalty}, crane={self.crane_penalty}, hoist={self.hoist_penalty}, move={self.move_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic5(truck={self.truck_penalty}, crane={self.crane_penalty}, hoist={self.hoist_penalty}, move={self.move_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 5 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tTruck penalty: {self.truck_penalty}\n"
            f"\tCrane penalty: {self.crane_penalty}\n"
            f"\tHoist penalty: {self.hoist_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
