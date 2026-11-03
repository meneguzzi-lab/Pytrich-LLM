import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic0(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_0=1.0, name="DepotGemini20FlashHeuristic0"):
        super().__init__(name=name)
        self.truck_move_cost = depot_gemini20flash_heuristic_0
        self.hoist_cost = depot_gemini20flash_heuristic_0
        self.deliver_cost = depot_gemini20flash_heuristic_0
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

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
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

            move_truck_count = 0
            hoist_count = 0
            deliver_count = 0
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

                    if 'move-truck' in subtask_lower:
                        move_truck_count += 1
                    if 'hoist' in subtask_lower:
                        hoist_count += 1
                    if 'deliver' in subtask_lower:
                        deliver_count += 1

            self.method_scores[method_name] = {
                'move_truck_count': move_truck_count,
                'hoist_count': hoist_count,
                'deliver_count': deliver_count,
                'subtask_count': subtask_count,
                'is_move_truck_method': 'move_truck' in method_name.lower(),
                'is_hoist_method': 'hoist' in method_name.lower(),
                'is_deliver_method': 'deliver' in method_name.lower(),
                'is_base_case': self._is_base_case_method(method_name),
                'complexity_score': move_truck_count + hoist_count + deliver_count
            }

    def _is_base_case_method(self, method_name):
        return 'done' in method_name.lower()  # Example for Depot Domain

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

            score += info['move_truck_count'] * self.truck_move_cost
            score += info['hoist_count'] * self.hoist_cost
            score += info['deliver_count'] * self.deliver_cost

            if info['is_base_case']:
                score -= 0.1  # Small bonus for reaching a base case

        # Adjust score based on method patterns
        if 'move-truck' in method_lower:
            score += self.truck_move_cost
        elif 'hoist' in method_lower:
            score += self.hoist_cost
        elif 'deliver' in method_lower:
            score += self.deliver_cost

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic0(truck_move_cost={self.truck_move_cost}, hoist_cost={self.hoist_cost}, deliver_cost={self.deliver_cost})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic0(truck_move_cost={self.truck_move_cost}, hoist_cost={self.hoist_cost}, deliver_cost={self.deliver_cost})"

    def __output__(self):
        return (
            f"DepotGemini20FlashHeuristic0 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tTruck Move Cost: {self.truck_move_cost}\n"
            f"\tHoist Cost: {self.hoist_cost}\n"
            f"\tDeliver Cost: {self.deliver_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
