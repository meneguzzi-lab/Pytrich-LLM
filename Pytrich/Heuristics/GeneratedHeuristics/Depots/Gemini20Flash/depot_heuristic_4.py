import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic4(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_4=1, name="depot_gemini20flash_heuristic_4"):
        super().__init__(name=name)
        self.truck_cost = 10
        self.hoist_cost = 5
        self.move_cost = 1
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

            move_count = 0
            load_count = 0
            unload_count = 0
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

            self.method_scores[method_name] = {
                'move_count': move_count,
                'load_count': load_count,
                'unload_count': unload_count,
                'subtask_count': subtask_count,
                'is_achieve_goal': self._is_achieve_goal_method(method_name),
                'is_move_truck': self._is_move_truck_method(method_name),
                'is_handle_crane': self._is_handle_crane_method(method_name),
                'is_base_case': self._is_base_case_method(method_name)
            }

    def _is_achieve_goal_method(self, method_name):
        return 'achieve_goal' in method_name.lower()

    def _is_move_truck_method(self, method_name):
        return 'move_truck' in method_name.lower()

    def _is_handle_crane_method(self, method_name):
        return 'handle_crane' in method_name.lower()

    def _is_base_case_method(self, method_name):
        base_indicators = ['base_case']
        return any(indicator in method_name.lower() for indicator in base_indicators)

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

            score += info['move_count'] * self.move_cost
            score += info['load_count'] * self.hoist_cost  # Loading/unloading with crane
            score += info['unload_count'] * self.hoist_cost # Loading/unloading with crane

            if info['is_achieve_goal']:
                score -= 1  # Encourage goal achievement

            if info['is_base_case']:
                score -= 0.5  # Slight preference for base cases

        if 'move_truck' in method_lower:
            score += self.truck_cost
        elif 'handle_crane' in method_lower:
            score += self.hoist_cost

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic4()"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic4()"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 4 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tTruck cost: {self.truck_cost}\n"
            f"\tHoist cost: {self.hoist_cost}\n"
            f"\tMove cost: {self.move_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
