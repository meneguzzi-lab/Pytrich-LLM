import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class DepotGemini20FlashHeuristic28(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_28=1, name="DepotGemini20FlashHeuristic28"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0
        self.refuel_cost = 2  # Cost of refuelling action
        self.move_cost = 1    # Cost of moving between locations
        self.hoist_cost = 1  # Cost of hoisting an object


    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):

        self.method_scores = {}

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

            # Count different types of subtasks
            refuel_count = 0
            move_count = 0
            hoist_count = 0
            deliver_count = 0
            subtask_count = 0

            # Get subtasks from method
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

                    # Count different operation types
                    if 'refuel' in subtask_lower:
                        refuel_count += 1
                    if 'move' in subtask_lower:
                        move_count += 1
                    if any(substring in subtask_lower for substring in ['hoist', 'load', 'unload']):
                        hoist_count += 1
                    if 'deliver' in subtask_lower:
                        deliver_count += 1

            self.method_scores[method_name] = {
                'refuel_count': refuel_count,
                'move_count': move_count,
                'hoist_count': hoist_count,
                'deliver_count': deliver_count,
                'subtask_count': subtask_count,
                'is_achieve_method': self._is_achieve_method(method_name),
                'is_transport_method': self._is_transport_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'complexity_score': refuel_count + move_count + hoist_count + deliver_count
            }


    def _is_achieve_method(self, method_name):
        return 'achieve' in method_name.lower()

    def _is_transport_method(self, method_name):
        transport_indicators = ['move', 'load', 'unload', 'drive']
        return any(indicator in method_name.lower() for indicator in transport_indicators)

    def _is_base_case_method(self, method_name):
        base_indicators = ['done']
        return any(indicator in method_name.lower() for indicator in base_indicators)

    def _calculate_node_heuristic(self, node):
        """Calculate heuristic value for a node."""

        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """Score a method based on its characteristics."""

        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Add costs based on actions involved in the method
            score += info['refuel_count'] * self.refuel_cost
            score += info['move_count'] * self.move_cost
            score += info['hoist_count'] * self.hoist_cost

            # Give a small bonus to methods that achieve goals
            if info['is_achieve_method']:
                score -= 0.5  # Reduced bonus, since the domain defines the subtasks ordering

        return max(score, 0)

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
            f"{self.name} info:\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tRefuel cost: {self.refuel_cost}\n"
            f"\tMove cost: {self.move_cost}\n"
            f"\tHoist cost: {self.hoist_cost}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
        )
