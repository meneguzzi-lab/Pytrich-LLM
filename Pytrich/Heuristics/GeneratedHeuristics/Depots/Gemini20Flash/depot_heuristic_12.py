import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Depot domain
class DepotGemini20FlashHeuristic12(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_12=1, name="depot_gemini20flash_heuristic_12"):
        super().__init__(name=name)
        self.truck_penalty = 10  # Cost associated with using a truck
        self.crane_penalty = 15  # Cost associated with using a crane
        self.hoist_penalty = 5   # Cost associated with using a hoist
        self.move_penalty  = 2    # Cost associated with moving

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
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

        # Get methods from the model
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

            # Count different types of subtasks/actions
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

                    # Identify different action/subtask types
                    if 'load' in subtask_lower or 'unload' in subtask_lower:
                        load_count += 1
                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'lift' in subtask_lower:
                        lift_count += 1

            self.method_scores[method_name] = {
                'load_count': load_count,
                'move_count': move_count,
                'lift_count': lift_count,
                'unload_count': unload_count,
                'is_truck_method': self._is_truck_method(method_name),
                'is_crane_method': self._is_crane_method(method_name),
                'is_hoist_method': self._is_hoist_method(method_name)
            }

    def _is_truck_method(self, method_name):
        return 'truck' in method_name.lower() and ('drive' in method_name.lower() or 'move' in method_name.lower())

    def _is_crane_method(self, method_name):
        return 'crane' in method_name.lower() and ('load' in method_name.lower() or 'unload' in method_name.lower())

    def _is_hoist_method(self, method_name):
        return 'hoist' in method_name.lower() and ('lift' in method_name.lower() or 'drop' in method_name.lower())

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on domain-specific costs and features.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Penalize usage of trucks, cranes, and hoists
            if info['is_truck_method']:
                score += self.truck_penalty
            if info['is_crane_method']:
                score += self.crane_penalty
            if info['is_hoist_method']:
                score += self.hoist_penalty

            # Add cost for load and move operations
            score += info['load_count'] * 3  #Arbitrary number
            score += info['move_count'] * self.move_penalty #Move operations also have a cost
            score += info['unload_count'] * 3

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic12(truck_penalty={self.truck_penalty}, crane_penalty={self.crane_penalty}, hoist_penalty={self.hoist_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic12(truck_penalty={self.truck_penalty}, crane_penalty={self.crane_penalty}, hoist_penalty={self.hoist_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 12 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tTruck penalty: {self.truck_penalty}\n"
            f"\tCrane penalty: {self.crane_penalty}\n"
            f"\tHoist penalty: {self.hoist_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
