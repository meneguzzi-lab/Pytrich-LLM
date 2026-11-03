import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini20FlashHeuristic18(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_18=0, name="DepotGemini20FlashHeuristic18"):
        super().__init__(name=name)
        self.name = name
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

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            
            num_move_tasks = 0
            num_lift_tasks = 0
            num_drop_tasks = 0
            num_load_tasks = 0
            num_unload_tasks = 0
            
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

                    if 'move' in subtask_lower:
                        num_move_tasks += 1
                    if 'lift' in subtask_lower:
                        num_lift_tasks += 1
                    if 'drop' in subtask_lower:
                        num_drop_tasks += 1
                    if 'load' in subtask_lower:
                        num_load_tasks += 1
                    if 'unload' in subtask_lower:
                        num_unload_tasks += 1
                        
            self.method_scores[method_name] = {
                'num_move_tasks': num_move_tasks,
                'num_lift_tasks': num_lift_tasks,
                'num_drop_tasks': num_drop_tasks,
                'num_load_tasks': num_load_tasks,
                'num_unload_tasks': num_unload_tasks,
                'subtask_count': subtask_count
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
            score += info['num_move_tasks'] * 0.2
            score += info['num_lift_tasks'] * 0.3
            score += info['num_drop_tasks'] * 0.3
            score += info['num_load_tasks'] * 0.4
            score += info['num_unload_tasks'] * 0.4
            score += info['subtask_count'] * 0.1

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic18()"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic18()"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 18 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
