import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class BarmanGemini20FlashHeuristic1(Heuristic):
    def __init__(self, barman_gemini20flash_heuristic_1_name="BarmanGemini20FlashHeuristic1"): # Renamed parameter
        super().__init__(name=barman_gemini20flash_heuristic_1_name)
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


        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            # Count subtasks
            subtask_count = 0
            clean_count = 0
            pour_count = 0
            fill_count = 0
            shake_count = 0
            holding_count = 0
            handempty_count = 0
            ontable_count = 0
            # get subtasks
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

                    if 'clean' in subtask_lower:
                        clean_count += 1
                    if 'pour' in subtask_lower:
                        pour_count += 1
                    if 'fill' in subtask_lower:
                        fill_count += 1
                    if 'shake' in subtask_lower:
                        shake_count += 1
                    if 'holding' in subtask_lower:
                        holding_count += 1
                    if 'handempty' in subtask_lower:
                        handempty_count += 1
                    if 'ontable' in subtask_lower:
                        ontable_count += 1

            self.method_scores[method_name] = {
                'subtask_count': subtask_count,
                'clean_count': clean_count,
                'pour_count': pour_count,
                'fill_count': fill_count,
                'shake_count': shake_count,
                'holding_count': holding_count,
                'handempty_count': handempty_count,
                'ontable_count': ontable_count,
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

            score += info['subtask_count'] * 0.1  # Each subtask adds a bit to the cost
            score += info['clean_count'] * 0.5    # Cleaning steps are relatively expensive
            score += info['pour_count'] * 0.3     # Pouring is a key operation
            score += info['fill_count'] * 0.4     # Filling is also important
            score += info['shake_count'] * 0.6    # Shaking is expensive, requires two ingredients at the right level
            score += info['holding_count'] * 0.2   # Holding might mean picking up/dropping, moderate cost
            score += info['handempty_count'] * 0.2 # Making a hand empty has a cost
            score += info['ontable_count'] * 0.2 # Putting a thing on the table has a cost

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini20FlashHeuristic1()"

    def __str__(self):
        return f"BarmanGemini20FlashHeuristic1()"

    def __output__(self):
        return (
            f"Barman Gemini 2.0 Flash Heuristic 1 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
