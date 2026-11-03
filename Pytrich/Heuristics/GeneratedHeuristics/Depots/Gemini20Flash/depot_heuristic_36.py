import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for depot domain. Aims to minimize the number of moves, loads, and unloads.
class DepotGemini20FlashHeuristic36(Heuristic):
    def __init__(self, move_penalty=1, load_penalty=2, unload_penalty=2, name="depot_gemini20flash_heuristic_36"):
        super().__init__(name=name)
        self.move_penalty = move_penalty
        self.load_penalty = load_penalty
        self.unload_penalty = unload_penalty
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
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
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
                'is_move_method': 'move' in method_name.lower(),
                'is_load_method': 'load' in method_name.lower(),
                'is_unload_method': 'unload' in method_name.lower(),
                'complexity_score': move_count + load_count + unload_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on various factors. Lower scores indicate more
        promising methods (cost-based heuristic).
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['move_count'] * self.move_penalty
            score += info['load_count'] * self.load_penalty
            score += info['unload_count'] * self.unload_penalty

        else:
            # Default penalties if method info is not available
            if 'move' in method_lower:
                score += self.move_penalty
            elif 'load' in method_lower:
                score += self.load_penalty
            elif 'unload' in method_lower:
                score += self.unload_penalty

        return score

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic36(move_penalty={self.move_penalty}, load_penalty={self.load_penalty}, unload_penalty={self.unload_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic36(move_penalty={self.move_penalty}, load_penalty={self.load_penalty}, unload_penalty={self.unload_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 36 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLoad penalty: {self.load_penalty}\n"
            f"\tUnload penalty: {self.unload_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
