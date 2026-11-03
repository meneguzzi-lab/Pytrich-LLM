import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Depot domain
class DepotGemini20FlashHeuristic22(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_22, move_penalty=1, lift_penalty=2, store_penalty=1, deliver_penalty=3, name="depot_gemini20flash_heuristic_22"):
        super().__init__(name=name)
        self.move_penalty = move_penalty
        self.lift_penalty = lift_penalty
        self.store_penalty = store_penalty
        self.deliver_penalty = deliver_penalty
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

            move_count = 0
            lift_count = 0
            store_count = 0
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

                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'lift' in subtask_lower:
                        lift_count += 1
                    if 'store' in subtask_lower:
                        store_count += 1
                    if 'deliver' in subtask_lower:
                        deliver_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'lift_count': lift_count,
                'store_count': store_count,
                'deliver_count': deliver_count,
                'subtask_count': subtask_count,
                'is_move_method': 'move' in method_name.lower(),
                'is_lift_method': 'lift' in method_name.lower(),
                'is_store_method': 'store' in method_name.lower(),
                'is_deliver_method': 'deliver' in method_name.lower(),
                'complexity_score': move_count + lift_count + store_count + deliver_count
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
        Score a method based on various factors.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['move_count'] * self.move_penalty
            score += info['lift_count'] * self.lift_penalty
            score += info['store_count'] * self.store_penalty
            score += info['deliver_count'] * self.deliver_penalty

            # Add a small bonus for methods with fewer subtasks, encouraging simpler plans
            score -= info['subtask_count'] * 0.1

        else:
          score += 1 #default penalty for methods without precomputed info

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic22(move={self.move_penalty}, lift={self.lift_penalty}, store={self.store_penalty}, deliver={self.deliver_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic22(move={self.move_penalty}, lift={self.lift_penalty}, store={self.store_penalty}, deliver={self.deliver_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 22 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLift penalty: {self.lift_penalty}\n"
            f"\tStore penalty: {self.store_penalty}\n"
            f"\tDeliver penalty: {self.deliver_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
