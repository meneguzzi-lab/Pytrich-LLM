import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic11(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_11_weight=1.0, name="RobotGemini20FlashHeuristic11"):
        super().__init__(name=name)
        self.robot_gemini20flash_heuristic_11_weight = robot_gemini20flash_heuristic_11_weight
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

            pickup_count = 0
            putdown_count = 0
            move_count = 0
            open_count = 0
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

                    if 'pickup' in subtask_lower:
                        pickup_count += 1
                    if 'putdown' in subtask_lower:
                        putdown_count += 1
                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'open' in subtask_lower:
                        open_count += 1

            self.method_scores[method_name] = {
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'move_count': move_count,
                'open_count': open_count,
                'subtask_count': subtask_count,
                'is_achieve_goals_method': self._is_achieve_goals_method(method_name),
                'is_release_method': self._is_release_method(method_name),
                'is_finished_method': self._is_finished_method(method_name),
                'complexity_score': pickup_count + putdown_count + move_count + open_count
            }

    def _is_achieve_goals_method(self, method_name):
        return 'achieve-goals' in method_name.lower()

    def _is_release_method(self, method_name):
        return 'release' in method_name.lower()

    def _is_finished_method(self, method_name):
        return 'finished' in method_name.lower()

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

            score += info['pickup_count'] * 0.2
            score += info['putdown_count'] * 0.2
            score += info['move_count'] * 0.3
            score += info['open_count'] * 0.1
            score += info['subtask_count'] * 0.05

            if info['is_achieve_goals_method']:
                score += 0.5
            if info['is_release_method']:
                score += 0.3
            if info['is_finished_method']:
                score -= 0.1

        return score * self.robot_gemini20flash_heuristic_11_weight

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic11(weight={self.robot_gemini20flash_heuristic_11_weight})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic11(weight={self.robot_gemini20flash_heuristic_11_weight})"

    def __output__(self):
        return (
            f"RobotGemini20FlashHeuristic11 info:\n"
            f"\tName: {self.name}\n"
            f"\tWeight: {self.robot_gemini20flash_heuristic_11_weight}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
