import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini20FlashHeuristic0(Heuristic):
    def __init__(self, robot_gemini_flash20_heuristic_0=0, name="robot_gemini20flash_heuristic_0"):
        super().__init__(name=name)
        self.pickup_penalty = 1.0
        self.putdown_penalty = 1.0
        self.move_penalty = 1.0
        self.open_penalty = 1.0
        self.goal_distance_multiplier = 0.5 # Reduce impact of distance to goals
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
                'is_goal_achieving': 'achieve-goals' in method_name.lower(),
                'complexity_score': pickup_count + putdown_count + move_count + open_count
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

            score += info['pickup_count'] * self.pickup_penalty
            score += info['putdown_count'] * self.putdown_penalty
            score += info['move_count'] * self.move_penalty
            score += info['open_count'] * self.open_penalty
            score += info['subtask_count'] * 0.1 # small penalty for more subtasks

            if info['is_goal_achieving']:
                score -= 0.2  # give a bonus to goal achieving methods

        return score

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic0(pickup_penalty={self.pickup_penalty}, putdown_penalty={self.putdown_penalty}, move_penalty={self.move_penalty}, open_penalty={self.open_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic0(pickup_penalty={self.pickup_penalty}, putdown_penalty={self.putdown_penalty}, move_penalty={self.move_penalty}, open_penalty={self.open_penalty})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 0 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPickup penalty: {self.pickup_penalty}\n"
            f"\tPutdown penalty: {self.putdown_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tOpen penalty: {self.open_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
