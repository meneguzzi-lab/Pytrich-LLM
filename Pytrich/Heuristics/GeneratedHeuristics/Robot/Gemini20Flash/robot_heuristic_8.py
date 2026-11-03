import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic8(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_8=1.0, name="RobotGemini20FlashHeuristic8"):
        super().__init__(name=name)
        self.robot_gemini20flash_heuristic_8 = robot_gemini20flash_heuristic_8
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
            achieve_goals_count = 0
            release_count = 0
            finished_count = 0
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
                    if 'achieve-goals' in subtask_lower:
                        achieve_goals_count += 1
                    if 'release' in subtask_lower:
                        release_count += 1
                    if 'finished' in subtask_lower:
                        finished_count += 1

            self.method_scores[method_name] = {
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'move_count': move_count,
                'open_count': open_count,
                'achieve_goals_count': achieve_goals_count,
                'release_count': release_count,
                'finished_count': finished_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': pickup_count + putdown_count + move_count + open_count
            }

    def _is_goal_achieving_method(self, method_name):
        return 'achieve-goals' in method_name.lower() or 'finished' in method_name.lower()

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
            score += info['open_count'] * 0.2
            score += info['subtask_count'] * 0.1

            if info['is_goal_achieving']:
                score -= 0.5

        if 'finished' in method_lower:
            score = 0
        elif 'achieve-goals' in method_lower:
            score += 0.1
        elif 'pickup' in method_lower:
            score += 0.2
        elif 'putdown' in method_lower:
            score += 0.2
        elif 'move' in method_lower:
            score += 0.3
        elif 'open' in method_lower:
            score += 0.2
        elif 'release' in method_lower:
            score += 0.1

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic8(robot_gemini20flash_heuristic_8={self.robot_gemini20flash_heuristic_8})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic8(robot_gemini20flash_heuristic_8={self.robot_gemini20flash_heuristic_8})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 8 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tRobot heuristic 8: {self.robot_gemini20flash_heuristic_8}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
