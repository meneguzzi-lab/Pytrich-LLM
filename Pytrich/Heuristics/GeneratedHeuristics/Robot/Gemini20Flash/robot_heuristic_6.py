import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic6(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_6=1.0, name="RobotGemini20FlashHeuristic6"):
        super().__init__(name=name)
        self.reach_goal_cost = robot_gemini20flash_heuristic_6  # Adjust as needed
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
        elif hasattr(self.model, 'abstract_tasks'):  # Corrected attribute name
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
            achieve_goals_count = 0
            release_count = 0
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
                    elif 'putdown' in subtask_lower:
                        putdown_count += 1
                    elif 'move' in subtask_lower:
                        move_count += 1
                    elif 'open' in subtask_lower:
                        open_count += 1
                    elif 'achieve-goals' in subtask_lower:
                        achieve_goals_count += 1
                    elif 'release' in subtask_lower:
                        release_count += 1

            self.method_scores[method_name] = {
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'move_count': move_count,
                'open_count': open_count,
                'achieve_goals_count': achieve_goals_count,
                'release_count': release_count,
                'subtask_count': subtask_count,
                'is_goal_method': self._is_goal_method(method_name),
                'is_transport_method': self._is_transport_method(method_name)
            }

    def _is_goal_method(self, method_name):
        return 'achieve-goals' in method_name.lower()

    def _is_transport_method(self, method_name):
        transport_indicators = ['pickup', 'putdown', 'move', 'open']
        return any(indicator in method_name.lower() for indicator in transport_indicators)

    def _calculate_node_heuristic(self, node):
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['pickup_count'] * 0.2
            score += info['putdown_count'] * 0.2
            score += info['move_count'] * 0.3
            score += info['open_count'] * 0.1
            score += info['achieve_goals_count'] * 0.05
            score += info['release_count'] * 0.01
            score += info['subtask_count'] * 0.01

            if info['is_goal_method']:
                score -= 0.5

            if info['is_transport_method']:
                score += 0.1

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic6(reach_goal_cost={self.reach_goal_cost})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic6(reach_goal_cost={self.reach_goal_cost})"

    def __output__(self):
        return (
            f"RobotGemini20FlashHeuristic6 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tReach Goal Cost: {self.reach_goal_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
