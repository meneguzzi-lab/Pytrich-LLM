import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic16(Heuristic):
    def __init__(self, robot_gemini20flash_heuristic_16: str = "robot_gemini20flash_heuristic_16"):
        super().__init__(name=robot_gemini20flash_heuristic_16)
        self.pickup_putdown_penalty = 0.5
        self.move_open_penalty = 0.3
        self.goal_distance_penalty = 0.2
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

            pickup_putdown_count = 0
            move_open_count = 0
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

                    if any(term in subtask_lower for term in ['pickup', 'putdown']):
                        pickup_putdown_count += 1
                    if any(term in subtask_lower for term in ['move', 'open']):
                        move_open_count += 1

            self.method_scores[method_name] = {
                'pickup_putdown_count': pickup_putdown_count,
                'move_open_count': move_open_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'is_finished_method': self._is_finished_method(method_name),
                'complexity_score': pickup_putdown_count + move_open_count
            }

    def _is_goal_achieving_method(self, method_name):
        return 'achieve-goals' in method_name.lower()

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

            score += info['pickup_putdown_count'] * self.pickup_putdown_penalty
            score += info['move_open_count'] * self.move_open_penalty
            score += info['subtask_count'] * 0.1

            if info['is_goal_achieving']:
                score -= 0.2

            if info['is_finished_method']:
                score -= 0.3

        if 'achieve-goals' in method_lower:
            score += 0.1
        elif 'pickup' in method_lower or 'putdown' in method_lower:
            score += self.pickup_putdown_penalty
        elif 'move' in method_lower or 'open' in method_lower:
            score += self.move_open_penalty

        # Goal Distance Estimation (Simplified)
        if hasattr(node, 'state') and node.state is not None:
            goal_distance = self._estimate_goal_distance(node.state)
            score += goal_distance * self.goal_distance_penalty

        return max(score, 0)

    def _estimate_goal_distance(self, state_bitwise):
        """Estimates the distance to the goal based on unsatisfied goal predicates."""
        unsatisfied_goals = 0

        # Iterate through the goal predicates and check if they are satisfied in the current state.
        if self.model and self.model.goals:
            for goal_fact_id in self.model.goals:
                # Check if the goal fact is present in the state
                if not (state_bitwise & (1 << goal_fact_id)):
                    unsatisfied_goals += 1

        # Return the count of unsatisfied goals as the estimated distance.
        # This is a simple estimate; more sophisticated methods could consider object locations, etc.
        return unsatisfied_goals

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 10.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic(pickup_putdown={self.pickup_putdown_penalty}, move_open={self.move_open_penalty}, goal_distance={self.goal_distance_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic(pickup_putdown={self.pickup_putdown_penalty}, move_open={self.move_open_penalty}, goal_distance={self.goal_distance_penalty})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 16 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPickup/Putdown penalty: {self.pickup_putdown_penalty}\n"
            f"\tMove/Open penalty: {self.move_open_penalty}\n"
            f"\tGoal distance penalty: {self.goal_distance_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
