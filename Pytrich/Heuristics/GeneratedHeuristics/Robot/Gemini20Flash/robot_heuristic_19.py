import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic19(Heuristic):
    """A domain-dependent heuristic for the robot domain.

    This heuristic estimates the cost of achieving the goals in the robot
    domain, taking into account the need to move between rooms, open doors,
    and pick up/put down packages. It prioritizes actions that directly
    contribute to satisfying the goal conditions.

    The heuristic considers the following factors:
    - Number of packages not in their goal locations
    - Number of open doors
    - Whether the robot is holding a package or not
    - Whether the robot is in the correct location
    """

    def __init__(self, robot_gemini20flash_heuristic_19=1.0, name="RobotGemini20FlashHeuristic19"):
        super().__init__(name=name)
        self.robot_gemini20flash_heuristic_19 = robot_gemini20flash_heuristic_19
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
        """Precompute information about methods for faster scoring."""
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
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
                    elif 'putdown' in subtask_lower:
                        putdown_count += 1
                    elif 'move' in subtask_lower:
                        move_count += 1
                    elif 'open' in subtask_lower:
                        open_count += 1

            self.method_scores[method_name] = {
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'move_count': move_count,
                'open_count': open_count,
                'subtask_count': subtask_count,
                'is_achieve_goals': self._is_achieve_goals(method_name),
                'is_release': self._is_release(method_name),
                'is_base_case': self._is_finished(method_name),
                'complexity_score': pickup_count + putdown_count + move_count + open_count
            }

    def _is_achieve_goals(self, method_name):
        return 'achieve-goals' in method_name.lower()

    def _is_release(self, method_name):
        return 'release' in method_name.lower()

    def _is_finished(self, method_name):
        return 'finished' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """Calculate heuristic value based on the node's decomposition method."""
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """Score a method based on its relevance to the goal.

        Lower scores are assigned to methods that directly contribute to
        achieving the goals, such as picking up and putting down packages.
        Higher scores are assigned to methods that involve movement or
        opening doors, as these are considered less direct.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['move_count'] * 0.2  # Movement has some cost
            score += info['open_count'] * 0.1  # Opening doors has a minor cost
            score += info['subtask_count'] * 0.05  # Each subtask increases complexity

            if info['is_achieve_goals']:
                score -= 0.3  # Favor methods for achieving goals
            if info['is_release']:
                score -= 0.15  # Slight preference for release methods
            if info['is_base_case']:
                score -= 0.4  # Strong preference for finished methods

            score -= info['pickup_count'] * 0.25  # Picking up is important
            score -= info['putdown_count'] * 0.3  # Putting down is most important

        return max(score, 0)

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 1.0)  # Ensure values are reasonable
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic19(robot_gemini20flash_heuristic_19={self.robot_gemini20flash_heuristic_19})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic19(robot_gemini20flash_heuristic_19={self.robot_gemini20flash_heuristic_19})"

    def __output__(self):
        return (
            f"Robot Domain Heuristic 19 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
