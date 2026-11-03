import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic14(Heuristic):
    """Domain-dependent heuristic for the robot domain.

    This heuristic estimates the cost of achieving the goals in the robot domain,
    taking into account the need to move packages to their goal locations, and
    the costs associated with picking up, putting down, moving between rooms,
    and opening doors.

    The heuristic prioritizes actions that directly contribute to achieving
    the goal (i.e., putting packages in the correct locations) and penalizes
    actions that move the robot away from the goal or involve unnecessary steps.
    """

    def __init__(self, robot_gemini20flash_heuristic_14=1.0, name="RobotGemini20FlashHeuristic14"):
        super().__init__(name=name)
        self.pickup_putdown_penalty = robot_gemini20flash_heuristic_14
        self.move_penalty = 1.0
        self.open_door_penalty = 0.5
        self.holding_penalty = 0.2  #Added

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """Initializes the heuristic with the model.

        Precomputes information about methods for faster scoring.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """Precomputes information about methods for faster scoring.
        """
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
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'is_base_case': self._is_base_case_method(method_name)
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly contributes to achieving package placement goals."""
        goal_indicators = ['release-putdown_abstract']
        return any(indicator in method_name.lower() for indicator in goal_indicators)

    def _is_base_case_method(self, method_name):
        """Check if method is a base case (no further decomposition needed)."""
        base_indicators = ['finished']  # Base case method patterns
        return any(indicator in method_name.lower() for indicator in base_indicators)

    def _calculate_node_heuristic(self, node):
        """Calculate heuristic value for a node based on its decomposition method."""
        if not hasattr(node, 'method') or not node.method:
            return 0

        h_value = self._score_method(node.method, node)
        return h_value

    def _score_method(self, method, node):
        """Score a method based on various factors.

        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['pickup_count'] * self.pickup_putdown_penalty
            score += info['putdown_count'] * self.pickup_putdown_penalty
            score += info['move_count'] * self.move_penalty
            score += info['open_count'] * self.open_door_penalty

            if info['is_goal_achieving']:
                score -= 0.5  # Reward goal achieving methods
            if info['is_base_case']:
                score -= 0.2  # Slight reward for base cases

        # Additional penalties based on current state (holding)
        if hasattr(node, 'state') and any('holding' in str(fact) for fact in self.model.state_explicit_repr(node.state)):
            score += self.holding_penalty  # Penalize if already holding something

        return max(score, 0)

    def __call__(self, parent_node, node):
        """Calculates and returns the heuristic value for the given node."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic14(pickup_putdown_penalty={self.pickup_putdown_penalty}, move_penalty={self.move_penalty}, open_door_penalty={self.open_door_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic14(pickup_putdown_penalty={self.pickup_putdown_penalty}, move_penalty={self.move_penalty}, open_door_penalty={self.open_door_penalty})"

    def __output__(self):
        return (
            f"Robot Gemini 2.0 Flash Heuristic 14 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPickup/Putdown penalty: {self.pickup_putdown_penalty}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tOpen door penalty: {self.open_door_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
