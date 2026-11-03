import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class RobotGemini20FlashHeuristic13(Heuristic):
    """A domain-dependent heuristic for the robot domain.

    This heuristic estimates the cost of achieving the goals by considering the number
    of packages that are not in their goal locations and the cost of moving the robot
    between rooms and opening doors. It prioritizes methods that directly move packages
    to their goal locations and penalizes actions that involve moving the robot or
    opening doors.
    """

    def __init__(self, robot_gemini20flash_heuristic_13=1.0, name="RobotGemini20FlashHeuristic13"):
        super().__init__(name=name)
        self.mismatch_penalty = robot_gemini20flash_heuristic_13  # Penalty for each package in the wrong location
        self.move_penalty = 0.5       # Penalty for each move action
        self.open_door_penalty = 0.2   # Penalty for each open door action
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
        """Precomputes information about methods for faster heuristic calculation."""
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
            open_count = 0
            pickup_count = 0
            putdown_count = 0
            subtask_count = 0

            # Try different possible attributes for subtasks
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
                    if 'open' in subtask_lower:
                        open_count += 1
                    if 'pickup' in subtask_lower:
                        pickup_count += 1
                    if 'putdown' in subtask_lower:
                        putdown_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'open_count': open_count,
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'subtask_count': subtask_count,
                'complexity_score': move_count + open_count + pickup_count + putdown_count
            }

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given node.

        The heuristic value is based on the number of packages that are not in their
        goal locations, the cost of moving the robot between rooms, and the cost of
        opening doors.
        """
        state = node.state
        goal_mismatches = 0
        # Iterate through facts to find mismatches between current and goal locations
        for fact in self.model.facts:
            if fact.name.startswith('in ') and not (state & (1 << fact.global_id)):
                package = fact.name.split()[1]
                location = fact.name.split()[2]

                goal_fact_name = f'goal_in {package} {location}'
                goal_fact = next((f for f in self.model.facts if f.name == goal_fact_name), None)

                if goal_fact and (state & (1 << goal_fact.global_id)) != (state & (1 << fact.global_id)):
                    goal_mismatches += 1

        # Adjust heuristic value based on method used
        method_cost = 0

        if hasattr(node, 'method') and node.method:
            method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)

            if method_name in self.method_scores:
                info = self.method_scores[method_name]
                method_cost += info['move_count'] * self.move_penalty
                method_cost += info['open_count'] * self.open_door_penalty

        return goal_mismatches * self.mismatch_penalty + method_cost

    def __call__(self, parent_node, node):
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini20FlashHeuristic13(mismatch_penalty={self.mismatch_penalty}, move_penalty={self.move_penalty}, open_door_penalty={self.open_door_penalty})"

    def __str__(self):
        return f"RobotGemini20FlashHeuristic13(mismatch_penalty={self.mismatch_penalty}, move_penalty={self.move_penalty}, open_door_penalty={self.open_door_penalty})"
