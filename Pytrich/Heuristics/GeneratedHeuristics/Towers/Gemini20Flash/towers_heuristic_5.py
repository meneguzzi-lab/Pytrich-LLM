import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for towers domain
class TowersGemini20FlashHeuristic5(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_5 = 1, name="towers_gemini20flash_heuristic_5"):
        super().__init__(name=name)
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

        # Try different possible attributes for methods
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

            # Count different types of subtasks
            move_count = 0
            exchange_count = 0
            rotate_count = 0
            select_direction_count = 0
            shift_tower_count = 0
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

                    if 'move_abstract' in subtask_lower:
                        move_count += 1
                    if 'exchange' in subtask_lower:
                        exchange_count += 1
                    if 'rotateTower' in subtask_name:
                        rotate_count += 1
                    if 'selectDirection' in subtask_name:
                        select_direction_count += 1
                    if 'shiftTower' in subtask_name:
                        shift_tower_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'select_direction_count': select_direction_count,
                'shift_tower_count': shift_tower_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': move_count + exchange_count + rotate_count
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly contributes to achieving goal.
        """
        # Methods that handle the main ACHIEVE-AT task completion
        goal_indicators = ['exchangeClear']
        return any(indicator in method_name.lower() for indicator in goal_indicators)

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
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0  # Start with 0 cost

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Small penalties for operation complexity
            score += info['move_count'] * 0.1
            score += info['exchange_count'] * 0.2
            score += info['rotate_count'] * 0.15
            score += info['select_direction_count'] * 0.05
            score += info['shift_tower_count'] * 0.02
            score += info['subtask_count'] * 0.01


            # Small bonuses (negative cost) for productive methods
            if info['is_goal_achieving']:
                score -= 0.3  # Bonus for methods that achieve goals

        # Method-specific scoring based on name patterns
        if 'exchangeClear' in method_lower:
            score -= 0.2
        elif 'rotateTower' in method_lower:
            score += 0.1
        elif 'selectDirection' in method_lower:
            score += 0.05

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic5()"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic5()"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 5 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
