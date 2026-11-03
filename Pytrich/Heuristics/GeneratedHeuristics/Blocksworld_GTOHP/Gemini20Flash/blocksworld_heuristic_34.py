import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Blocksworld domain
class BlocksworldGemini20FlashHeuristic34(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_34:str, misplaced_blocks_penalty=1, action_cost=1):
        super().__init__(name=blocksworld_gemini20flash_heuristic_34)
        self.misplaced_blocks_penalty = misplaced_blocks_penalty
        self.action_cost = action_cost
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
        elif hasattr(self.model, 'tasks'):
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks.  The key tasks are `do_put_on`
            put_on_count = 0
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

                    if 'do_put_on' in subtask_lower:
                        put_on_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'subtask_count': subtask_count,
                'is_goal_related': self._is_goal_related_method(method_name),
                'complexity_score': put_on_count + subtask_count,
            }

    def _is_goal_related_method(self, method_name):
        """Check if method contributes directly to achieving the goal."""
        return 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on the number of misplaced blocks
        and the decomposition method.
        """
        if not hasattr(node, 'state') or not hasattr(node, 'model'):
            return 0

        state = node.state
        model = node.model

        # Estimate the number of misplaced blocks based on the goal
        misplaced_blocks = self._estimate_misplaced_blocks(state, model)
        method_score = 0

        if hasattr(node, 'method') and node.method:
            method_score = self._score_method(node.method)

        return self.misplaced_blocks_penalty * misplaced_blocks + method_score

    def _score_method(self, method):
        """
        Score a method based on its potential to achieve goals.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Reward goal-related methods
            if info['is_goal_related']:
                score -= 0.5  # Small reward for goal-related methods

            # Penalize complex methods (more subtasks)
            score += info['subtask_count'] * 0.1

        return score

    def _estimate_misplaced_blocks(self, state, model):
        """
        Estimate the number of misplaced blocks based on the goal conditions.
        This is a simplified estimate assuming each 'on' goal requires a move.
        """
        goal_count = 0
        for goal_fact in model.goals:
           if 'on' in goal_fact.name.lower():
               goal_count += 1
        return goal_count

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic34(misplaced_blocks_penalty={self.misplaced_blocks_penalty}, action_cost={self.action_cost})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic34(misplaced_blocks_penalty={self.misplaced_blocks_penalty}, action_cost={self.action_cost})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 34 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced blocks penalty: {self.misplaced_blocks_penalty}\n"
            f"\tAction cost: {self.action_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
