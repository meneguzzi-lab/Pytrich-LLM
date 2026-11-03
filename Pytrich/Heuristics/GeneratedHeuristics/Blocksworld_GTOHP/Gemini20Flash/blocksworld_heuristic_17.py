import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Blocksworld domain
class BlocksworldGemini20FlashHeuristic17(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_17 ="blocksworld_gemini20flash_heuristic_17", ontable_bonus=0.1, on_block_bonus=0.2, clear_block_bonus=0.3, incorrect_stack_penalty=0.5):
        super().__init__(name=blocksworld_gemini20flash_heuristic_17)
        self.ontable_bonus = ontable_bonus
        self.on_block_bonus = on_block_bonus
        self.clear_block_bonus = clear_block_bonus
        self.incorrect_stack_penalty = incorrect_stack_penalty
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
            # Get methods from abstract tasks
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks
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

                    if 'put_on' in subtask_lower or 'do_put_on' in subtask_lower:
                        put_on_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': put_on_count + subtask_count
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly contributes to achieving placement goals."""
        # Methods that handle the main ACHIEVE-AT task completion
        goal_indicators = ['put_on', 'do_put_on']
        return any(indicator in method_name.lower() for indicator in goal_indicators)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return self._calculate_goal_distance(node)

        return self._score_method(node.method, node) + self._calculate_goal_distance(node)

    def _calculate_goal_distance(self, node):
        """
        Calculates a heuristic estimate based on how far the current state is from achieving the goals.
        This checks the number of blocks that are not in their goal positions.
        """
        state = node.state
        goal_state = self.model.goals

        # Number of blocks not at the correct place based on goal state
        misplaced_blocks = 0
        for goal_fact_id in goal_state:
            goal_fact = self.model.get_component(goal_fact_id)
            if goal_fact_id not in state:
                misplaced_blocks += 1

        return misplaced_blocks * self.incorrect_stack_penalty

    def _score_method(self, method, node):
        """
        Score a method based on various factors.
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Small penalties for operation complexity
            score += info['put_on_count'] * 0.1
            score += info['subtask_count'] * 0.05

            # Small bonuses (negative cost) for productive methods
            if info['is_goal_achieving']:
                score -= 0.2  # Bonus for methods that achieve goals

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(ontable_bonus={self.ontable_bonus}, on_block_bonus={self.on_block_bonus}, clear_block_bonus={self.clear_block_bonus}, incorrect_stack_penalty={self.incorrect_stack_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(ontable_bonus={self.ontable_bonus}, on_block_bonus={self.on_block_bonus}, clear_block_bonus={self.clear_block_bonus}, incorrect_stack_penalty={self.incorrect_stack_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 17 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tOntable bonus: {self.ontable_bonus}\n"
            f"\tOn block bonus: {self.on_block_bonus}\n"
            f"\tClear block bonus: {self.clear_block_bonus}\n"
            f"\tIncorrect stack penalty: {self.incorrect_stack_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
