import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain
class BlocksworldGemini20FlashHeuristic32(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_32=1, misplaced_block_penalty=1, name="blocksworld_gemini20flash_heuristic_32"):
        super().__init__(name=name)
        self.misplaced_block_penalty = misplaced_block_penalty
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
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks.  We are looking for 'do_put_on'.
            puton_count = 0
            subtask_count = 0

            # Get subtasks from method
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
                        puton_count += 1

            self.method_scores[method_name] = {
                'puton_count': puton_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': puton_count
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly contributes to achieving goals."""
        return 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method
        and the number of misplaced blocks in the current state.
        """
        h_value = 0

        # Penalize misplaced blocks based on the initial state and goal
        if hasattr(self.model, 'initial_state') and hasattr(self.model, 'goals'):
            h_value += self._calculate_misplaced_blocks_heuristic(node) * self.misplaced_block_penalty

        # Add method-specific heuristic if the node has a method
        if hasattr(node, 'method') and node.method:
            h_value += self._score_method(node.method, node)

        return h_value

    def _calculate_misplaced_blocks_heuristic(self, node):
        """
        Calculates a heuristic based on the number of misplaced blocks
        compared to the goal state.  Since we don't have access to the
        current state explicitly, this is estimated from the initial state
        and goal conditions.
        """
        misplaced_blocks = 0
        if not hasattr(self.model, 'facts'):
            return 0

        facts = list(self.model.facts)

        # Convert sets to lists for indexing
        initial_state = list(self.model.initial_state)
        goals = list(self.model.goals)

        # Analyze goal conditions related to block positions (e.g., 'on', 'ontable')
        goal_block_positions = []
        for goal_fact_id in goals:
            goal_fact = facts[goal_fact_id]
            if 'on' in goal_fact.name or 'ontable' in goal_fact.name:
                goal_block_positions.append(goal_fact.name)

        # Count how many of these goal positions are NOT in the initial state
        for goal_position in goal_block_positions:
            found_in_initial_state = False
            for initial_fact_id in initial_state:
               initial_fact = facts[initial_fact_id]
               if goal_position == initial_fact.name:
                   found_in_initial_state = True
                   break

            if not found_in_initial_state:
                misplaced_blocks += 1

        return misplaced_blocks

    def _score_method(self, method, node):
        """
        Score a method based on various factors.  Lower scores indicate
        more promising methods (cost-based heuristic).
        """
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Small penalty for each 'puton' operation
            score += info['puton_count'] * 0.5

            # Bonus for goal-achieving methods
            if info['is_goal_achieving']:
                score -= 0.2

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic32(misplaced_penalty={self.misplaced_block_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic32(misplaced_penalty={self.misplaced_block_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 32 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
