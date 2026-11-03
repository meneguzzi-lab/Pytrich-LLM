import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Heuristic for the Blocksworld domain.
# NOTE: The Blocksworld domain file was not provided. This heuristic is based on
# a common understanding of the Blocksworld problem and an inferred set of tasks
# and methods, such as 'do_put_on', 'make_clear', 'stack', 'unstack', etc.
# The heuristic scores decomposition methods to guide the search, preferring
# methods that directly achieve goals or are base cases over those that
# perform preparatory work (like clearing blocks).
class BlocksworldGemini25ProHeuristic12(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Blocksworld domain.

    This heuristic evaluates the quality of a decomposition method rather than
    a state. It guides the planner to prefer methods that are more likely
    to lead to a solution quickly. It does this by assigning costs to methods
    based on the actions they likely decompose into.

    - Lower cost for base-case methods that terminate a task decomposition.
    - Lower cost for methods that directly achieve a goal (e.g., stacking a block).
    - Higher cost for methods that perform preparatory actions (e.g., clearing a block),
      as they increase plan length.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_12={'move_penalty': 0.2, 'clear_penalty': 0.1}, name="blocksworld_gemini25pro_heuristic_12"):
        super().__init__(name=name)
        # Penalties for different types of operations. These are kept small to
        # act as a guide without overriding the primary search algorithm.
        self.move_penalty = blocksworld_gemini25pro_heuristic_12.get('move_penalty', 0.2)
        self.clear_penalty = blocksworld_gemini25pro_heuristic_12.get('clear_penalty', 0.1)

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-analyzing the methods in the domain model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the model to classify them and calculate a
        complexity score. This information is cached for efficient lookups
        during the search.
        """
        if not self.model or not hasattr(self.model, 'methods'):
            return

        for method in self.model.methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            method_lower = method_name.lower()

            # Analyze subtasks to determine method's purpose and complexity
            move_count = 0
            clear_count = 0
            subtask_count = 0
            
            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = getattr(subtask, 'name', str(subtask)).lower()
                    if self._is_move_subtask(subtask_name):
                        move_count += 1
                    if self._is_clear_subtask(subtask_name):
                        clear_count += 1
            
            is_base = self._is_base_case_method(method_lower, subtask_count)
            is_goal_achieving = self._is_goal_achieving_method(method_lower)

            self.method_scores[method_name] = {
                'move_count': move_count,
                'clear_count': clear_count,
                'subtask_count': subtask_count,
                'is_base_case': is_base,
                'is_goal_achieving': is_goal_achieving,
                'complexity_score': (move_count * self.move_penalty) + (clear_count * self.clear_penalty)
            }
    
    def _is_move_subtask(self, subtask_name):
        """Checks if a subtask involves moving a block."""
        move_ops = ['unstack', 'stack', 'pickup', 'putdown', 'move']
        return any(op in subtask_name for op in move_ops)

    def _is_clear_subtask(self, subtask_name):
        """Checks if a subtask involves clearing a block."""
        return 'clear' in subtask_name

    def _is_base_case_method(self, method_name, subtask_count):
        """
        Checks if a method is a base case (terminating condition).
        Typically, these have no subtasks or are explicitly named as such.
        """
        base_indicators = ['base', 'done', 'noop', 'already_']
        return subtask_count == 0 or any(indicator in method_name for indicator in base_indicators)

    def _is_goal_achieving_method(self, method_name):
        """
        Checks if a method directly achieves an 'on' goal.
        These are typically the most productive methods.
        """
        goal_indicators = ['put_on', 'stack']
        return any(indicator in method_name for indicator in goal_indicators)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by scoring
        the method that led to this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no cost

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Assigns a cost to a method. Lower scores are better.
        - Penalizes methods that move or clear blocks.
        - Rewards methods that are base cases or achieve goals directly.
        """
        method_name = method.name if hasattr(method, 'name') else str(method)
        score = 0.0

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['complexity_score']

            # Apply a small bonus (negative cost) for desirable methods
            if info['is_goal_achieving']:
                score -= 0.1  # Prefer methods that stack blocks
            
            if info['is_base_case']:
                score -= 0.2  # Strongly prefer methods that terminate decomposition
        else:
            # Fallback for methods not pre-analyzed (should not happen in normal flow)
            if self._is_move_subtask(method_name.lower()):
                score += self.move_penalty
            if self._is_clear_subtask(method_name.lower()):
                score += self.clear_penalty

        # Ensure the heuristic value is non-negative and small
        return max(0, score)

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation.
        """
        h_value = self._calculate_node_heuristic(node)
        # Keep heuristic values small to guide rather than dominate the search
        h_value = min(h_value, 1.0) 
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic12(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __str__(self):
        return f"BlocksworldGemini25ProHeuristic12(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
