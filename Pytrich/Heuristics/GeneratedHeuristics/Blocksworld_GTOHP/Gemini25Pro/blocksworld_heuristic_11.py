import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic11(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic evaluates the promise of a decomposition method based on a static analysis
    of its name and the subtasks it generates. It assigns small costs (penalties) to methods
    based on the type of actions they are likely to perform, aiming to guide the search towards
    more direct solutions.

    The core idea is to penalize actions that are merely preparatory (like clearing a block)
    more than actions that directly achieve a goal (like stacking a block in its final position).
    Base cases, where a goal is already satisfied, are given zero cost.

    - `clear_penalty`: Cost associated with methods that clear a block. This is typically the most
      costly operation as it may involve moving multiple obstructing blocks.
    - `stack_penalty`: Cost for methods that perform a stacking action. This represents progress
      towards a goal.
    - `move_to_table_penalty`: A smaller, additional penalty for moving a block to the table, as this
      is an intermediate step and not a final goal configuration.
    """
    def __init__(self, clear_penalty=0.2, stack_penalty=0.1, move_to_table_penalty=0.05, name="blocksworld_gemini25pro_heuristic_11"):
        super().__init__(name=name)
        self.clear_penalty = clear_penalty
        self.stack_penalty = stack_penalty
        self.move_to_table_penalty = move_to_table_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-analyzing the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain model to pre-calculate scores based on their subtasks.
        This avoids repeated computation during the search.
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

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network

            unstack_count = 0
            stack_count = 0
            putdown_count = 0

            if subtasks:
                for subtask in subtasks:
                    subtask_name = str(subtask.task) if hasattr(subtask, 'task') else str(subtask)
                    subtask_lower = subtask_name.lower()

                    if 'unstack' in subtask_lower:
                        unstack_count += 1
                    if 'stack' in subtask_lower:
                        stack_count += 1
                    if 'put-down' in subtask_lower:
                        putdown_count += 1

            self.method_scores[method_name] = {
                'unstack_count': unstack_count,
                'stack_count': stack_count,
                'putdown_count': putdown_count,
                'is_base_case': self._is_base_case_method(method_name),
                'is_clearing': self._is_clearing_method(method_name)
            }

    def _is_base_case_method(self, method_name):
        """Checks if a method represents a terminal condition (base case)."""
        base_indicators = ['-base', '_base', 'done', 'already']
        return any(indicator in method_name.lower() for indicator in base_indicators)

    def _is_clearing_method(self, method_name):
        """Checks if a method's primary purpose is to clear a block."""
        return 'clear' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the method used to decompose the current task.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # Primitive actions have cost 1, so heuristic estimate is 0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Assigns a heuristic cost to a method.
        Lower scores indicate more promising methods.
        The costs are kept small to maintain admissibility.
        """
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            if info['is_base_case']:
                return 0 # Base cases have no cost.

            score = 0
            # Penalize methods that require clearing blocks.
            score += info['unstack_count'] * self.clear_penalty
            # Add cost for stacking actions, representing progress.
            score += info['stack_count'] * self.stack_penalty
            # Add a smaller cost for intermediate moves to the table.
            score += info['putdown_count'] * self.move_to_table_penalty
            
            # General penalty for clearing tasks as they are preparatory work.
            if info['is_clearing']:
                 score += self.clear_penalty
            return score
        
        # Fallback for methods not precomputed
        if self._is_base_case_method(method_name):
            return 0
        
        score = 0.1 # Default cost for an unknown decomposition
        if self._is_clearing_method(method_name):
            score += self.clear_penalty
        if 'put-down' in method_name.lower():
            score += self.move_to_table_penalty
        if 'stack' in method_name.lower():
            score += self.stack_penalty
        
        return score

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic calculation during search.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure the heuristic value is non-negative and reasonably bounded.
        h_value = min(max(h_value, 0), 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic11(clear={self.clear_penalty}, stack={self.stack_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 11 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tStack Penalty: {self.stack_penalty}\n"
            f"\tMove to Table Penalty: {self.move_to_table_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
