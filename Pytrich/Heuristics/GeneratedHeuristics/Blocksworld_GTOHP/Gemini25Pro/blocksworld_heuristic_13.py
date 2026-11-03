import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain.
# This heuristic is designed to guide the search by scoring decomposition methods
# based on a static analysis of the domain model. It estimates the cost of
# applying a method, preferring those that make direct progress towards building
# the goal towers while penalizing preparatory or complex decompositions.
class BlocksworldGemini25ProHeuristic13(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Rationale:
    1.  Domain Insight: The primary challenge in Blocksworld is managing block-clearing
        dependencies. The cost of a plan is dominated by the number of blocks that
        need to be moved to clear the way for stacking operations.
    2.  Heuristic Strategy: This heuristic scores decomposition methods based on the
        complexity of their subtasks. It assigns a small, admissible cost to each method,
        reflecting the estimated number of primitive actions required.
    3.  Cost Calculation:
        - Penalizes methods that introduce primitive move actions (stack, unstack, etc.)
          or abstract 'clear' tasks.
        - Rewards methods that represent base cases (i.e., the goal of the task is
          already achieved) with a zero cost.
        - Gives a small bonus to methods that directly achieve a `put_on` goal, as they
          represent tangible progress.
    4.  Admissibility: All costs and penalties are kept very small (<< 1.0, the cost of
        a primitive action) to ensure the heuristic is admissible and does not
        overestimate the true cost to the goal. This guides the search without
        compromising optimality.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_13={"move_penalty": 0.2, "clear_penalty": 0.1}, name="blocksworld_gemini25pro_heuristic_13"):
        super().__init__(name=name)
        self.move_penalty = blocksworld_gemini25pro_heuristic_13.get("move_penalty", 0.2)
        self.clear_penalty = blocksworld_gemini25pro_heuristic_13.get("clear_penalty", 0.1)
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
        Statically analyzes all decomposition methods in the domain model to
        pre-calculate scores based on their subtasks. This avoids recalculating
        this information during the search.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            move_count = 0
            clear_count = 0
            
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            
            subtask_count = len(subtasks)

            for subtask in subtasks:
                subtask_name = subtask.name.lower()
                if any(kw in subtask_name for kw in ['unstack', 'stack', 'pickup', 'putdown']):
                    move_count += 1
                # Penalize methods that call for clearing a block
                if 'clear' in subtask_name:
                    clear_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'clear_count': clear_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method, move_count),
                'is_base_case': self._is_base_case_method(method_name, subtask_count),
                'is_clear_method': self._is_clear_method(method)
            }

    def _is_goal_achieving_method(self, method, move_count):
        """Checks if a method directly achieves a final `on` goal."""
        # A goal-achieving method is one that decomposes a `put_on` task
        # and results in a primitive move action like `stack`.
        if not hasattr(method, 'compound_task') or not hasattr(method.compound_task, 'name'):
            return False
        
        task_name = str(method.compound_task.name).lower()
        return ('put_on' in task_name or 'on' in task_name) and move_count > 0

    def _is_base_case_method(self, method_name, subtask_count):
        """Checks if a method is a base case for recursion (no-op)."""
        # A base case has no subtasks or a name that implies it.
        if subtask_count == 0:
            return True
        return any(kw in method_name.lower() for kw in ['base', 'done', 'noop', 'already'])

    def _is_clear_method(self, method):
        """Checks if a method is related to the `clear` sub-problem."""
        if not hasattr(method, 'compound_task') or not hasattr(method.compound_task, 'name'):
            return False
        return 'clear' in str(method.compound_task.name).lower()
        
    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a search node by scoring its
        associated decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method means no decomposition cost
        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on its precomputed characteristics.
        Lower scores are better.
        """
        score = 0.0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            
            if info['is_base_case']:
                return 0.0 # Base cases have zero cost.

            # Add cost for each move action required by the method.
            score += info['move_count'] * self.move_penalty
            # Add cost for each 'clear' subtask, as clearing may require moves.
            score += info['clear_count'] * self.clear_penalty

            # Give a small bonus (negative cost) for methods that directly
            # achieve a goal, encouraging the planner to make progress.
            if info['is_goal_achieving']:
                score -= 0.05
        else:
            # Fallback for any method not caught in precomputation.
            if hasattr(method, 'task_network'):
                score += len(method.task_network) * 0.01

        # Ensure the heuristic value is non-negative.
        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # Clamp the value to a small maximum to maintain admissibility and stability.
        h_value = min(h_value, 1.0)
        
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic13(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 13 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
