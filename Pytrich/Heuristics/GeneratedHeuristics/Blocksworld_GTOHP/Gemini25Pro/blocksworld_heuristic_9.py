import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic9(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Blocksworld domain.

    Rationale:
    The core challenge in Blocksworld is determining the best sequence of moves to build
    goal towers without unnecessarily dismantling correctly placed blocks or making
    many intermediate moves (e.g., moving blocks to the table).

    This heuristic guides the search by evaluating the quality of a decomposition method
    based on how directly it contributes to achieving the final goal configuration.

    Heuristic Strategy:
    1.  **Goal-Oriented Scoring:** The heuristic first parses the problem's goal state to
        understand the final tower structure. It rewards methods that directly place a
        block onto its final goal position.
    2.  **Penalty for Intermediate Moves:** Methods that result in placing a block on the
        table or on another block that is not its final destination are penalized. These
        are often necessary steps for clearing other blocks but do not make direct progress.
    3.  **Complexity Analysis:** It performs a one-time static analysis of all methods in
        the domain to count the number of primitive actions (stack, unstack, etc.) in their
        subtasks. This provides a baseline cost, penalizing more complex decompositions.
    4.  **Admissibility:** All heuristic values are kept as small fractions, significantly
        less than the cost of a single primitive action (cost=1). This ensures the
        heuristic remains admissible by providing a gentle nudge rather than an
        overestimated cost-to-go.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_9, stack_penalty=0.2, ontable_penalty=0.3, clear_penalty=0.1):
        super().__init__(name="blocksworld_gemini25pro_heuristic_9")
        self.stack_penalty = stack_penalty
        self.ontable_penalty = ontable_penalty
        self.clear_penalty = clear_penalty
        self.method_scores = {}
        self.goal_config = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by parsing the model's goal state and pre-analyzing
        all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._parse_goal_config()
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _parse_goal_config(self):
        """
        Parses the model's goal facts to build a dictionary of final block positions.
        Example: A goal `on(b1, b4)` results in `self.goal_config['b1'] = 'b4'`.
        """
        if not self.model or not self.model.goals:
            return
        for goal_fact in self.model.goals:
            if goal_fact.startswith('on('):
                try:
                    parts = goal_fact[3:-1].split(',')
                    if len(parts) == 2:
                        block_above = parts[0].strip()
                        block_below = parts[1].strip()
                        self.goal_config[block_above] = block_below
                except Exception:
                    pass
            elif goal_fact.startswith('ontable('):
                try:
                    block = goal_fact[8:-1].strip()
                    self.goal_config[block] = 'table'
                except Exception:
                    pass

    def _precompute_method_info(self):
        """
        Statically analyzes all methods to count the primitive operations within their
        decompositions. This is done once before the search begins for efficiency.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            unstack_count = 0
            stack_count = 0
            pickup_count = 0
            putdown_count = 0
            subtasks = method.task_network if hasattr(method, 'task_network') else []

            for subtask in subtasks:
                subtask_name = subtask.name.lower()
                if 'unstack' in subtask_name:
                    unstack_count += 1
                elif 'stack' in subtask_name:
                    stack_count += 1
                elif 'pickup' in subtask_name:
                    pickup_count += 1
                elif 'putdown' in subtask_name:
                    putdown_count += 1

            self.method_scores[method_name] = {
                'unstack_count': unstack_count,
                'stack_count': stack_count,
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'subtask_count': len(subtasks),
                'is_base_case': len(subtasks) == 0,
                'complexity_score': unstack_count + stack_count + pickup_count + putdown_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a search node based on its applied method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Scores a method based on its alignment with the goal configuration. Lower scores
        indicate more promising methods.
        """
        score = 0.0
        info = self.method_scores.get(method.name)
        if not info:
            return 0.1  # Default small cost for unknown methods

        if info['is_base_case']:
            return 0.0

        score += info['complexity_score'] * self.clear_penalty

        task_name = ""
        if hasattr(node, 'task') and node.task:
            task_name = node.task.name
        elif hasattr(method, 'compound_task') and method.compound_task:
            task_name = method.compound_task.name

        if 'do_put_on' in task_name:
            try:
                params = task_name.split('(')[1][:-1].split(',')
                block = params[0].strip()
                target = params[1].strip()

                if self.goal_config.get(block) == target:
                    score -= self.stack_penalty  # Bonus for goal-achieving move
                else:
                    if info['putdown_count'] > 0:
                        score += self.ontable_penalty  # Penalty for intermediate on-table move
                    elif info['stack_count'] > 0:
                        score += self.stack_penalty  # Penalty for stacking on non-goal block
            except IndexError:
                pass

        return max(0.0, score)

    def __call__(self, parent_node, node):
        """
        Main entry point for the heuristic calculation during search.
        """
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 0.9)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic9(stack={self.stack_penalty}, ontable={self.ontable_penalty}, clear={self.clear_penalty})"

    def __str__(self):
        return "BlocksworldGemini25ProHeuristic9"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 9 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tGoal Configuration Blocks: {len(self.goal_config)}\n"
            f"\tStack Penalty: {self.stack_penalty}\n"
            f"\tOn-Table Penalty: {self.ontable_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
