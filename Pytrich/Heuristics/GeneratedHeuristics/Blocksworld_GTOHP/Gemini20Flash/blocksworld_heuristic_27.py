import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for Blocksworld
class BlocksworldGemini20FlashHeuristic27(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_27, name="BlocksworldGemini20FlashHeuristic27"):
        super().__init__(name=name)
        self.move_penalty = 1  # Penalty for each move operation
        self.unstack_penalty = 2 #Penalty for unstacking blocks.
        self.stack_penalty = 2 #Penalty for stacking blocks.
        self.ontable_penalty = 1 #Penalty for putting on the table.
        self.clear_penalty = 1 #Penalty to make a block clear
        self.goal_achieved_bonus = -3  # Bonus for achieving a goal condition
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        # Precompute heuristic information if needed (e.g., analyzing methods)
        self._precompute_method_info()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        (Optional) Precompute information about methods for faster scoring.
        For example, identify methods that involve moving blocks.
        """
        self.method_scores = {}
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)


        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            move_count = 0
            stack_count = 0
            unstack_count = 0
            ontable_count = 0
            clear_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
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
                        move_count+=1
                    if 'stack' in subtask_lower:
                        stack_count+=1
                    if 'unstack' in subtask_lower:
                        unstack_count +=1
                    if 'ontable' in subtask_lower:
                        ontable_count += 1
                    if 'clear' in subtask_lower:
                        clear_count += 1


            self.method_scores[method_name] = {
                'move_count': move_count,
                'stack_count': stack_count,
                'unstack_count': unstack_count,
                'ontable_count': ontable_count,
                'clear_count' : clear_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name)
            }


    def _is_goal_achieving_method(self, method_name):
        """Check if method directly contributes to achieving goal conditions."""
        return 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given node.
        This involves estimating the cost to achieve the goal state from the current state.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on various factors, including domain-specific insights.
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed information
        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score += info['move_count'] * self.move_penalty
            score += info['stack_count'] * self.stack_penalty
            score += info['unstack_count'] * self.unstack_penalty
            score += info['ontable_count'] * self.ontable_penalty
            score += info['clear_count'] * self.clear_penalty

            if info['is_goal_achieving']:
                score += self.goal_achieved_bonus

        # Additional scoring based on method name (for methods not precomputed)
        if 'move' in method_lower:
            score += self.move_penalty  # Small penalty for generic move operations
        if 'unstack' in method_lower:
            score += self.unstack_penalty
        if 'stack' in method_lower:
            score += self.stack_penalty
        if 'ontable' in method_lower:
            score += self.ontable_penalty
        if 'clear' in method_lower:
            score += self.clear_penalty

        return max(score, 0)


    def __call__(self, parent_node, node):
        """
        Calculate and return the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic27(move_penalty={self.move_penalty}, goal_bonus={self.goal_achieved_bonus})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic27(move_penalty={self.move_penalty}, goal_bonus={self.goal_achieved_bonus})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash  Heuristic 27 info:\n"
            f"\tName: {self.name}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tUnstack penalty: {self.unstack_penalty}\n"
            f"\tStack penalty: {self.stack_penalty}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tClear penalty: {self.clear_penalty}\n"
            f"\tGoal achieved bonus: {self.goal_achieved_bonus}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
