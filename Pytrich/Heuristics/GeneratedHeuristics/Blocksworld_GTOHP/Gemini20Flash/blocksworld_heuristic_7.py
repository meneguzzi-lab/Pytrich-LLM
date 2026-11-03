import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Blocksworld domain
class BlocksworldGemini20FlashHeuristic7(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_7="blocksworld_gemini20flash_heuristic_7", goal_reward=1.0, ontable_penalty=0.5, on_penalty=0.3, clear_reward=0.2):
        super().__init__(name=blocksworld_gemini20flash_heuristic_7)
        self.goal_reward = goal_reward
        self.ontable_penalty = ontable_penalty
        self.on_penalty = on_penalty
        self.clear_reward = clear_reward

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
        elif hasattr(self.model, 'abstract_tasks'):
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count operations related to putting blocks
            put_on_count = 0
            grasp_count = 0
            unstack_count = 0
            stack_count = 0
            subtask_count = 0

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

                    if 'put_on' in subtask_lower or 'put-on' in subtask_lower:
                        put_on_count += 1
                    if 'grasp' in subtask_lower:
                        grasp_count += 1
                    if 'unstack' in subtask_lower:
                        unstack_count += 1
                    if 'stack' in subtask_lower:
                        stack_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'grasp_count': grasp_count,
                'unstack_count': unstack_count,
                'stack_count': stack_count,
                'subtask_count': subtask_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': put_on_count + grasp_count + unstack_count + stack_count,
                'is_base_case': self._is_base_case_method(method_name),
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if method achieves the goal.
        In blocksworld, this often involves 'put_on' or 'stack' operations."""
        return 'put_on' in method_name.lower() or 'stack' in method_name.lower()

    def _is_base_case_method(self, method_name):
        """Check if the method is a base case with no further decomposition."""
        # Heuristic: if the method name includes 'done' or 'finished', it's a base case
        return 'done' in method_name.lower() or 'finished' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its state and method.
        """
        if not hasattr(node, 'method') or not node.method:
            return self._estimate_goal_distance(node)

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on domain-specific factors.
        """
        score = 0  # Initialize score
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Utilize precomputed information if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Give a bonus (negative cost) for goal-achieving methods
            if info['is_goal_achieving']:
                score -= self.goal_reward  # Reward for 'put_on' or 'stack'

            # Penalize complex methods with many subtasks
            score += info['subtask_count'] * 0.1

            # Base cases are good, so give them a small reward
            if info['is_base_case']:
                score -= 0.1
        # Additional heuristics based on method names
        if 'grasp' in method_lower:
            score += 0.2  # Small penalty for grasping operations
        elif 'unstack' in method_lower:
            score += 0.3  # Slightly higher penalty for unstacking
        elif 'move' in method_lower:
            score += 0.1  # Small penalty for moving

        return score

    def _estimate_goal_distance(self, node):
        """
        Estimate distance to the goal based on the state.
        This simple heuristic just counts the number of blocks not in their goal position.
        """
        if not hasattr(self.model, 'goals'):
            return 0

        # Convert goals to a set for faster checking
        goals_set = set(self.model.goals)

        # Penalize facts in the initial state that are not in the goal state
        misplaced_blocks = 0
        for fact_id in self.model.initial_state:
          fact = self.model.get_component(fact_id)
          if fact_id not in goals_set:
            misplaced_blocks += 1 #self.ontable_penalty
            #if fact.name.startswith('on'):
            #  misplaced_blocks += self.on_penalty

        return misplaced_blocks * self.ontable_penalty


    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic7(goal_reward={self.goal_reward}, ontable_penalty={self.ontable_penalty}, on_penalty={self.on_penalty}, clear_reward={self.clear_reward})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic7(goal_reward={self.goal_reward}, ontable_penalty={self.ontable_penalty}, on_penalty={self.on_penalty}, clear_reward={self.clear_reward})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 7 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tGoal reward: {self.goal_reward}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tOn penalty: {self.on_penalty}\n"
            f"\tClear reward: {self.clear_reward}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
