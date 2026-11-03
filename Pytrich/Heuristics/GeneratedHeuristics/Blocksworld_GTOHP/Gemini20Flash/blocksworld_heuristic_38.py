import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for blocksworld domain
class BlocksworldGemini20FlashHeuristic38(Heuristic):
    def __init__(self, blocksworld_heuristic_38, misplaced_block_penalty=1, ontable_penalty=0.5, clear_penalty=0.2, name="BlocksworldGemini20FlashHeuristic38"):
        super().__init__(name=name)
        self.misplaced_block_penalty = misplaced_block_penalty
        self.ontable_penalty = ontable_penalty
        self.clear_penalty = clear_penalty
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

                    if 'do_put_on' in subtask_lower:
                        put_on_count += 1

            self.method_scores[method_name] = {
                'put_on_count': put_on_count,
                'subtask_count': subtask_count,
                'is_goal_method': self._is_goal_method(method_name),
                'complexity_score': put_on_count
            }

    def _is_goal_method(self, method_name):
        """Check if method directly contributes to achieving goal (placing blocks on each other)."""
        return 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method and state.
        """
        if not hasattr(node, 'method') or not node.method:
            return self._estimate_goal_distance(node)

        method_score = self._score_method(node.method, node)
        state_distance = self._estimate_goal_distance(node)
        return method_score + state_distance

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

            # Small penalties for complexity
            score += info['put_on_count'] * 0.1
            score += info['subtask_count'] * 0.05

            # Small bonuses (negative cost) for productive methods
            if info['is_goal_method']:
                score -= 0.2  # Bonus for methods that achieve goals

        return max(score, 0)

    def _estimate_goal_distance(self, node):
        """
        Estimate the distance to the goal state based on the current state.
        Considers misplaced blocks, blocks on the table, and clear blocks.
        """
        state = node.state
        goal = self.model.goals

        misplaced_blocks = 0
        ontable_blocks = 0
        clear_blocks = 0

        # Assuming goal is represented as a set of facts
        for fact in goal:
            fact_name = fact.name
            if 'on' in fact_name:
                # Count 'on' facts not present in the current state
                if fact not in state:
                    misplaced_blocks += 1

        # Estimate additional cost for blocks that need to be moved from the table
        for fact in self.model.facts:
            fact_name = fact.name
            if 'ontable' in fact_name and fact in state:
                ontable_blocks += 1

         # Estimate cost for blocks that are not clear when they should be
        for fact in self.model.facts:
            fact_name = fact.name
            if 'clear' in fact_name and fact not in state:
                clear_blocks += 1

        # Total estimated cost
        total_cost = (
            misplaced_blocks * self.misplaced_block_penalty +
            ontable_blocks * self.ontable_penalty +
            clear_blocks * self.clear_penalty
        )

        return total_cost

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic38(misplaced={self.misplaced_block_penalty}, ontable={self.ontable_penalty}, clear={self.clear_penalty})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic38(misplaced={self.misplaced_block_penalty}, ontable={self.ontable_penalty}, clear={self.clear_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 38 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced block penalty: {self.misplaced_block_penalty}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tClear penalty: {self.clear_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
