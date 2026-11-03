import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for Blocksworld
class BlocksworldGemini20FlashHeuristic37(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_37 =1, ontable_penalty=1, on_penalty=2, clear_penalty=1, handempty_bonus=-1, name="BlocksworldGemini20FlashHeuristic37"):
        super().__init__(name=name)
        self.ontable_penalty = ontable_penalty
        self.on_penalty = on_penalty
        self.clear_penalty = clear_penalty
        self.handempty_bonus = handempty_bonus
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

            # Count different types of subtasks or actions
            puton_count = 0  # Count 'puton' actions/subtasks
            pickup_count = 0  # Count 'pickup' actions/subtasks
            stack_count = 0
            unstack_count = 0

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

                    if 'put_on' in subtask_lower:
                        puton_count += 1
                    if 'pick_up' in subtask_lower:
                        pickup_count += 1
                    if 'stack' in subtask_lower:
                        stack_count += 1
                    if 'unstack' in subtask_lower:
                        unstack_count += 1

            self.method_scores[method_name] = {
                'puton_count': puton_count,
                'pickup_count': pickup_count,
                'stack_count':stack_count,
                'unstack_count': unstack_count,
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': puton_count + pickup_count
            }

    def _is_goal_achieving_method(self, method_name):
        """Check if a method directly contributes to achieving the goals."""
        return 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on the current state and goals.
        """
        if not hasattr(node, 'state') or not node.state:
            return 0

        state = node.state
        goal_count = 0
        # Count how many goals are not yet achieved in the current state
        for goal in self.model.goals:
            if not (state & 1 << goal):
                goal_count += 1

        # Adjust heuristic based on the number of unachieved goals
        h_value = goal_count * self.on_penalty


        if hasattr(node, 'method') and node.method:
            h_value += self._score_method(node.method, node)

        return h_value


    def _score_method(self, method, node):
        """
        Score a method based on its characteristics.
        Lower scores indicate more promising methods.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Give a bonus to goal achieving methods
            if info['is_goal_achieving']:
                score -= 1

            # Penalize complex methods based on the number of subtasks
            score += (info['puton_count'] + info['pickup_count']) * 0.5

        return score


    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic37(ontable={self.ontable_penalty}, on={self.on_penalty}, clear={self.clear_penalty}, handempty={self.handempty_bonus})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic37(ontable={self.ontable_penalty}, on={self.on_penalty}, clear={self.clear_penalty}, handempty={self.handempty_bonus})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 37 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tOntable penalty: {self.ontable_penalty}\n"
            f"\tOn penalty: {self.on_penalty}\n"
            f"\tClear penalty: {self.clear_penalty}\n"
            f"\tHandempty bonus: {self.handempty_bonus}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
