import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for Blocksworld-GTHOP domain
class BlocksworldGemini20FlashHeuristic20(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_20 = 'blocksworld_gemini20flash_heuristic_20', ontable_bonus=0.1, clear_bonus=0.1, name="blocksworld_gemini20flash_heuristic"):
        super().__init__(name=name)
        self.ontable_bonus = ontable_bonus
        self.clear_bonus = clear_bonus
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

                    if 'put_on' in subtask_lower:
                        puton_count += 1

            self.method_scores[method_name] = {
                'puton_count': puton_count,
                'subtask_count': subtask_count,
                'is_goal_method': self._is_goal_method(method_name),
                'complexity_score': puton_count
            }

    def _is_goal_method(self, method_name):
        """Check if method is directly related to achieving goals (e.g., putting blocks on correct positions)."""
        return 'do_put_on' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its state and decomposition method.
        """
        h_value = 0

        # Check if the node has a method and score it
        if hasattr(node, 'method') and node.method:
            h_value += self._score_method(node.method, node)

        # Heuristic based on the state: blocks on the table and blocks that are clear
        if hasattr(node, 'state'):
            state = node.state
            h_value -= self._calculate_state_bonuses(state)

        return h_value

    def _score_method(self, method, node):
        """
        Score a method based on precomputed info and method name.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        # Use precomputed information if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Add a small penalty for complexity (more subtasks)
            score += info['subtask_count'] * 0.01

            # Give a bonus (negative cost) for methods that directly achieve goals
            if info['is_goal_method']:
                score -= 0.1  # Slightly encourage goal-achieving methods

            # Add a cost per 'puton' action
            score += info['puton_count'] * 0.05

        return score

    def _calculate_state_bonuses(self, state):
        """
        Calculate bonuses based on the state of the world.
        """
        bonuses = 0
        #Facts are not directly accessible so this is a stub
        #For example, one could check if a fact like 'ontable' exists in the state and grant bonus if so

        return bonuses

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic(ontable_bonus={self.ontable_bonus}, clear_bonus={self.clear_bonus})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic(ontable_bonus={self.ontable_bonus}, clear_bonus={self.clear_bonus})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic 20 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tOntable bonus: {self.ontable_bonus}\n"
            f"\tClear bonus: {self.clear_bonus}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
