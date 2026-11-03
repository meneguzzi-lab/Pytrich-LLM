import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain.
class BlocksworldGemini25ProHeuristic3(Heuristic):
    def __init__(self, blocksworld_gemini25pro_heuristic_3=None, primitive_penalty=0.1, abstract_penalty=0.2, name="blocksworld_gemini25pro_heuristic_3"):
        super().__init__(name=name)
        self.primitive_penalty = primitive_penalty
        self.abstract_penalty = abstract_penalty
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

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes information about each method to score them during the search.
        The score is based on the complexity of the method's subtasks.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            primitive_count = 0
            abstract_count = 0

            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network

            # A method with no subtasks is a base case (e.g., goal already satisfied).
            is_base_case = not subtasks

            for subtask in subtasks:
                subtask_name = ""
                if hasattr(subtask, 'name'):
                    subtask_name = subtask.name
                elif hasattr(subtask, 'task'):
                    subtask_name = str(subtask.task)
                else:
                    subtask_name = str(subtask)
                
                if self._is_primitive_action(subtask_name):
                    primitive_count += 1
                # Check for recursive or auxiliary abstract tasks
                elif self._is_abstract_task(subtask_name):
                    abstract_count += 1

            self.method_scores[method_name] = {
                'primitive_count': primitive_count,
                'abstract_count': abstract_count,
                'is_base_case': is_base_case,
                'complexity_score': (primitive_count * self.primitive_penalty) + (abstract_count * self.abstract_penalty)
            }

    def _is_primitive_action(self, action_name):
        """Check if an action is a primitive (manipulates the world directly)."""
        primitive_actions = ['pickup', 'putdown', 'stack', 'unstack']
        return any(pa in action_name.lower() for pa in primitive_actions)

    def _is_abstract_task(self, task_name):
        """Check if a task is an abstract task that requires further decomposition."""
        # `do_put_on` can be recursive, and `do_clear` is a common complex subproblem.
        abstract_tasks = ['do_put_on', 'do_clear']
        return any(at in task_name.lower() for at in abstract_tasks)

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a search node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no estimated cost

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on its precomputed complexity.
        Lower scores are better, indicating a less complex decomposition.
        """
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Base cases (e.g., the goal is already achieved) have zero cost.
            if info['is_base_case']:
                return 0
            
            # The score is a weighted sum of primitive and abstract subtasks.
            # This prefers methods that solve the problem with fewer, simpler steps.
            score = info['complexity_score']
            return score

        # Default score for unknown methods
        return self.primitive_penalty * 2

    def __call__(self, parent_node, node):
        """
        Calculate the heuristic value for the given node.
        The heuristic estimates the cost of applying a method, guiding the search
        towards simpler decompositions.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure the heuristic value remains small and non-negative for admissibility.
        h_value = max(0, min(h_value, 1.0))
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic3(primitive_penalty={self.primitive_penalty}, abstract_penalty={self.abstract_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 3 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPrimitive Action Penalty: {self.primitive_penalty}\n"
            f"\tAbstract Task Penalty: {self.abstract_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
