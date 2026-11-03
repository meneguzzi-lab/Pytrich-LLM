import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Depot domain to be used as an example when prompting LLMs to generate heuristics
class DepotGemini20FlashHeuristic38(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_38=None, move_penalty=1.0, lift_penalty=0.5, name="DepotGemini20FlashHeuristic38"):
        super().__init__(name=name)
        self.move_penalty = move_penalty
        self.lift_penalty = lift_penalty
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
            move_count = 0
            lift_count = 0
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

                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'lift' in subtask_lower or 'drop' in subtask_lower:
                        lift_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'lift_count': lift_count,
                'subtask_count': subtask_count,
                'is_move_method': 'move' in method_name.lower(),
                'is_lift_method': 'lift' in method_name.lower() or 'drop' in method_name.lower(),
                'is_base_case': 'base' in method_name.lower() or 'done' in method_name.lower(),
                'complexity_score': move_count + lift_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Score a method based on various factors.
        Lower scores indicate more promising methods (cost-based heuristic).
        """
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)

        # Use precomputed info if available
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Penalize complexity
            score += info['move_count'] * self.move_penalty
            score += info['lift_count'] * self.lift_penalty

            # Reward goal achieving methods (adjust values as needed)
            if info['is_move_method']:
                score += 0.1  # Slightly encourage move methods
            if info['is_lift_method']:
                score += 0.2 # Encourage lift/drop operations
            if info['is_base_case']:
                score -= 0.5 # Strongly encourage base cases

        # Additional domain-specific heuristics can be added here

        return score

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic38(move_penalty={self.move_penalty}, lift_penalty={self.lift_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic38(move_penalty={self.move_penalty}, lift_penalty={self.lift_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 38 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove penalty: {self.move_penalty}\n"
            f"\tLift penalty: {self.lift_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
