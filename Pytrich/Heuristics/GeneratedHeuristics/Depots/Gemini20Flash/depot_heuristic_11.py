import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Depot domain
class DepotGemini20FlashHeuristic11(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_11=1, name="DepotGemini20FlashHeuristic11"):
        super().__init__(name=name)
        self.name = name
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
            move_count = 0
            load_count = 0
            unload_count = 0
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

                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'load_count': load_count,
                'unload_count': unload_count,
                'subtask_count': subtask_count,
                'is_move_method': self._is_move_method(method_name),
                'is_load_method': self._is_load_method(method_name),
                'is_unload_method': self._is_unload_method(method_name),
                'is_base_case': self._is_base_case_method(method_name),
                'is_goal_achieving': self._is_goal_achieving_method(method_name),
                'complexity_score': move_count + load_count + unload_count
            }

    def _is_move_method(self, method_name):
        """Check if method is for moving.
        For depot, this often means moving a truck or hoist.
        """
        return 'move' in method_name.lower()

    def _is_load_method(self, method_name):
        """Check if method is for loading.
        For depot, this often means loading a crate onto a truck.
        """
        return 'load' in method_name.lower()

    def _is_unload_method(self, method_name):
        """Check if method is for unloading.
        For depot, this often means unloading a crate from a truck.
        """
        return 'unload' in method_name.lower()

    def _is_base_case_method(self, method_name):
        """Check if method is a base case (no further decomposition needed)."""
        # In Depot, base cases are often methods that directly call primitive actions
        # or have no subtasks.
        return 'm0' in method_name.lower() or 'm1' in method_name.lower()

    def _is_goal_achieving_method(self, method_name):
        """Check if method directly achieves goals (e.g., putting a crate at a destination)."""
        # For depot, these are often methods related to achieving the 'at' predicate for crates.
        return 'achieve' in method_name.lower()

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        Return very small values to ensure admissibility.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

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

            # Very small penalties for complexity
            score += info['move_count'] * 0.1
            score += info['load_count'] * 0.1
            score += info['unload_count'] * 0.1
            score += info['subtask_count'] * 0.05

            # Small bonuses (negative cost) for productive methods
            if info['is_goal_achieving']:
                score -= 0.2  # Small bonus for achieving goals

            if info['is_base_case']:
                score -= 0.1  # Small bonus for termination

        # Penalize move operations
        if 'move' in method_lower:
            score += 0.3

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        Keep values very small for admissibility.
        """
        h_value = self._calculate_node_heuristic(node)
        # Ensure very small values
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic11()"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic11()"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 11 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
