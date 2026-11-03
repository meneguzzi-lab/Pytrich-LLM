import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini25ProHeuristic8(Heuristic):
    """
    A domain-dependent heuristic for the Depot domain.

    This heuristic estimates the cost of a decomposition method by analyzing its subtasks.
    It assigns a higher penalty to methods that involve transportation (driving a truck)
    and a lower penalty for local manipulation (lifting, dropping, loading crates).
    The core idea is to guide the search towards solutions that minimize costly truck
    movements between different places, prioritizing tasks that can be solved locally.

    Rationale:
    1.  **Domain Analysis**: The most expensive operation in the Depot domain is the `Drive`
        action, which moves trucks between locations. Other actions like `Lift`, `Drop`,
        `Load`, and `Unload` are comparatively cheaper as they occur within a single location.
    2.  **Heuristic Strategy**: The heuristic calculates a cost for each decomposition method
        based on the number and type of expensive subtasks it contains. By penalizing
        `Drive` actions more heavily, it encourages the planner to prefer methods that
        accomplish goals without moving trucks, or to batch movements effectively.
    3.  **Admissibility**: The heuristic is admissible because it counts a subset of the actions
        required to execute a plan derived from a method. The calculated score is a lower
        bound on the actual cost of the sub-plan, assuming each action has a cost of at
        least 1 and driving costs more.
    4.  **Efficiency**: Method scores are pre-computed during the `initialize` phase. This
        makes the heuristic evaluation during search very fast, as it's just a dictionary lookup.
    """
    def __init__(self, depot_heuristic_8, drive_penalty=5, manipulation_penalty=1, name="depot_gemini25pro_heuristic_8"):
        super().__init__(name=name)
        # The user requested this specific parameter naming and internal renaming
        self.depot_gemini25pro_heuristic_8 = depot_heuristic_8
        self.drive_penalty = drive_penalty
        self.manipulation_penalty = manipulation_penalty
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
        Precompute information about methods for faster scoring during search.
        This heuristic assigns a cost to each method based on the expensive actions
        (like Drive) in its decomposition.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            drive_count = 0
            manipulation_count = 0
            is_base_case = False

            subtasks = method.task_network if hasattr(method, 'task_network') else []

            if not subtasks and not hasattr(method, 'task_network'):
                is_base_case = True

            for subtask in subtasks:
                subtask_name = subtask.name.lower()

                # Penalize driving and tasks that lead to driving
                if 'drive' in subtask_name or 'do_get_truck' in subtask_name:
                    drive_count += 1
                
                # Penalize local manipulation actions
                if any(op in subtask_name for op in ['lift', 'drop', 'load', 'unload', 'do_lift', 'do_load', 'do_unload']):
                    manipulation_count += 1

                # Identify base cases (no-op)
                if 'nop' in subtask_name:
                    is_base_case = True
                    drive_count = 0
                    manipulation_count = 0
                    break

            self.method_scores[method_name] = {
                'drive_count': drive_count,
                'manipulation_count': manipulation_count,
                'is_base_case': is_base_case
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Score a method based on its pre-computed characteristics.
        Lower scores indicate more promising (cheaper) methods.
        """
        method_name = method.name
        
        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            
            if info['is_base_case']:
                return 0
            
            # The cost is a weighted sum of expensive operations.
            score = (info['drive_count'] * self.drive_penalty +
                     info['manipulation_count'] * self.manipulation_penalty)
            return score
        
        # Default score for unknown methods, assuming at least one manipulation
        return self.manipulation_penalty 

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini25ProHeuristic8(drive_penalty={self.drive_penalty}, manipulation_penalty={self.manipulation_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tDrive penalty: {self.drive_penalty}\n"
            f"\tManipulation penalty: {self.manipulation_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
