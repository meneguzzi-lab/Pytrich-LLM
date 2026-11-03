import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini25ProHeuristic7(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Depot domain.

    This heuristic analyzes the inherent complexity of each decomposition method in the domain
    and assigns a static score based on the types of operations involved. The core idea is to guide
    the planner towards simpler or more direct methods first, improving search efficiency.

    The heuristic assigns penalties based on the following insights:
    1.  **Clearing Stacks (`do_clear`)**: This is often the most complex subproblem, potentially leading
        to recursive calls to move multiple crates. It receives the highest penalty.
    2.  **Transportation (`Drive`, `do_get_truck`)**: Moving trucks between locations is a primary cost
        driver, especially for goals involving different depots/distributors. It receives a high penalty.
    3.  **Loading/Unloading**: These actions are necessary for transportation and are more complex than
        simple lifting and dropping. They receive a moderate penalty.
    4.  **Lifting/Dropping**: These are the most fundamental manipulation actions and receive the lowest penalty.

    Base-case methods, which terminate a decomposition branch (e.g., the goal is already satisfied),
    are assigned a score of zero to be strongly preferred.

    The heuristic values are kept small and are not a true cost-to-go estimate, but rather a preference
    score to break ties and guide the search admissibly when combined with a path cost.
    """
    def __init__(self, depot_heuristic_7):
        super().__init__(name="depot_gemini25pro_heuristic_7")
        self.depot_gemini25pro_heuristic_7 = depot_heuristic_7
        self.method_scores = {}
        self.preprocessing_time = 0

        # Penalties are weighted based on the operational complexity in the Depot domain.
        self.clear_penalty = 0.5        # Highest penalty for complex, potentially recursive clearing tasks.
        self.transport_penalty = 0.4    # High penalty for moving trucks between locations.
        self.load_unload_penalty = 0.2  # Moderate penalty for loading/unloading crates from trucks.
        self.lift_drop_penalty = 0.1    # Lowest penalty for basic hoist manipulations.

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # The initial node has no associated method, so its heuristic value is 0.
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes every method in the domain and assigns a complexity score based on its subtasks.
        This is a one-time process during initialization.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = 0
            if not hasattr(method, 'task_network'):
                self.method_scores[method.name] = score
                continue

            subtask_names = [st.name for st in method.task_network]
            all_subtasks_str = ' '.join(subtask_names)

            # Penalize methods involving transportation.
            if 'Drive' in all_subtasks_str or any(t in all_subtasks_str for t in ['do_get_truck', 'do_load_truck', 'do_unload_truck', 'm3_do_put_on']):
                score += self.transport_penalty

            # Penalize methods that require clearing a surface.
            if 'do_clear' in all_subtasks_str:
                score += self.clear_penalty

            # Penalize methods involving loading or unloading operations.
            if 'Load' in all_subtasks_str or 'Unload' in all_subtasks_str or any(t in all_subtasks_str for t in ['do_load_truck', 'do_unload_truck', 'do_lift_crate']):
                score += self.load_unload_penalty

            # Penalize methods with basic lift/drop manipulations.
            if 'Lift' in all_subtasks_str or 'Drop' in all_subtasks_str or 'do_lift_crate' in all_subtasks_str:
                score += self.lift_drop_penalty

            # Base cases (e.g., using 'nop' or achieving a goal that's already true) have zero cost.
            if method.name in ['m0_do_put_on', 'm4_do_clear', 'm6_do_get_truck'] or 'nop' in all_subtasks_str:
                score = 0

            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Retrieves the pre-calculated heuristic value for the method applied in the given node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name
        # Return the pre-computed score, with a default for safety.
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for the current node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini25ProHeuristic7(transport={self.transport_penalty}, clear={self.clear_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tTransport Penalty: {self.transport_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tLoad/Unload Penalty: {self.load_unload_penalty}\n"
            f"\tLift/Drop Penalty: {self.lift_drop_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
