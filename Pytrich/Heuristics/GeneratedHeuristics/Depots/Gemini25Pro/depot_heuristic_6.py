import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask

class DepotGemini25ProHeuristic6(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Depot domain.

    This heuristic estimates the cost of a decomposition method by analyzing its subtasks.
    The core idea is that the main challenges in the Depot domain are transportation between
    locations (driving) and clearing stacks of crates. The heuristic assigns higher costs
    to methods that involve these complex operations, guiding the planner to prefer simpler,
    local actions first.

    The heuristic pre-computes a score for each method in the domain during initialization.
    The score is a weighted sum of the costs of its subtasks. This approach ensures that the
    heuristic calculation during search is a fast dictionary lookup.

    Design Rationale:
    1.  **Identify Key Cost Drivers**: Driving trucks between locations and clearing crates 
        off surfaces (unstacking) are the most time-consuming sequences of actions. The
        heuristic assigns the highest penalties to methods involving these tasks (`drive_penalty`,
        `clear_penalty`).
    2.  **Hierarchical Cost Estimation**: Instead of only counting primitive actions, the heuristic
        assigns costs to abstract tasks as well (e.g., `do_clear`). This leverages the domain's
        hierarchical structure to make better estimates.
    3.  **Prioritize Simplicity**: Methods with fewer or cheaper subtasks receive lower scores,
        promoting simpler plans. Base cases (e.g., a task that is already achieved) have a score
        of zero, making them the most preferred choice when applicable.
    4.  **Admissibility**: The heuristic values are small, positive costs associated with applying
        a method. They act as an estimate of the immediate cost of a decomposition, helping to
        guide the search. While not a classic state-based cost-to-goal estimate like in A* on a state
        graph, in the context of HTN search, it guides the choice of which decomposition to explore first.
    5.  **Efficiency**: All scores are pre-computed in the `initialize` phase. The runtime cost
        of the heuristic is a single dictionary lookup, making it very efficient.
    """
    def __init__(self, depot_heuristic_6):
        config = depot_heuristic_6 if isinstance(depot_heuristic_6, dict) else {}
        self.drive_penalty = config.get('drive_penalty', 5)
        self.clear_penalty = config.get('clear_penalty', 10) 
        self.load_unload_penalty = config.get('load_unload_penalty', 2)
        self.lift_drop_penalty = config.get('lift_drop_penalty', 1)

        super().__init__(name="depot_gemini25pro_heuristic_6")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic by pre-computing scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyze all decomposition methods in the domain model and assign a heuristic score to each.
        The score reflects the estimated cost of the method's subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = 0
            method_name = method.name

            # Base cases (nop methods) have zero cost
            is_nop_method = False
            if hasattr(method, 'task_network') and len(method.task_network) == 1:
                subtask = method.task_network[0]
                if isinstance(subtask, Operator) and 'nop' in subtask.name.lower():
                    is_nop_method = True
            
            if is_nop_method:
                self.method_scores[method_name] = 0
                continue

            # Score method based on its subtasks
            if hasattr(method, 'task_network'):
                for subtask in method.task_network:
                    name = subtask.name.lower()
                    if 'clear' in name:
                        score += self.clear_penalty
                    elif 'drive' in name or 'get_truck' in name:
                        score += self.drive_penalty
                    elif 'load' in name or 'unload' in name:
                        score += self.load_unload_penalty
                    elif 'lift' in name or 'drop' in name:
                        score += self.lift_drop_penalty
                    else:
                        # Add a small default cost for any other unweighted tasks
                        # to prefer methods with fewer subtasks.
                        score += 0.5 
            
            # Add a special penalty for the most complex inter-place move method
            # to ensure it's only chosen when necessary.
            if 'm3_do_put_on' in method_name:
                score += self.drive_penalty

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given search node.
        The value is retrieved from the pre-computed scores based on the method used to reach this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0
        
        method_name = node.method.name
        return self.method_scores.get(method_name, 1) # Default cost of 1 if method not found

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini25ProHeuristic6(drive={self.drive_penalty}, clear={self.clear_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tDrive Penalty: {self.drive_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tLoad/Unload Penalty: {self.load_unload_penalty}\n"
            f"\tLift/Drop Penalty: {self.lift_drop_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
