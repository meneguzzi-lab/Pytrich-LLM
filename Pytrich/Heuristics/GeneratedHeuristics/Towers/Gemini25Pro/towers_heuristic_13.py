import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic13(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    Rationale and Domain Insights:
    1.  **Core Problem:** The 'towers' domain models the classic Towers of Hanoi problem.
        The fundamental challenge is the recursive nature of moving a stack of rings,
        where the cost (number of moves) grows exponentially with the number of rings.
        The total number of primitive 'move' actions determines the plan length and, therefore, the solution quality.

    2.  **Heuristic Strategy:** This heuristic operates by analyzing the decomposition methods rather than the state.
        This is a constraint of the Pytrich planner's heuristic interface, which does not provide direct state access
        within the heuristic evaluation call. The heuristic estimates the cost-to-go by evaluating the productivity
        of a chosen decomposition method.

    3.  **Cost Metric:** The primary metric for progress in this domain is the execution of the primitive 'move' action.
        Therefore, the heuristic value for a given search node is defined as the number of primitive 'move' actions
        (or their abstract precursor, 'move_abstract') present in the sub-task network of the method applied to reach that node.
        This provides a direct, admissible lower-bound estimate of the work done at that decomposition step.

    4.  **Admissibility:** By setting the heuristic value `h(n)` to the number of primitive actions introduced by a method,
        we ensure admissibility. The value `h(n)` is a count of actions that must be performed as part of the sub-plan,
        so it can never overestimate the true cost of completing that sub-plan. The cost of each action is 1, and our
        heuristic contributes exactly 1 for each action found in the method's decomposition.

    5.  **Implementation:**
        -   **Preprocessing (`initialize`):** The heuristic pre-processes all decomposition methods from the domain model.
            It iterates through each method's subtasks and counts how many are 'move' or 'move_abstract'. This count is stored.
        -   **Evaluation (`__call__`):** During the search, when a node is evaluated, the heuristic looks up the pre-computed
            count of 'move' actions for the method that generated the node. This value is returned as the heuristic cost.
        -   **Efficiency:** This approach is highly efficient as the costly analysis is done once at the beginning. The per-node
            evaluation is a simple dictionary lookup.

    This design focuses the search towards methods that directly contribute to the plan by executing moves, while correctly assigning
    a cost of zero to purely structural or recursive methods that do not perform moves themselves.
    """
    def __init__(self, towers_gemini25pro_heuristic_13=None, move_penalty=1.0):
        """
        Initializes the TowersGemini25ProHeuristic13.

        :param towers_gemini25pro_heuristic_13: A configuration dictionary (optional).
        :param move_penalty: The cost associated with a single 'move' action.
        """
        super().__init__(name="towers_gemini25pro_heuristic_13")

        if isinstance(towers_gemini25pro_heuristic_13, dict):
            self.move_penalty = towers_gemini25pro_heuristic_13.get('move_penalty', move_penalty)
        else:
            self.move_penalty = move_penalty

        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing information about domain methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain to calculate their inherent cost,
        defined as the number of primitive move actions in their sub-task network.
        """
        if not self.model:
            return

        methods = getattr(self.model, 'decompositions', [])
        if not methods:
            methods = getattr(self.model, 'methods', [])

        move_action_names = {'move', 'move_abstract'}

        for method in methods:
            method_name = getattr(method, 'name', str(method))
            move_count = 0
            subtasks = getattr(method, 'task_network', [])

            if subtasks:
                for subtask in subtasks:
                    subtask_name = getattr(subtask, 'name', str(subtask))
                    if subtask_name in move_action_names:
                        move_count += 1
            
            self.method_scores[method_name] = {
                'move_count': move_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the pre-computed cost of the method used to reach this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = getattr(node.method, 'name', str(node.method))
        
        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            score = info['move_count'] * self.move_penalty
            return score
        
        return 0

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic13(move_penalty={self.move_penalty})"

    def __str__(self):
        return f"TowersGemini25ProHeuristic13(move_penalty={self.move_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro Heuristic 13 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
