import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask

class DepotGemini25ProHeuristic0(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Depot domain.

    Design Rationale:
    This heuristic guides the planner by evaluating the cost of hierarchical method decompositions.
    The core idea is that the cost of a plan is primarily driven by the number of primitive actions executed.
    Therefore, methods that expand into more or costlier primitive actions should be penalized.

    The heuristic operates in two stages:
    1. Preprocessing (`_precompute_method_info`): Before the search begins, it analyzes every method
       in the domain. For each method, it calculates a score based on the primitive actions
       (subtasks) in its decomposition. The score is an admissible estimate of the minimum cost
       of applying that method, counting only the immediate actions and ignoring the cost of
       further decomposing abstract subtasks (which is equivalent to assuming their cost is zero).

    2. Evaluation (`_calculate_node_heuristic`): During the search, whenever the planner considers
       applying a method to a task, the heuristic provides the precomputed score for that method.
       Methods with lower scores (i.e., fewer immediate primitive actions) are preferred.

    Domain Insights:
    - Action Costs: All actions in the Depot domain have a uniform cost of 1 (except 'nop').
    - Key Actions: The most significant actions are `Drive` (for moving between locations) and the set
      of handling actions (`Lift`, `Drop`, `Load`, `Unload`) for manipulating crates.
    - Admissibility: By only summing the costs of primitive actions directly present in a method's
      decomposition, the heuristic remains admissible. The true cost of expanding a method is the
      sum of its primitive subtasks plus the cost of solving its abstract subtasks. Since the cost
      of solving abstract subtasks is always non-negative, our estimate is a lower bound (h <= h*).

    This approach provides a computationally cheap yet effective way to guide the search towards
    simpler and more direct solutions, such as preferring to move a crate within the same location
    over using a truck to transport it between locations if both options are available.
    """
    def __init__(self, drive_cost=1.0, handle_cost=1.0, depot_gemini25pro_heuristic_0=None, name="depot_gemini25pro_heuristic_0"):
        """
        Initializes the heuristic.

        Args:
            drive_cost (float): The heuristic cost associated with a 'Drive' action.
            handle_cost (float): The heuristic cost for handling actions like Lift, Drop, Load, Unload.
            depot_gemini25pro_heuristic_0: Parameter included to satisfy specific naming requirements.
            name (str): The name of the heuristic.
        """
        super().__init__(name=name)
        self.drive_cost = drive_cost
        self.handle_cost = handle_cost
        self.method_scores = {}
        self.preprocessing_time = 0
        self.model = None

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes method scores.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # The initial node has no method applied, so its heuristic value is 0.
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain and assigns them a heuristic score.
        The score is the sum of the costs of all primitive actions in the method's decomposition.
        This is an admissible estimate of the method's execution cost.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        # Create a set of primitive action names for efficient lookup.
        primitive_actions = {op.name for op in self.model.operators}
        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            score = 0.0

            if hasattr(method, 'task_network'):
                for subtask in method.task_network:
                    # We only consider primitive actions for the immediate cost.
                    # The cost of abstract tasks is assumed to be 0 for admissibility.
                    if subtask.name in primitive_actions:
                        subtask_name_lower = subtask.name.lower()
                        if "drive" in subtask_name_lower:
                            score += self.drive_cost
                        elif any(h in subtask_name_lower for h in ["lift", "drop", "load", "unload"]):
                            score += self.handle_cost
                        # 'nop' actions have a cost of 0 and are implicitly ignored.

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the method used to create this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        return self.method_scores.get(method_name, 0.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini25ProHeuristic0(drive_cost={self.drive_cost}, handle_cost={self.handle_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tDrive Cost Penalty: {self.drive_cost}\n"
            f"\tHandle Cost Penalty: {self.handle_cost}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
