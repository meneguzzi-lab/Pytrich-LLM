import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic16(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic statically analyzes the domain's methods to guide the planner.
    It operates on the principle that the Towers of Hanoi problem has a well-defined
    recursive structure. The heuristic assigns a cost to each decomposition method
    based on its role in this structure:

    1.  **Base Cases (Cost 0):** Methods that terminate a recursive branch, like
        `exchangeClear`, are considered zero-cost as they complete a sub-goal.
    2.  **Primitive Actions (Cost 1):** Methods that directly decompose into a single
        primitive `move` action, like `newMethod21`, represent the fundamental unit
        of work and are given a small base cost.
    3.  **Recursive Steps (Cost 2):** Methods that perform a move and also make a
        recursive call (e.g., `exchangeLR`, `exchangeRL`, `m-rotateTower`)
        are the core of the algorithm. They are penalized more than a simple move
        but are preferred over pure control-flow methods.
    4.  **Control Flow (Cost 3):** High-level methods that primarily manage the
        recursive process without directly causing a move (e.g., `m-shiftTower`,
        `selectedDirection`) are given the highest penalty to encourage the planner
        to delve into the sub-problems that perform actual work.

    This scoring system encourages the planner to bottom-out the recursion and execute
    moves, providing an effective guide through the search space without needing to
    analyze the dynamic state, which keeps the heuristic computationally cheap.
    """
    def __init__(self, towers_gemini25pro_heuristic_16=None, action_cost=1, recursion_cost=2, control_flow_cost=3, name="towers_gemini25pro_heuristic_16"):
        super().__init__(name=name)
        # Heuristic cost parameters for different method types
        self.action_cost = action_cost
        self.recursion_cost = recursion_cost
        self.control_flow_cost = control_flow_cost

        # Dictionary to store precomputed scores for each method
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic by pre-computing scores for each method in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_scores()
        self.preprocessing_time = time.time() - start_time

        # Calculate the h-value for the initial node to initialize the heuristic base class
        initial_h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _precompute_method_scores(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a heuristic score
        based on their structural role in the Towers of Hanoi algorithm.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            # Default score for methods that don't fit other categories (control flow)
            score = self.control_flow_cost

            subtask_names = [t.name for t in method.task_network]

            # Base case: method with no subtasks (e.g., exchangeClear)
            if not method.task_network:
                score = 0
            # Action wrapper: method decomposing to the primitive 'move' action
            elif len(subtask_names) == 1 and 'move ' in subtask_names[0]:
                score = self.action_cost
            # Recursive step: method that includes a 'move_abstract' subtask
            elif 'move_abstract' in subtask_names:
                score = self.recursion_cost

            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given search node by looking up the
        pre-computed score for the node's associated method.
        """
        # If the node represents a primitive action, it has no 'method' attribute.
        # Its cost is inherent to the action itself.
        if not hasattr(node, 'method') or not node.method:
            if hasattr(node, 'task') and node.task and hasattr(node.task, 'cost'):
                return node.task.cost
            return 0

        method_name = node.method.name
        # Return the precomputed score, or the default cost if not found.
        return self.method_scores.get(method_name, self.control_flow_cost)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic16(action={self.action_cost}, rec={self.recursion_cost}, ctrl={self.control_flow_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 16 ({self.name}):\n"
            f"\tAction Cost: {self.action_cost}\n"
            f"\tRecursion Cost: {self.recursion_cost}\n"
            f"\tControl Flow Cost: {self.control_flow_cost}\n"
            f"\tMethods Scored: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
