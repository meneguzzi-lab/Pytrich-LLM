import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator

class BlocksworldGemini25ProHeuristic10(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost of a decomposition method based on the number of
    primitive actions it contains. It follows a state-independent approach by pre-analyzing
    all methods in the domain during initialization.

    The core idea is that shorter decompositions are generally preferable, as they represent
    more direct ways to accomplish a task. The heuristic is designed to be admissible by
    providing a cost estimate that is a lower bound on the actual cost of executing the
    decomposition's sub-plan.

    Design Rationale:
    1.  **Analyze Domain Structure**: The primary challenge in Blocksworld is clearing blocks
        to make other blocks accessible. This often involves sequences of 'unstack' and
        'putdown' actions before a productive 'stack' action can occur. The heuristic
        quantifies this effort by counting the primitive actions in a method's expansion.
    2.  **Leverage Domain-Specific Knowledge**: It identifies 'base cases' (methods with no
        actions, typically for goals already achieved) and assigns them zero cost. It also
        recognizes methods related to the main 'do_put_on' task, slightly preferring them.
    3.  **Consider Hierarchical Structure**: By scoring decomposition methods, the heuristic directly
        evaluates choices within the task hierarchy, guiding the planner toward more efficient
        task refinements.
    4.  **Computational Efficiency**: All method analysis is done once during the `initialize`
        phase. The heuristic evaluation during search is a fast dictionary lookup and simple
        multiplication, ensuring minimal overhead.
    5.  **Admissibility**: By setting a small, uniform cost for each action, the heuristic's
        value for a method is proportional to the minimum length of the primitive plan that
        method generates. If the base action cost is set to be less than or equal to the
        true minimum action cost in the domain (e.g., 1), the heuristic remains admissible.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_10=None, action_cost=0.1, name="blocksworld_gemini25pro_heuristic_10"):
        """
        Initializes the heuristic.
        
        Args:
            blocksworld_gemini25pro_heuristic_10 (any, optional): An unused parameter included to satisfy a
                specific naming requirement in the generation prompt. Defaults to None.
            action_cost (float, optional): The base cost assigned to each primitive action.
                Kept small to ensure admissibility. Defaults to 0.1.
            name (str, optional): The name of the heuristic. Defaults to "blocksworld_gemini25pro_heuristic_10".
        """
        super().__init__(name=name)
        self.action_cost = action_cost
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for all decomposition methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain model to pre-calculate their heuristic scores.
        Scores are based on the number of primitive actions in the method's decomposition.
        """
        if not hasattr(self.model, 'decompositions') or not self.model.decompositions:
            return

        for method in self.model.decompositions:
            method_name = method.name
            action_count = 0

            # A method is a base case if its task network is empty.
            is_base_case = not bool(method.task_network)

            # Count the number of primitive actions (Operators) in the decomposition.
            for subtask in method.task_network:
                if isinstance(subtask, Operator):
                    action_count += 1
            
            # Check if this method decomposes a primary task like 'do_put_on'.
            is_primary_task_method = False
            if hasattr(method.compound_task, 'name') and 'put_on' in method.compound_task.name.lower():
                is_primary_task_method = True

            self.method_scores[method_name] = {
                'action_count': action_count,
                'is_base_case': is_base_case,
                'is_primary_task_method': is_primary_task_method
            }

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is based on the pre-computed score of the method applied in the node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        return self._score_method(node.method)

    def _score_method(self, method) -> float:
        """
        Scores a given method based on pre-computed information.
        Lower scores are better.
        """
        score = 0.0
        method_name = method.name

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            if info['is_base_case']:
                return 0.0  # Base cases have zero cost as the goal is already met.

            # The fundamental cost is the number of actions in the decomposition.
            score = info['action_count'] * self.action_cost

            # Apply a tiny bonus (cost reduction) for methods that directly address a primary task.
            if info['is_primary_task_method']:
                score -= 0.01
        else:
            # Fallback calculation if method wasn't pre-scored.
            action_count = sum(1 for sub in method.task_network if isinstance(sub, Operator))
            score = action_count * self.action_cost

        return max(score, 0.0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main entry point for the heuristic evaluation.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self) -> str:
        return f"BlocksworldGemini25ProHeuristic10(action_cost={self.action_cost})"

    def __str__(self) -> str:
        return self.__repr__()

    def __output__(self) -> str:
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 10 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tBase Action Cost: {self.action_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
