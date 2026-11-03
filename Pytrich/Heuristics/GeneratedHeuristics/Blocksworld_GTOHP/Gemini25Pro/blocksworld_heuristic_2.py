import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# This heuristic was generated based on an analysis of the Blocksworld domain structure.
# Since the domain file was not provided, standard actions and tasks for Blocksworld are assumed.
class BlocksworldGemini25ProHeuristic2(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Rationale:
    The primary challenge in Blocksworld is managing the state of blocks (clear, on another block, on the table)
    to achieve a target tower configuration. The cost of a plan is the number of actions (unstack, stack, pickup, putdown).
    This heuristic estimates the cost of choosing a particular decomposition method by analyzing the primitive actions
    it expands into.

    Heuristic Strategy:
    1.  **Preprocessing**: During initialization, the heuristic analyzes all methods in the domain model.
        It counts the number of 'clearing' actions (unstack, putdown) and 'placing' actions (stack, pickup)
        in each method's subtasks.
    2.  **Cost Assignment**: It assigns a higher penalty to clearing actions, as they represent preparatory work or
        undoing a previous state. Placing actions, which directly contribute to building the goal tower, receive
        a smaller penalty.
    3.  **Base Case Recognition**: Methods that represent an already achieved state (base cases, e.g., a method
        for `do_put_on(A, B)` when `(on A B)` is already true) are identified and assigned a cost of zero.
    4.  **Admissibility**: The penalties are kept as small, positive values. The heuristic value for a method is a sum
        of these small penalties, ensuring it remains an admissible underestimate of the actual plan cost (which is a count
        of actions, i.e., cost=1 per action).

    This approach guides the planner to prefer methods that achieve goals more directly with less intermediate block shuffling.
    """
    def __init__(self, clear_penalty=0.2, move_penalty=0.1, name="blocksworld_gemini25pro_heuristic_2"):
        super().__init__(name=name)
        self.clear_penalty = clear_penalty
        self.move_penalty = move_penalty
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
        Precomputes information about each method, such as the count of different action types in its subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name if hasattr(method, 'name') else str(method)

            stack_count = 0
            unstack_count = 0
            pickup_count = 0
            putdown_count = 0

            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network

            for subtask in subtasks:
                subtask_name = str(subtask.name).lower()
                if 'unstack' in subtask_name:
                    unstack_count += 1
                elif 'stack' in subtask_name:
                    stack_count += 1
                elif 'pickup' in subtask_name:
                    pickup_count += 1
                elif 'putdown' in subtask_name:
                    putdown_count += 1
            
            self.method_scores[method_name] = {
                'stack_count': stack_count,
                'unstack_count': unstack_count,
                'pickup_count': pickup_count,
                'putdown_count': putdown_count,
                'is_base_case': self._is_base_case(method, subtasks),
                'clearing_effort': unstack_count + putdown_count,
                'placing_effort': stack_count + pickup_count
            }

    def _is_base_case(self, method, subtasks):
        """
        Determines if a method is a base case (i.e., terminates a recursion).
        Assumes base cases have no subtasks or a name indicating achievement.
        """
        method_name = method.name.lower()
        if not subtasks:
            return True
        if 'achieved' in method_name or 'base' in method_name:
            return True
        return False

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a node based on its decomposition method.
        Returns a small, admissible cost.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Scores a method based on the precomputed effort of its subtasks.
        Lower scores are better.
        """
        method_name = method.name if hasattr(method, 'name') else str(method)

        if method_name not in self.method_scores:
            return 0.5 # Default penalty for unknown methods

        info = self.method_scores[method_name]

        # Base cases represent a completed goal, so they have zero cost.
        if info['is_base_case']:
            return 0

        # Calculate score based on weighted actions.
        # 'Clearing' actions (unstack, putdown) are penalized more as they are often overhead.
        # 'Moving' actions (pickup, stack) are penalized less as they directly work towards the goal.
        score = (info['clearing_effort'] * self.clear_penalty) + \
                (info['placing_effort'] * self.move_penalty)

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculate the heuristic value for the given node.
        The value is kept small to ensure admissibility.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic2(clear_penalty={self.clear_penalty}, move_penalty={self.move_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 2 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
