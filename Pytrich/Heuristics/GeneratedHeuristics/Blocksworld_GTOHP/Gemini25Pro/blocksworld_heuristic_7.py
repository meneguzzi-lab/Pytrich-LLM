import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Blocksworld domain.
# This heuristic is designed to work with the Pytrich Planner framework.
# It operates by statically analyzing the domain's methods during a one-time
# preprocessing step. It estimates the cost of applying a method by summing
# the costs of the primitive actions (stack, unstack, pickup, putdown)
# in its decomposition. Methods that represent base cases (i.e., the goal
# is already achieved) are assigned a cost of zero.
# This approach provides a fast and admissible heuristic that guides the search
# towards methods that result in shorter plans.
class BlocksworldGemini25ProHeuristic7(Heuristic):
    def __init__(self, blocksworld_gemini25pro_heuristic_7):
        """
        Initializes the heuristic with configurable costs for primitive actions.

        :param blocksworld_gemini25pro_heuristic_7: A dictionary of parameters. Expected keys are:
                                      'stack_cost', 'unstack_cost', 'pickup_cost',
                                      'putdown_cost'.
        """
        # Costs are kept low and fractional to ensure the heuristic remains admissible,
        # as the true cost of any action is 1.
        self.stack_cost = blocksworld_gemini25pro_heuristic_7.get("stack_cost", 0.2)
        self.unstack_cost = blocksworld_gemini25pro_heuristic_7.get("unstack_cost", 0.2)
        self.pickup_cost = blocksworld_gemini25pro_heuristic_7.get("pickup_cost", 0.1)
        self.putdown_cost = blocksworld_gemini25pro_heuristic_7.get("putdown_cost", 0.1)
        
        super().__init__(name="blocksworld_gemini25pro_heuristic_7")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing costs for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model once.
        It calculates a cost for each method based on the primitive Blocksworld
        actions found in its subtask network. Base cases are assigned a cost of 0.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name = method.name
            subtasks = method.task_network

            # A method with no subtasks is a base case (e.g., goal is already met).
            if not subtasks:
                self.method_scores[method_name] = {
                    'cost': 0.0,
                    'is_base_case': True
                }
                continue

            # Count the occurrences of each type of primitive action.
            action_counts = {
                'stack': 0, 'unstack': 0,
                'pickup': 0, 'putdown': 0
            }

            for subtask in subtasks:
                # This heuristic assumes subtasks are Operators with a 'name' attribute.
                if hasattr(subtask, 'name'):
                    subtask_name = subtask.name.lower()
                    for action_type in action_counts.keys():
                        if action_type in subtask_name:
                            action_counts[action_type] += 1
                            break

            # Calculate the total heuristic cost for the method.
            cost = (action_counts['stack'] * self.stack_cost +
                    action_counts['unstack'] * self.unstack_cost +
                    action_counts['pickup'] * self.pickup_cost +
                    action_counts['putdown'] * self.putdown_cost)

            self.method_scores[method_name] = {
                'cost': cost,
                'is_base_case': False
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node by scoring its method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        return self._score_method(node.method)

    def _score_method(self, method):
        """
        Retrieves the pre-calculated score for a given method.
        Returns a low cost for promising methods and 0 for base cases.
        """
        method_name = method.name
        
        if method_name in self.method_scores:
            return self.method_scores[method_name]['cost']

        # Fallback for any method not found during preprocessing.
        # A default cost of 1.0 is a safe, uninformative estimate.
        return 1.0

    def __call__(self, parent_node, node):
        """
        The main heuristic function called by the planner.
        It calculates the heuristic value and adds a small tie-breaking component.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # Add a small penalty based on the plan depth (g-cost).
        # This acts as a tie-breaker, preferring nodes that are closer to the
        # root of the search tree when heuristic estimates are equal.
        h_value += node.g * 0.001

        super().update_info(h_value)
        return h_value
    
    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic7(costs=[s:{self.stack_cost},u:{self.unstack_cost},p:{self.pickup_cost},d:{self.putdown_cost}])"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 7 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tStack Cost: {self.stack_cost}\n"
            f"\tUnstack Cost: {self.unstack_cost}\n"
            f"\tPickup Cost: {self.pickup_cost}\n"
            f"\tPutdown Cost: {self.putdown_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
