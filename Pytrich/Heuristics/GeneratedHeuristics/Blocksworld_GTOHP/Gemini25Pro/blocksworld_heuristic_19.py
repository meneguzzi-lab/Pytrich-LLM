import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic19(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost of a decomposition method based on the
    primitive actions it contains. The core idea is to assign different costs
    to actions that "clear" blocks versus actions that directly build the
    goal configuration.

    - "Clearing" actions (unstack, putdown) are penalized more, as they are
      often necessary but do not directly make progress on the goal towers.
    - "Building" actions (pickup, stack) are penalized less, as they are
      directly contributing to the desired state.
    - Base case methods (which terminate a decomposition) are given a cost of 0.

    The heuristic values are kept small and are based on action counts to
    remain admissible, assuming each primitive action has a cost of at least 1.
    The heuristic provides guidance by preferring methods that build goal towers
    more directly over those that only clear other blocks out of the way.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_19):
        super().__init__(name="blocksworld_gemini25pro_heuristic_19")
        # Tunable penalties for different action types.
        # 'clear_penalty' is for moving blocks out of the way.
        # 'build_penalty' is for moving blocks into their goal positions.
        self.clear_penalty = blocksworld_gemini25pro_heuristic_19.get("clear_penalty", 0.2)
        self.build_penalty = blocksworld_gemini25pro_heuristic_19.get("build_penalty", 0.1)
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic, pre-calculating scores for each method.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyze all decomposition methods in the domain model once.
        It counts the types of primitive actions (subtasks) within each method
        to estimate its cost and purpose (e.g., clearing vs. building).
        """
        if not self.model:
            return

        methods = self.model.decompositions
        if not methods:
            return

        for method in methods:
            method_name = method.name
            
            action_counts = {
                'unstack': 0,
                'putdown': 0,
                'pickup': 0,
                'stack': 0
            }
            is_base_case = self._is_base_case(method)

            if not is_base_case:
                subtasks = method.task_network
                for subtask in subtasks:
                    subtask_name = subtask.name.lower()
                    for action_type in action_counts.keys():
                        if action_type in subtask_name:
                            action_counts[action_type] += 1
            
            self.method_scores[method_name] = {
                'counts': action_counts,
                'is_base_case': is_base_case,
                'subtask_count': len(method.task_network)
            }

    def _is_base_case(self, method):
        """
        Determines if a method is a base case (i.e., it has no subtasks).
        Base cases represent the termination of a recursive task decomposition.
        """
        return not method.task_network

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is based on the precomputed scores for the node's method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        if method_name not in self.method_scores:
            return 1.0 

        info = self.method_scores[method_name]

        if info['is_base_case']:
            return 0.0

        counts = info['counts']
        score = (counts['unstack'] + counts['putdown']) * self.clear_penalty
        score += (counts['pickup'] + counts['stack']) * self.build_penalty
        
        return max(score, 0.0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        h_value = min(h_value, 1.0)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic19(clear_penalty={self.clear_penalty}, build_penalty={self.build_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 19 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tBuild Penalty: {self.build_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
