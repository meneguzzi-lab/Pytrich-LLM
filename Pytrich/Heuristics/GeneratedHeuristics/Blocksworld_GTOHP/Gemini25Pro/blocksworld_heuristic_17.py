import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic17(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    Rationale:
    The primary challenge in Blocksworld is resolving conflicts to achieve a goal tower.
    To place block A on block B, both A and B must be clear. This often requires
    moving other blocks out of the way. A good heuristic should estimate the
    number of moves required to resolve these conflicts and build the target configuration.

    This heuristic works by statically analyzing the methods (decompositions) of the domain.
    It estimates the cost of a method based on the number of primitive 'move' actions
    (unstack, stack, pickup, putdown) and 'clearing' subtasks its decomposition contains.
    Methods that lead to shorter action sequences are preferred, guiding the search towards
    more efficient plans.

    Heuristic Value h(n):
    The heuristic value for a node 'n' expanded by a method 'm' is calculated as:
    h(n) = (count_of_moves_in_m * move_penalty) + (count_of_clear_tasks_in_m * clear_penalty)

    This approach is admissible if the penalties are set to be less than or equal to the
    true cost of the actions (typically 1). It respects the hierarchy by evaluating
    decompositions and is computationally efficient as it relies on a one-time preprocessing
    step.
    """

    def __init__(self, blocksworld_gemini25pro_heuristic_17: dict = None, name="blocksworld_gemini25pro_heuristic_17"):
        """
        Initializes the Blocksworld heuristic.

        Args:
            blocksworld_gemini25pro_heuristic_17 (dict, optional): A configuration dictionary. 
                Can contain 'move_penalty' and 'clear_penalty'. Defaults to None.
            name (str): The name of the heuristic.
        """
        super().__init__(name=name)
        config = blocksworld_gemini25pro_heuristic_17 if blocksworld_gemini25pro_heuristic_17 is not None else {}
        
        # Penalty for actions that move a block (e.g., stack, unstack).
        # Set to 1.0 to count each physical move.
        self.move_penalty = config.get('move_penalty', 1.0)
        
        # Penalty for subtasks that aim to clear a block. This represents the effort
        # of achieving a necessary precondition.
        self.clear_penalty = config.get('clear_penalty', 0.5)
        
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by preprocessing the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain to estimate their intrinsic cost.
        This cost is based on the number of move and clear operations in their subtasks.
        The results are cached for fast lookup during search.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'decompositions'):
            methods.extend(self.model.decompositions)
        if hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)
        
        # Use a set to handle potential duplicates from different sources
        unique_methods = list(set(methods))

        for method in unique_methods:
            method_name = method.name
            
            move_count = 0
            clear_count = 0
            
            subtasks = getattr(method, 'task_network', [])
            
            if not subtasks:  # This is a base case method.
                self.method_scores[method_name] = {
                    'move_count': 0,
                    'clear_count': 0,
                    'is_base_case': True,
                }
                continue

            for subtask in subtasks:
                subtask_name = subtask.name.lower()
                
                # Primitive actions that involve moving a block.
                if any(op in subtask_name for op in ['unstack', 'stack', 'pickup', 'putdown']):
                    move_count += 1
                
                # Abstract tasks that require clearing a block as a sub-goal.
                if 'clear' in subtask_name or 'make-clear' in subtask_name:
                    clear_count += 1
            
            self.method_scores[method_name] = {
                'move_count': move_count,
                'clear_count': clear_count,
                'is_base_case': False,
            }

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        The value is determined by scoring the method used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        return self._score_method(node.method)

    def _score_method(self, method) -> float:
        """
        Scores a given method based on the precomputed analysis.
        A lower score indicates a more promising method (lower estimated cost).
        """
        score = 0.0
        method_name = method.name
        
        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            
            if info['is_base_case']:
                return 0.0  # Base cases are terminal and have zero remaining cost.
            
            # The estimated cost is a sum of penalties for required actions.
            score += info['move_count'] * self.move_penalty
            score += info['clear_count'] * self.clear_penalty
        else:
            # Fallback for primitive actions that might not be part of a method's subtasks
            # in the initial task network.
            method_name_lower = method_name.lower()
            if any(op in method_name_lower for op in ['unstack', 'stack', 'pickup', 'putdown']):
                score = self.move_penalty  # A single move action
        
        return max(score, 0.0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic17(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tMove Penalty: {self.move_penalty}\n"
            f"\tClear Penalty: {self.clear_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
