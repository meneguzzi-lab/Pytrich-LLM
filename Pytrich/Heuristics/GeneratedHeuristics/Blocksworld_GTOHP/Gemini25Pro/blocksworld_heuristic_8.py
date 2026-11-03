import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic8(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic estimates the cost to reach a goal from a given search node
    by analyzing the method chosen for decomposition. It prioritizes methods that
    achieve goals with fewer preparatory steps (like clearing blocks).

    The heuristic value is calculated based on a pre-analysis of all methods
    in the domain. Methods are scored based on the number and type of subtasks
    they decompose into.

    - Base Case Methods: Methods that represent a task already being achieved
      (e.g., an empty decomposition) are given a cost of 0.
    - Action-Oriented Methods: Methods decomposing into primitive actions
      (e.g., pick-up, stack) are assigned a small cost per action.
    - Preparatory Methods: Methods that involve clearing other blocks are given a
      higher penalty, as this implies additional moves are required before the
      primary goal can be addressed.

    This approach ensures the heuristic is admissible (by underestimating the
    true cost) while providing guidance to the planner by favoring more direct
    solution paths.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_8=None, name="blocksworld_gemini25pro_heuristic_8", move_penalty=0.1, clear_penalty=0.2):
        """
        Initializes the BlocksworldGemini25ProHeuristic8.

        Args:
            blocksworld_gemini25pro_heuristic_8: Parameter included to match the specific generation request. Not used.
            name (str): The name of the heuristic.
            move_penalty (float): A small cost associated with each primitive move action (pick-up, stack).
            clear_penalty (float): A higher cost for methods involving clearing a block, which implies more work.
        """
        super().__init__(name=name)
        self.move_penalty = move_penalty
        self.clear_penalty = clear_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-analyzing the domain's methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods (decompositions) in the domain model to assign a
        complexity score to each. This is done once at the beginning to speed
        up heuristic calculation during the search.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            method_lower = method_name.lower()

            move_action_count = 0
            clear_task_count = 0
            is_base_case = False

            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network
            
            if not subtasks:
                is_base_case = True
            else:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)
                    
                    subtask_lower = subtask_name.lower()

                    # Count primitive move actions
                    if any(move_op in subtask_lower for move_op in ['pick-up', 'put-down', 'stack', 'unstack']):
                        move_action_count += 1
                    
                    # Check for tasks related to clearing blocks
                    if 'clear' in subtask_lower:
                        clear_task_count += 1
            
            # Additional check for base cases based on method naming conventions
            if not is_base_case and 'already' in method_lower:
                is_base_case = True

            self.method_scores[method_name] = {
                'move_action_count': move_action_count,
                'clear_task_count': clear_task_count,
                'is_base_case': is_base_case
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is based on the precomputed score of the method used to
        generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no additional cost from this node

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
        
        if method_name in self.method_scores:
            info = self.method_scores[method_name]
            
            if info['is_base_case']:
                return 0.0 # Base cases have zero cost as they complete a task
            
            # Calculate cost based on penalties
            score = 0.0
            score += info['move_action_count'] * self.move_penalty
            score += info['clear_task_count'] * self.clear_penalty
            
            return score
        
        return 0.1 # Default small cost for unknown methods

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        Returns a small, admissible cost estimate for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        
        # Ensure the heuristic value is a small, non-negative number to maintain admissibility
        h_value = max(h_value, 0.0)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic8(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __str__(self):
        return f"BlocksworldGemini25ProHeuristic8(move_penalty={self.move_penalty}, clear_penalty={self.clear_penalty})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 8 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMove action penalty: {self.move_penalty}\n"
            f"\tClear task penalty: {self.clear_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
