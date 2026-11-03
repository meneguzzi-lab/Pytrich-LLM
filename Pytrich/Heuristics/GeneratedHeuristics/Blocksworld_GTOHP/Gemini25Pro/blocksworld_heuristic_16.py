import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BlocksworldGemini25ProHeuristic16(Heuristic):
    """
    A domain-dependent heuristic for the Blocksworld domain.

    This heuristic guides the search by estimating the cost of applying a 
    decomposition method. It performs a static analysis of the domain's methods
    during an initialization phase. Each method is scored based on the complexity
    and type of its subtasks.

    The heuristic assigns costs based on the following principles:
    - Base case methods (with no subtasks) have a cost of 0, as they complete a task.
    - Methods decomposing into primitive actions (pickup, putdown, stack, unstack) 
      have a cost proportional to the number of actions.
    - Methods that generate complex sub-problems (e.g., clearing a block) are given
      a higher estimated cost, discouraging decompositions that lead to more work.

    This approach favors direct, simple solutions and helps the planner prioritize
    methods that make immediate progress towards a primitive plan.
    """
    def __init__(self, blocksworld_gemini25pro_heuristic_16={"action_cost": 1.0, "clear_cost": 2.0}, name="blocksworld_gemini25pro_heuristic_16"):
        """
        Initializes the Blocksworld heuristic.

        Args:
            blocksworld_gemini25pro_heuristic_16 (dict): A dictionary containing cost parameters.
                - "action_cost": The base cost for a primitive action.
                - "clear_cost": The estimated cost for a 'clear' subtask.
            name (str): The name of the heuristic.
        """
        super().__init__(name=name)
        self.action_cost = blocksworld_gemini25pro_heuristic_16.get("action_cost", 1.0)
        self.clear_cost = blocksworld_gemini25pro_heuristic_16.get("clear_cost", 2.0)
        self.method_scores = {}
        self.preprocessing_time = 0
        self.primitive_keywords = ['pickup', 'putdown', 'stack', 'unstack']
        self.complex_task_keywords = {'clear': self.clear_cost}

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        initial_h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, initial_h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the model and assigns a heuristic score to each.
        The score estimates the planning effort required by the method's subtasks.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = 0
            is_base_case = not method.task_network

            if not is_base_case:
                for subtask in method.task_network:
                    subtask_name = subtask.name.lower()
                    
                    # Check if the subtask is a known primitive action
                    is_primitive = any(keyword in subtask_name for keyword in self.primitive_keywords)
                    if is_primitive:
                        score += self.action_cost
                        continue

                    # Check if the subtask is a known complex task
                    is_complex = False
                    for keyword, cost in self.complex_task_keywords.items():
                        if keyword in subtask_name:
                            score += cost
                            is_complex = True
                            break
                    if is_complex:
                        continue
                    
                    # Penalize recursive or other abstract tasks with a default cost
                    score += self.action_cost
            
            self.method_scores[method.name] = {
                'score': score,
                'is_base_case': is_base_case,
                'subtask_count': len(method.task_network)
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        The value is retrieved from the pre-computed scores based on the node's method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0  # No method applied, no estimated cost

        method_name = node.method.name
        method_info = self.method_scores.get(method_name)

        if method_info:
            return method_info['score']
        else:
            # Fallback for methods not found during preprocessing (should not happen)
            return len(node.method.task_network) * self.action_cost

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini25ProHeuristic16(action_cost={self.action_cost}, clear_cost={self.clear_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """
        Returns a formatted string with information about the heuristic's configuration and performance.
        """
        return (
            f"Blocksworld Gemini 2.5 Pro Heuristic 16 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPrimitive Action Cost: {self.action_cost}\n"
            f"\tClear Task Cost: {self.clear_cost}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
