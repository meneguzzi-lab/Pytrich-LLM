import time
from typing import List, Union
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask

class RoverGemini25ProHeuristic21(Heuristic):
    """
    Domain-dependent heuristic for the HDDL ROVER domain.

    Rationale:
    The primary cost in the ROVER domain stems from rover actions, especially movement (`navigate`),
    data collection (`sample_*`, `take_image`), and communication (`communicate_*`). This heuristic
    guides the planner by assigning a small, admissible cost to each decomposition method based on
    the complexity of its subtasks.

    The heuristic operates in two stages:
    1. Preprocessing (Initialization): It analyzes every method in the domain and calculates a
       score. This score is a lower-bound estimate of the plan cost that will result from choosing
       that decomposition. A cost is assigned for each primitive action and a smaller cost for each
       abstract task introduced. This encourages the planner to prefer methods that are simpler,
       more direct, or are base cases.
    2. Evaluation (During Search): The heuristic value for a new search node is simply the
       pre-computed score of the method that was used to generate it. This is computationally
       very fast, as it's just a dictionary lookup.

    This approach is admissible because the costs are small, positive values that underestimate the
    true cost (ignoring most navigation and all future decomposition costs). It effectively guides
    the search away from unnecessarily complex or recursive decompositions when simpler ones are available.
    """
    def __init__(self, rover_gemini25pro_heuristic_21="rover_gemini25pro_heuristic_21", primitive_action_cost=0.1, abstract_task_cost=0.05):
        super().__init__(name="RoverGemini25ProHeuristic21")
        self.primitive_action_cost = primitive_action_cost
        self.abstract_task_cost = abstract_task_cost
        self.method_scores = {}
        self.preprocessing_time = 0.0

        # Define operators that contribute to the plan cost.
        # Control-flow operators like 'nop', 'visit', 'unvisit' are excluded.
        self.costly_operator_names = {
            "navigate", "sample_soil", "sample_rock", "drop", "calibrate", 
            "take_image", "communicate_soil_data1", "communicate_soil_data2", 
            "communicate_rock_data1", "communicate_rock_data2", "communicate_image_data"
        }

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for each decomposition method.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # The initial node has no parent method, so its heuristic contribution is 0.
        # The heuristic value is accumulated as decompositions are applied.
        return super().initialize(model, 0)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a score to each.
        This score serves as an admissible estimate of the method's complexity and cost.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            score = 0
            if not hasattr(method, 'task_network'):
                self.method_scores[method.name] = 0
                continue

            for subtask in method.task_network:
                # Penalize methods that decompose into costly primitive actions.
                if isinstance(subtask, Operator) and subtask.name in self.costly_operator_names:
                    score += self.primitive_action_cost
                # Penalize methods that introduce more abstract tasks, as they require
                # further decomposition, adding to plan length and complexity.
                elif isinstance(subtask, AbstractTask):
                    score += self.abstract_task_cost
            
            self.method_scores[method.name] = score

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """
        Calculates the heuristic value for a new node.
        The value is the pre-calculated score of the method used to generate the node,
        estimating the immediate cost added by this decomposition step.
        """
        h_value = 0.0
        if hasattr(node, 'method') and node.method:
            # Retrieve the pre-computed score for the applied method.
            h_value = self.method_scores.get(node.method.name, 0.0)
        
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini25ProHeuristic21(primitive_cost={self.primitive_action_cost}, abstract_cost={self.abstract_task_cost})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Rover Gemini25Pro Heuristic 21 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Scored: {len(self.method_scores)}\n"
            f"\tPrimitive Action Cost: {self.primitive_action_cost}\n"
            f"\tAbstract Task Cost: {self.abstract_task_cost}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
