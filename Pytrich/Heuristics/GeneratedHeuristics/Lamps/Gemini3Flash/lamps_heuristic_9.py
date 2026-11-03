from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class LampsGemini3FlashHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    1. Bottleneck Analysis:
       The core bottleneck in this domain is the required status of each lamp (coordinate).
       Every lamp has a goal status (Y or N). To change a lamp's status, a 'flip' primitive
       action must be applied to that specific coordinate. Since each 'flip' costs 1
       and affects exactly one coordinate, the number of coordinates currently in the
       wrong status is a direct lower bound on the number of 'flip' actions needed.

    2. Admissible Lower Bounds:
       - Term 1: Mismatched Goal Count. Every goal fact (on ?x ?y ?status) that is NOT
         true in the current state requires at least one 'flip' operation. 
         Admissibility: Each 'flip' costs 1 and changes exactly one coordinate's status.
       - Term 2: Abstract Task Count. Every abstract task currently in the task network
         (e.g., play, turn, propagate) must eventually be decomposed. In Pytrich, every
         decomposition method has a precondition check (a grounded operator) that costs 1.
         Admissibility: These precondition checks are distinct from the 'flip' actions.

    3. Strategy:
       H = (Number of mismatched goal cells) + (Number of abstract tasks in network).
       This is admissible because it separately counts the cost of necessary flips and the
       cost of the control structure (method decompositions) required to apply those flips.
    """

    def __init__(self, lamps_gemini3flash_heuristic_9=None, name="LampsGemini3FlashHeuristic9"):
        super().__init__(name=name)
        self.goal_fact_ids = []
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Precompute the global IDs of all goal facts.
        """
        start_time = time.time()
        self.model = model
        
        self.goal_fact_ids = []
        if hasattr(self.model, 'goals') and self.model.goals:
            # Iterate through the bitmask of the goal state to find which fact IDs are required.
            # self.model.goals is a bitwise integer where each bit corresponds to a fact's global_id.
            for i in range(self.model.goals.bit_length()):
                if (self.model.goals >> i) & 1:
                    self.goal_fact_ids.append(i)

        self.preprocessing_time = time.time() - start_time
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        """
        Calculate the heuristic value based on mismatched goals and task structure.
        """
        # 1. Count mismatched goals
        mismatched = 0
        for fid in self.goal_fact_ids:
            # If the goal fact is NOT true in the current state
            if not ((state_bitwise >> fid) & 1):
                mismatched += 1

        # 2. Count abstract tasks in the network
        # Abstract tasks have a 'decompositions' attribute, while primitive Operators do not.
        num_abstract = 0
        for task in task_network:
            if hasattr(task, 'decompositions'):
                num_abstract += 1

        # Total cost is the sum of these independent requirements.
        return float(mismatched + num_abstract)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called during search for every node expansion.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(goal_count={len(self.goal_fact_ids)})"

    def __str__(self):
        return f"{self.name} monitoring {len(self.goal_fact_ids)} goals"

    def __output__(self):
        return (
            f"{self.name} Info:\n"
            f"\tGoals Monitored: {len(self.goal_fact_ids)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Node Calls: {self.calls}\n"
            f"\tMin H-Value Found: {self.min_hvalue}\n"
        )
