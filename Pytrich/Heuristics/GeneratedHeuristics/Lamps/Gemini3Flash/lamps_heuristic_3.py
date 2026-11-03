"""
LampsGemini3FlashHeuristic3 - Domain-Dependent Heuristic for the Lamps (Lights Out) Domain.

Bottleneck Analysis:
In the Lamps domain, the primary bottleneck is the status of each grid cell. Every coordinate (x, y)
starting in the wrong status must be the target of exactly one flip (if we assume optimality) or an odd 
number of flips to reach the desired goal status. In the Pytrich HTN model, every primitive action 'flip' 
costs 1 unit. While a 'turn' or 'propagate' task can handle multiple cells, each cell in the cascade 
is changed via its own 'flip' operator, thus consuming one unit of path cost per cell status change.

Admissibility Sketch:
1. State Mismatch Term: We count the number of atoms in the goal bitmask that are not yet set in 
   the current state bitmask. Since each goal atom in this domain (e.g., +on[l0,l1,N]) represents a 
   specific cell status, and changing a cell's status requires at least one 'flip' operator (cost 1), 
   the count of unsatisfied goals is an admissible lower bound on the remaining cost.
2. Task Network Overhead: We include a small sub-unit penalty (0.01) per task in the network to 
   provide guidance towards reducing the decomposition tree, specifically to favor the 'end_game' 
   method once the goal state is achieved.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic3(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_3=None, name="LampsGemini3FlashHeuristic3"):
        super().__init__(name=name)
        self.goal_mask = 0
        self.num_goal_facts = 0
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search starts. 
        Precompute the goal bitmask for fast bitwise mismatch checking.
        """
        start_time = time.time()
        self.model = model
        
        # self.model.goals is a bitwise integer representing the required atoms.
        self.goal_mask = model.goals
        
        # Count bits for internal logging
        try:
            self.num_goal_facts = self.goal_mask.bit_count()
        except AttributeError:
            self.num_goal_facts = bin(self.goal_mask).count('1')
            
        self.preprocessing_time = time.time() - start_time
        
        # Calculate initial heuristic value
        h_value = self._compute(initial_node.state, initial_node.task_network)
        
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network: list) -> float:
        """
        Core heuristic calculation using bitwise logic for maximum efficiency.
        """
        # Calculate bits that are present in goal but missing in current state.
        # This gives exactly the number of unsatisfied goal status predicates.
        diff = self.goal_mask & ~state
        
        try:
            # bit_count() is highly optimized in Python 3.10+
            mismatches = diff.bit_count()
        except AttributeError:
            # Fallback for older Python versions
            mismatches = bin(diff).count('1')

        # Admissible mismatch base + guidance for reducing task network complexity.
        # len(task_network) ensures we prefer nodes that resolve primitive actions and
        # close recursive play tasks quickly.
        return float(mismatches + 0.01 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every node expansion. parent_node is the parent, node is the child.
        Must be fast.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"LampsGemini3FlashHeuristic3 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Goal Atoms: {self.num_goal_facts}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
