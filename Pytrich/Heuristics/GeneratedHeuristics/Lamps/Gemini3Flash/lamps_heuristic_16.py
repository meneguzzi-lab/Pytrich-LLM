"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
The primary bottleneck in the Lamps domain is the number of 'flip' operations required to reach the target 
configuration of cell statuses. Each 'flip' action costs 1 unit. While a single 'turn' task can trigger 
a chain of flips via propagation, each individual flip in that chain is a primitive action that incurs 
a cost. Thus, the total number of status mismatches between the current state and the goal state 
provides a robust, admissible lower bound on the remaining path cost.

Admissibility Sketches:
1. Mismatch Term: Each cell whose status (on ?x ?y status) does not match the goal must be flipped at least 
   once. Since each flip costs 1 and affects only one cell, the number of mismatches is a lower bound on flips.
2. Non-Flip Operator Term: Any primitive operators already present in the task network (such as method 
   preconditions or administrative no-ops) have fixed costs that must be paid. These actions do not satisfy 
   cell status goals. Summing their costs with the mismatch term remains admissible.
3. Guidance: A small penalty (0.01) is added for abstract tasks in the network to guide the search 
   towards decomposition and completion.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic16(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_16=None, name="LampsGemini3FlashHeuristic16"):
        super().__init__(name=name)
        self.flip_op_ids = set()
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Preprocesses operator types and calculates initial h.
        """
        start_time = time.time()
        self.model = model

        # Identify global IDs for all flip operators for fast lookup in __call__
        for op in self.model.operators:
            if "flip" in op.name.lower():
                self.flip_op_ids.add(op.global_id)

        self.preprocessing_time = time.time() - start_time

        # Initial heuristic value
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        """
        Calculates the heuristic value based on state mismatches and task network overhead.
        """
        # 1. Count mismatched goal facts
        # self.model.goals is a bitset where each 1 represents a fact that must be true.
        # (~state_bitwise) flips all bits. ANDing them finds goals that are 0 in the current state.
        mismatched_bits = self.model.goals & (~state_bitwise)
        h = bin(mismatched_bits).count('1')

        # 2. Add costs for existing tasks in the network
        for task in task_network:
            # Check if the task is an Operator by its global_id range
            if self.model.iop_init <= task.global_id <= self.model.iop_end:
                # If it's a flip, its contribution to reaching the goal is already counted in mismatches.
                # If it's not a flip (e.g., a method precondition action), it's an additional cost.
                if task.global_id not in self.flip_op_ids:
                    h += getattr(task, 'cost', 1)
            else:
                # It's an AbstractTask. We add a small guidance penalty.
                # Decompositions cost 0, but will eventually produce actions.
                h += 0.01

        return float(h)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every node expansion. Must be fast.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic16(precomputed_flips={len(self.flip_op_ids)})"

    def __str__(self):
        return f"LampsGemini3FlashHeuristic16 [Mismatches + TN cost]"
