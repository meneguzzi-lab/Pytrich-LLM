from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

Bottleneck Analysis:
The bottleneck in the Lamps domain is the total number of status changes (flips) required 
to satisfy all goal conditions, as each flip is a mandatory primitive action with a unit cost.

Admissibility Sketch:
1. Mismatch Count: Each cell (x, y) whose status in the current state does not match the 
   goal status must be targeted by at least one 'flip' action. Since each 'flip' action 
   costs exactly 1 unit and only affects a single cell, the number of currently 
   unsatisfied 'on' goal atoms is an absolute lower bound on the remaining plan cost.
2. Decomposition Cost: Pytrich charges 1 for primitive actions including flips and 
   precondition checks. Since satisfying a mismatch requires at least one flip,
   the mismatch count remains a safe and admissible lower bound regardless of method expansion costs.
"""

class LampsGemini3FlashHeuristic6(Heuristic):
    def __init__(self, lamps_gemini3flash_heuristic_6=None, name="LampsGemini3FlashHeuristic6"):
        super().__init__(name=name)
        self.goal_bits = []
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search starts. Identifies which fact IDs correspond to goal conditions.
        """
        start_time = time.time()
        self.model = model

        # 1. Get explicit names of goal facts from bitwise model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)

        # 2. Build a mapping for fast ID lookup
        name_to_id = {f.name: f.global_id for f in self.model.facts}

        # 3. Store the global IDs of facts that represent the goal state
        self.goal_bits = []
        for g_name in goal_fact_names:
            if g_name in name_to_id:
                self.goal_bits.append(name_to_id[g_name])

        self.preprocessing_time = time.time() - start_time

        # Initial compute
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise: int) -> float:
        """
        Computes the number of unsatisfied goal atoms.
        Each unsatisfied atom represents a cell that MUST be flipped at least once.
        """
        mismatches = 0
        for bit_id in self.goal_bits:
            # If the bit is not set in current state, the goal atom is false
            if not ((state_bitwise >> bit_id) & 1):
                mismatches += 1
        
        return float(mismatches)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every node expansion. Returns the mismatch count as the lower bound.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        """
        Utility to parse grounded fact strings: +predicate[arg1,arg2]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def __repr__(self):
        return f"{self.name}(goal_atoms={len(self.goal_bits)})"

    def __str__(self):
        return f"{self.name} checking {len(self.goal_bits)} goal atoms"

    def __output__(self):
        return (
            f"Lamps Gemini3 Flash Heuristic 6 Info:\n"
            f"\tName: {self.name}\n"
            f"\tGoal Facts Monitored: {len(self.goal_bits)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Search Calls: {self.calls}\n"
            f"\tMin H-Value Found: {self.min_hvalue}\n"
        )
