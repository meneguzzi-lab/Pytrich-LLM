from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Lamps grid domain.

    1. Bottleneck Identification:
       The core constraint is the grid status. Every cell (x, y) that currently differs from its 
       goal status must undergo at least one 'flip' action. Since status is binary (Y/N) and 
       each 'flip' action affects exactly one cell, the number of incorrect cells is a 
       strict lower bound on the number of 'flip' actions needed.

    2. Admissible Lower Bounds:
       - Term 1 (Grid Hamming Distance): The count of 'on(x, y, status)' goal facts not yet 
         satisfied in the state. Each 'flip' primitive action changes one cell. Thus, 
         h_hamming is a strict lower bound on the cost of flips.
       - Term 2 (Task Cost): Any 'flip' or 'turn' task already in the network represents at 
         least one unavoidable action cost. 

    This heuristic is admissible because it counts only the necessary status-changing actions 
       (flips) and uses a sub-unit tie-breaker for remaining task complexity.
    """

    def __init__(self, lamps_gemini3flash_heuristic_5=None, name="LampsGemini3FlashHeuristic5"):
        super().__init__(name=name)
        self.goal_fact_ids = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once at the start of the search. 
        Pre-extracts bit positions for goal facts to enable O(1) Hamming calculation.
        """
        self.model = model
        self.goal_fact_ids = []
        
        # Extract indices of bits representing goal facts from the bitwise model.goals
        if hasattr(model, 'goals') and model.goals:
            g_state = model.goals
            idx = 0
            # Efficiently collect indices of set bits
            while g_state > 0:
                if g_state & 1:
                    self.goal_fact_ids.append(idx)
                g_state >>= 1
                idx += 1

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network) -> float:
        """
        Calculates a lower bound on the remaining cost.
        """
        # 1. Hamming Distance on the Grid
        # Each cell currently not in its goal status needs at least one flip action.
        h_hamming = 0
        for fid in self.goal_fact_ids:
            # Fast bitwise check to see if goal fact at fid is true in current state
            if not ((state >> fid) & 1):
                h_hamming += 1

        # 2. Guidance / Tie-breaking
        # We add a small magnitude of the task network size to guide the search towards
        # decompositions that simplify the plan, ensuring play/propagate tasks are resolved.
        return float(h_hamming + (0.001 * len(task_network)))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called during node expansion.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        """
        Parses +predicate[arg1,arg2] format.
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
