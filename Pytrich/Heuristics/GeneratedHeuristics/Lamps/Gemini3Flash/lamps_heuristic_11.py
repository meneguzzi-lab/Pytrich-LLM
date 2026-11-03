from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini3FlashHeuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    1. Bottleneck Analysis:
    The fundamental bottleneck in the Lamps domain is the requirement to reconcile the status (on/off, Y/N) 
    of every coordinate in the grid to match the goal configuration. The only primitive action is 'flip',
    which costs 1. While the HTN orchestrates flips through 'turn' and 'propagate' tasks, every status
    change of a lamp from one state to another requires at least one 'flip' action.

    2. Admissible Lower Bounds:
    - Term 1 (Mismatch Count): For every coordinate (x, y), if the goal specifies a status S and the 
      current status is not S, the lamp must be flipped at least once. Since each flip costs 1 unit,
      the total number of status mismatches is a robust and admissible lower bound for the remaining cost.
    - Tie-breaking: A small sub-unit penalty based on the task network size favors progress through the
      hierarchy and reaching the base cases of recursive propagation.
    """

    def __init__(self, lamps_gemini3flash_heuristic_11=None, name="LampsGemini3FlashHeuristic11"):
        super().__init__(name=name)
        self.lamps_gemini3flash_heuristic_11 = lamps_gemini3flash_heuristic_11
        self.goal_bits = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Preprocesses the goal state into a bitmask.
        """
        self.model = model
        # Pytrich goals are stored as a bitwise integer.
        self.goal_bits = model.goals
        
        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        """
        Calculates the heuristic value based on goal satisfaction and network state.
        
        h = mismatch_count + tie_breaker
        """
        # Mismatch Count Logic:
        # Bits that are set in goal_bits but NOT in state_bitwise are unsatisfied goal facts.
        # These are found via: (goals AND (NOT state)).
        unsatisfied_bits = self.goal_bits & ~state_bitwise
        
        # Each unsatisfied lamp status represents at least 1 mandatory 'flip' action (cost 1).
        # bin(x).count('1') is an efficient way to count set bits in Python integers.
        mismatch_count = float(bin(unsatisfied_bits).count('1'))

        # Tie-breaker Logic:
        # Pytrich's A* cost includes primitive check actions for method preconditions.
        # Adding a small magnitude for the network length helps prioritize 'end' methods
        # and deeper levels of the hierarchy where progress is being made.
        tie_breaker = 0.001 * len(task_network)

        return mismatch_count + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Evaluates every node expanded during search.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini3FlashHeuristic11(goal_bits={bin(self.goal_bits).count('1')})"
