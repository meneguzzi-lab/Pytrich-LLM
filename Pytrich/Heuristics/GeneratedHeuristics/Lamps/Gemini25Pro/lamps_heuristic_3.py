from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic calculates the number of mismatched cells between the current state
and the goal state. This is equivalent to the Hamming distance between the current
and goal grid configurations.

Heuristic Design:
1.  **Bottleneck:** The primary bottleneck is the number of incorrectly configured
    lamps. Each incorrect lamp must be flipped to the correct state at least once.

2.  **Admissible Lower Bound:**
    -   `h(n) = number of mismatched cells`. This is an admissible lower bound on the
        number of required `flip` actions.
    -   **Admissibility Sketch:** Each `flip` action changes the state of exactly one
        cell `(x, y)`. Therefore, to correct `M` mismatched cells, a minimum of
        `M` `flip` actions are necessary. Since each action has a cost of 1, this
        value is a lower bound on the remaining plan cost.

This heuristic follows the 'Winning Patterns':
- P2 (State-aware cost): The cost is directly derived from the current state vs goal.
- P3 (Preprocess once, lookup fast): All goal information and fact mappings are
  preprocessed in `initialize()`. `__call__` uses fast lookups and bitwise checks.
- P4 (Identify bottleneck): The heuristic is based on correcting the bottleneck of
  mismatched lamps.
"""


class LampsGemini25ProHeuristic3(Heuristic):
    """
    A heuristic for the Lamps domain that counts the number of cells
    that are not in their goal state.
    """

    def __init__(self, lamps_gemini25pro_heuristic_3=None, name="LampsGemini25ProHeuristic3"):
        super().__init__(name=name)
        # A dictionary mapping coordinate tuples (x, y) to their goal status ('Y' or 'N').
        self.goal_on_status = {}

        # A dictionary mapping (x, y, status) tuples to their corresponding Fact object.
        self.on_facts_map = {}

        # A set of all unique (x, y) coordinate tuples in the domain.
        self.coords = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before the search starts. Preprocesses the domain to build
        fast lookup tables for goal states and fact IDs.
        """
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into a tuple
        of (predicate, [arg1, arg2, ...]).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Extracts the goal configuration and builds mappings for all `on` predicates.
        This method is called only once from initialize().
        """
        # 1. Process the goal state to build the self.goal_on_status map.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            predicate, args = self._parse_fact_name(name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.goal_on_status[(x, y)] = s

        # 2. Process all domain facts to build self.on_facts_map and collect coordinates.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args[0], args[1], args[2]
                self.on_facts_map[(x, y, s)] = fact
                self.coords.add((x, y))

    def _compute(self, state_bitwise: int, task_network: list) -> int:
        """
        Computes the heuristic value by counting mismatched cells.
        """
        mismatch_count = 0
        if not self.coords:
            return 0

        for x, y in self.coords:
            goal_status = self.goal_on_status.get((x, y))
            if goal_status is None:
                continue  # This coordinate is not specified in the goal.

            # We check if the fact representing the GOAL state for this cell is true.
            # If it's not true, the cell is in a mismatched state.
            goal_fact = self.on_facts_map.get((x, y, goal_status))

            if goal_fact:
                is_at_goal = (state_bitwise >> goal_fact.global_id) & 1
                if not is_at_goal:
                    mismatch_count += 1

        # If the board state matches the goal, but the task network is not yet empty,
        # at least one method precondition operator (cost=1) is needed to finish.
        if mismatch_count == 0 and len(task_network) > 0:
            return 1

        return mismatch_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This is called for every expanded node and must be efficient.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return float(h_value)
