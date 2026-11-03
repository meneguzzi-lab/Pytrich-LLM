from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic calculates the number of lamps that are not in their goal state.
This is equivalent to the Hamming distance between the current grid state and the goal grid state.

**Bottleneck Analysis**
The primary bottleneck in the Lamps domain is achieving the correct state for all lamps on the grid.
The only action that can change a lamp's state is `flip`, but its effects can propagate, making it non-trivial
to isolate changes. The number of mismatched lamps represents the minimum number of state changes required.

**Admissibility Sketch**
The heuristic is defined as h(n) = number of lamps in state S that do not match their goal state.
1. Each mismatched lamp must have its state changed at least once to reach the goal.
2. The only primitive action that can change a lamp's state is `flip`.
3. Each `flip` action changes the state of exactly one lamp.
4. Therefore, to fix `k` mismatched lamps, a minimum of `k` `flip` actions are required over the entire plan.
The cost of the plan is the total number of primitive actions (each costing 1), so h(n) <= h*(n), where h*(n) is the
number of actions in an optimal plan. The heuristic is admissible.
"""

class LampsGemini25ProHeuristic13(Heuristic):
    """
    An admissible heuristic for the Lamps domain that computes the Hamming distance
    between the current state of the lamp grid and the goal state.
    """

    def __init__(self, lamps_gemini25pro_heuristic_13=None, name="LampsGemini25ProHeuristic13"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Map from (x, y, status) to the corresponding Fact object
        self.on_facts = {}
        # Map from (x, y) to the goal status ('Y' or 'N')
        self.goal_config = {}
        # List of all unique (x, y) coordinates
        self.all_coords = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model.
        This is called once before the search begins. It extracts static information
        about the lamps grid and the goal configuration.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name into a (predicate, [args]) tuple.
        Handles format: +predicate[arg1,arg2]
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess(self):
        """
        Extracts the grid structure and goal configuration from the model facts.
        Populates self.on_facts, self.all_coords, and self.goal_config.
        """
        coords = set()

        # 1. Parse all model facts to find all lamp coordinates and their 'on' facts.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_facts[(x, y, status)] = fact
                coords.add((x, y))

        self.all_coords = sorted(list(coords))

        # 2. Parse the goal state to build the goal configuration map.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.goal_config[(x, y)] = status

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Calculates the heuristic value: the number of mismatched lamps.
        If goals are met but task network is not empty, returns 1 as at least
        one more step (like end_game) is needed.
        """
        mismatch_count = 0

        for x, y in self.all_coords:
            goal_status = self.goal_config.get((x, y))

            # If a lamp has no specific goal state, it doesn't contribute to the heuristic.
            if goal_status is None:
                continue

            # Get the fact corresponding to the goal state for this lamp
            goal_fact = self.on_facts.get((x, y, goal_status))

            if goal_fact is not None:
                # Check if the bit for this goal fact is set in the current state
                is_in_goal_state = (state_bitwise >> goal_fact.global_id) & 1
                if not is_in_goal_state:
                    mismatch_count += 1
        
        # If all lamps match the goal but the task network is not empty,
        # we still need at least one action (e.g., to apply the 'end_game' method's precondition check).
        if mismatch_count == 0 and len(task_network) > 0:
            return 1
            
        return mismatch_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate the heuristic value for a search node.
        This must be fast as it's called for every expanded node.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"{self.name}(lamps={len(self.all_coords)})"

    def __str__(self):
        return f"{self.name} with {len(self.all_coords)} lamps"

    def __output__(self):
        return (
            f"{self.name} Info:\n"
            f"\tTotal Lamps: {len(self.all_coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
        )
