import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsGemini25ProHeuristic16(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    This heuristic calculates the number of lamps that are not in their goal state.

    **Bottleneck Analysis:**
    The core difficulty in the Lamps domain is the cascading 'propagate' effect. A single 'turn' (player move) on one lamp can trigger a chain reaction of 'flip' actions along a row and column, making it hard to predict the full outcome of a move. This complex interaction is the primary bottleneck.

    **Admissibility Sketch:**
    The heuristic value is the count of lamps whose current status ('Y' or 'N') does not match their goal status. This is an admissible lower bound on the remaining cost for the following reason:
    1. The cost model counts each primitive 'flip' action as 1 unit of cost.
    2. To change a lamp from an incorrect state to its correct goal state, it must be the target of a 'flip' action at least once.
    3. Therefore, the number of currently mismatched lamps is a lower bound on the minimum number of 'flip' actions required to reach the goal. This heuristic is a variation of the Hamming distance.
    """

    def __init__(self, lamps_gemini25pro_heuristic_16=None, name="LampsGemini25ProHeuristic16"):
        super().__init__(name=name)
        # Preprocessed data structures for fast lookups
        self.goal_config = {}  # Maps (x, y) coordinates to their goal status ('Y' or 'N')
        self.on_fact_ids = {}  # Maps (x, y, status) tuples to the global_id of the corresponding fact
        self.coords = set()    # A set of all (x, y) coordinate tuples in the grid

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+on[l0,l1,Y]' into a predicate and arguments.
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

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Preprocesses the model to build data structures
        for efficient heuristic calculation.
        """
        self.model = model

        # 1. Parse the goal state to build the target configuration map.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_fact_names:
            predicate, args = self._parse_fact_name(fact_name)
            if predicate == 'on' and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.goal_config[(x, y)] = status
                self.coords.add((x, y))

        # 2. Parse all model facts to map 'on' predicates to their global IDs.
        # This allows for O(1) lookups of fact IDs based on coordinates and status.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == 'on' and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                # Only store facts relevant to the grid coordinates defined by the goal.
                if (x, y) in self.coords:
                    self.on_fact_ids[(x, y, status)] = fact.global_id
        
        # Calculate and store the initial heuristic value.
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise: int) -> int:
        """
        Computes the heuristic value by counting mismatched lamps.
        This is the core calculation, designed to be fast.
        """
        mismatched_count = 0
        
        # Iterate over every lamp coordinate in the grid.
        for x, y in self.coords:
            goal_status = self.goal_config.get((x, y))
            if goal_status is None:
                continue

            # Check the current status of the lamp. We assume a binary state ('Y'/'N').
            # We check if the fact for status 'Y' is true. If not, we infer the status is 'N'.
            y_fact_id = self.on_fact_ids.get((x, y, 'Y'))
            
            if y_fact_id is None:
                # This case should not be reached if the problem is well-formed.
                continue

            is_currently_y = (state_bitwise >> y_fact_id) & 1
            current_status = 'Y' if is_currently_y else 'N'

            # If the current status doesn't match the goal, increment the count.
            if current_status != goal_status:
                mismatched_count += 1
        
        return mismatched_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every expanded node during search. Returns the heuristic value.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return float(h_value)
