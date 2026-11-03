from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time


class LampsGemini25ProHeuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    Heuristic Strategy:
    This heuristic calculates the number of lamps whose current state (Y or N)
    does not match their required state in the goal. This is equivalent to the
    Hamming distance between the current state vector and the goal state vector.

    Bottleneck Analysis:
    The primary bottleneck in the Lamps domain is the complex, non-local, and
    state-dependent chain reaction caused by a single `flip` action. A `turn`,
    which is the main choice for the planner, can trigger a cascade of flips,
    making the outcome of an action hard to predict and control. This creates a
    vast and complex search space.

    Admissibility Sketch:
    The heuristic `h(n)` is the count of mismatched lamps. Let this count be `k`.
    An optimal plan must change the state of each of these `k` lamps. To change
    the state of a lamp, it must be flipped an odd number of times. Any lamp
    that is already in its correct state must be flipped an even number of times
    (or not at all).

    Let `F` be the total number of `flip` actions in an optimal plan (the cost `h*(n)`).
    Let `M` be the set of `k` mismatched lamps.
    - For each lamp `c` in `M`, it must be flipped `2*i + 1` times (where `i >= 0`).
    - For each lamp `c` not in `M`, it must be flipped `2*j` times (where `j >= 0`).

    The total number of flips is the sum over all lamps:
    `F = sum_{c in M} (2*i_c + 1) + sum_{c not in M} (2*j_c)`
    `F = k * 1 + 2 * (sum(i_c) + sum(j_c))`

    Since `sum(i_c) >= 0` and `sum(j_c) >= 0`, it follows that `F >= k`.
    Therefore, the number of mismatched lamps `k` is a lower bound on the total
    number of required `flip` actions, making the heuristic admissible.
    """

    def __init__(self, lamps_gemini25pro_heuristic_0=None, name="LampsGemini25ProHeuristic0"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed data structures
        self.goal_positions = {}  # (x, y) -> 'Y' or 'N'
        self.on_facts_map = {}    # (x, y) -> {'Y': fact_id_Y, 'N': fact_id_N}
        self.coords = set()       # set of all (x, y) coordinates

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before the search starts. All expensive preprocessing, such as
        parsing facts and building data structures, should be done here.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Compute the heuristic value for the initial state
        h_value = self._compute(initial_node.state, initial_node.task_network)

        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Called for every node expansion. This method must be very fast.
        It uses the preprocessed data to calculate the heuristic value.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into a tuple.
        Returns: (predicate, [arg1, arg2, ...]) or (None, []) if parsing fails.
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
        Analyzes the model's facts and goals to build fast-access lookup tables.
        This method populates `self.goal_positions`, `self.on_facts_map`, and `self.coords`.
        """
        # 1. Process goal state to find the desired status for each lamp
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    coord = (x, y)
                    self.goal_positions[coord] = status

        # 2. Process all facts to map coordinates to 'on' fact IDs
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord = (x, y)

                if coord not in self.on_facts_map:
                    self.on_facts_map[coord] = {}
                    self.coords.add(coord)
                
                self.on_facts_map[coord][status] = fact.global_id

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Calculates the number of lamps that are not in their goal state.
        """
        mismatch_count = 0
        # If there are no goals, the problem is trivially solved.
        if not self.goal_positions:
            return 0

        for coord in self.coords:
            goal_status = self.goal_positions.get(coord)
            if goal_status is None:
                # This coordinate is not mentioned in the goal, so we don't care about its state.
                continue

            fact_ids = self.on_facts_map.get(coord)
            if not fact_ids:
                # Should not happen in a valid problem
                continue
            
            # Determine the current status using a fast bitwise check
            is_on_y = (state_bitwise >> fact_ids['Y']) & 1
            current_status = 'Y' if is_on_y else 'N'

            # Compare with the goal status
            if current_status != goal_status:
                mismatch_count += 1

        return mismatch_count

    def __output__(self):
        return (
            f"{self.name} Info:\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
