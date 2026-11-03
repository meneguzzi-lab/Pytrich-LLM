from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsGemini25ProHeuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Lamps (Lights Out variant) domain.

    Heuristic Design:
    This heuristic is based on a single, strong admissible term that counts the number of misplaced lamps.

    1.  **Bottleneck Analysis:** The primary challenge in the Lamps domain is the non-local, state-dependent
        chain reaction caused by a single `turn`. Flipping one lamp triggers a cascade of flips along
        its row and column, making the outcome of an action complex and hard to predict. A single `turn`
        can both fix and break multiple lamp states simultaneously.

    2.  **Admissible Lower Bound:** The heuristic value `h` is the number of lamps currently in a state
        that differs from their goal state (`h = M`).

    3.  **Admissibility Justification:** This heuristic is an admissible lower bound on the total number of
        primitive `flip` actions required to reach the goal.
        - **Proof Sketch:** Let the states be represented as bitvectors. The set of misplaced lamps corresponds to
          the bitwise difference `diff = current_state XOR goal_state`, and `M = popcount(diff)`.
        - A plan is a sequence of `turn`s, which apply state-change masks `T_1, T_2, ..., T_k`.
        - To reach the goal, the cumulative effect must be `diff`, i.e., `T_1 XOR T_2 XOR ... XOR T_k = diff`.
        - The total plan cost is the sum of `flip`s in each turn: `Cost = sum(popcount(T_i))`.
        - By the triangle inequality for popcount (`popcount(A XOR B) <= popcount(A) + popcount(B)`),
          we can show that `popcount(XOR(T_i)) <= sum(popcount(T_i))`.
        - Therefore, `popcount(diff) <= Cost`, which simplifies to `M <= Cost`.
        - Thus, the number of misplaced lamps is a lower bound on the number of flips required.
    """

    def __init__(self, lamps_gemini25pro_heuristic_15=None, name="LampsGemini25ProHeuristic15"):
        super().__init__(name=name)
        self.goal_state = {}      # (x, y) -> status ('Y' or 'N')
        self.on_fact_ids = {}     # (x, y, status) -> global_id of the 'on' fact
        self.coords = set()       # Set of all (x, y) coordinates in the grid

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Preprocesses the model to build data structures
        for efficient heuristic calculation.
        """
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node. This must be fast.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name into its predicate and arguments.
        Format: `+predicate[arg1,arg2]`
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
        Extracts goal state information and builds maps for all possible 'on' facts
        and their corresponding global IDs for quick lookups.
        """
        # 1. Extract the goal configuration from the model's goal specification.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            predicate, args = self._parse_fact_name(name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args
                self.goal_state[(x, y)] = s

        # 2. Iterate through all facts to map coordinates and states to fact IDs.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, s = args
                self.coords.add((x, y))
                self.on_fact_ids[(x, y, s)] = fact.global_id

    def _compute(self, state_bitwise: int) -> int:
        """
        Computes the number of misplaced lamps by comparing the current state
        to the preprocessed goal state for every coordinate on the grid.
        """
        misplaced_count = 0
        for coord in self.coords:
            goal_status = self.goal_state.get(coord)

            # If a coordinate is not in the goal, we cannot assess it.
            if not goal_status:
                continue

            # Find the global_id of the fact that *should* be true if the lamp is correct.
            goal_fact_id = self.on_fact_ids.get((*coord, goal_status))
            if goal_fact_id is None:
                continue

            # Check if the goal fact is true in the current state using a fast bitwise check.
            # If it's NOT true, the lamp is in the wrong state.
            is_correct = (state_bitwise >> goal_fact_id) & 1
            if not is_correct:
                misplaced_count += 1

        return misplaced_count
