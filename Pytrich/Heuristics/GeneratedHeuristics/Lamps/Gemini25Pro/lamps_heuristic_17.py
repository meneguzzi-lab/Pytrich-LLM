from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

class LampsGemini25ProHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    This heuristic calculates the number of lamps that are not in their goal state.
    This is an admissible lower bound on the number of primitive `flip` actions required.

    Domain Analysis & Bottleneck:
    - The core of the Lamps domain is to reach a target configuration of lamp statuses (Y/N).
    - The only state-changing action is `flip`, which toggles a single lamp.
    - A `turn` task initiates a `flip` which can propagate to neighbors, causing a cascade of flips.
    - The total plan cost is the number of primitive actions executed, primarily `flip`s.
    - The main bottleneck is determining the sequence of initial `turn`s whose cascading effects will result in the goal configuration. The cost is the total number of lamps flipped.

    Heuristic Design (h_mismatch):
    - This heuristic implements a single, effective lower bound:
        1. h_mismatch: The number of lamps whose current status does not match their goal status.

    - Admissibility Sketch:
        - Each lamp in an incorrect state must be flipped at least once to reach the goal.
        - A single primitive `flip` action can only change the state of one lamp.
        - Therefore, the total number of `flip` actions required is at least the number of mismatched lamps.
        - This is equivalent to the Hamming distance between the current state and the goal state and is an admissible heuristic.
    """

    def __init__(self, lamps_gemini25pro_heuristic_17=None, name="LampsGemini25ProHeuristic17"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Preprocessed data
        self.goal_state_map = {}  # (x, y) -> goal_status ('Y' or 'N')
        self.on_fact_ids = {}     # (x, y, status) -> fact global_id
        self.coords = []          # List of all (x, y) coordinate tuples

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model once before search.
        This involves parsing all facts to build maps for fast lookups during search.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles grounded format: "+predicate[arg1,arg2,...]"
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

    def _preprocess(self):
        """
        Extracts domain structure from the model's facts and goals.
        Populates data structures for fast lookups in __call__.
        """
        # 1. Parse goal state to build a map of (x,y) -> goal_status
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args
                self.goal_state_map[(x, y)] = status

        # 2. Parse all facts to map (x, y, status) to fact IDs and collect all coordinates
        coords_set = set()
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args
                self.on_fact_ids[(x, y, status)] = fact.global_id
                coords_set.add((x, y))
        self.coords = sorted(list(coords_set))

    def _compute(self, state_bitwise: int, task_network) -> float:
        """
        Calculate the heuristic value: the number of misplaced lamps.
        """
        mismatch_count = 0
        for x, y in self.coords:
            goal_status = self.goal_state_map.get((x, y))

            # If a coordinate is not in the goal, we don't count it.
            if goal_status is None:
                continue

            # Determine the current status of the lamp at (x, y)
            # We check the bit for the 'on(x,y,Y)' fact.
            # If it's true, status is 'Y', otherwise it's 'N'.
            id_y_status = self.on_fact_ids.get((x, y, 'Y'))
            current_status = 'N'  # Assume 'N' if 'Y' fact doesn't exist or is false
            if id_y_status is not None:
                is_y = (state_bitwise >> id_y_status) & 1
                if is_y:
                    current_status = 'Y'

            if current_status != goal_status:
                mismatch_count += 1

        return float(mismatch_count)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate the heuristic value for a given search node.
        This is called for every node expansion and must be fast.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"LampsGemini25ProHeuristic17(coords={len(self.coords)})"

    def __output__(self):
        return (
            f"Lamps Gemini 25 Pro Heuristic 17 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Coordinates: {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
        )
