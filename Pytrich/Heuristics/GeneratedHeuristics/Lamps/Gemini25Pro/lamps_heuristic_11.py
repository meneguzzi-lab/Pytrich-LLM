import time
from collections import defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsGemini25ProHeuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    This heuristic calculates the number of lamps that are not in their goal state.

    - **Bottleneck:** The core challenge is finding a sequence of 'turn' operations
      (which trigger cascading flips) to transform the initial grid of lamps into the
      goal configuration. The cost is the total number of primitive 'flip' actions.

    - **Admissibility Sketch:**
      The heuristic counts the number of lamps whose current status (Y/N) does not
      match their target status in the goal. Let this count be 'k'.
      The only primitive action that can change a lamp's status is 'flip'. Each
      'flip' action changes the status of exactly one lamp. Therefore, to fix 'k'
      incorrectly set lamps, a minimum of 'k' 'flip' actions are required.
      Since the cost of a plan is the number of primitive actions, and each 'flip'
      costs 1, this heuristic is an admissible lower bound on the remaining cost.
    """

    def __init__(self, lamps_gemini25pro_heuristic_11=None, name="LampsGemini25ProHeuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Precomputed data structures
        self.coords = set()            # Set of (x, y) coordinate tuples
        self.goal_config = {}          # Maps (x, y) -> goal_status
        self.on_facts_map = defaultdict(dict) # Maps (x, y) -> {status -> fact_object}

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search starts. Preprocesses the domain to build
        fast-access data structures for the heuristic calculation.
        """
        start_time = time.time()
        self.model = model

        self._preprocess_domain()

        self.preprocessing_time = time.time() - start_time

        # Calculate and return the initial heuristic value
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This is called for every node expansion and must be very fast.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+on[l0,l0,Y]' into a predicate and arguments.
        Returns (predicate, [arg1, arg2, ...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _preprocess_domain(self):
        """
        Extracts information about the grid and goal state from the model.
        - Populates `self.coords` with all (x, y) coordinates.
        - Populates `self.on_facts_map` for quick lookup of fact IDs.
        - Populates `self.goal_config` with the target status for each coordinate.
        """
        # 1. Process all facts to map out the grid and all possible 'on' states
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord_tuple = (x, y)
                self.coords.add(coord_tuple)
                self.on_facts_map[coord_tuple][status] = fact

        # 2. Process the goal state to find the target configuration
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    self.goal_config[(x, y)] = status

    def _compute(self, state_bitwise: int) -> int:
        """
        Computes the heuristic value: the number of misplaced lamps.
        """
        if not self.goal_config:
            return 0

        mismatch_count = 0
        # Iterate over all known coordinates from preprocessing
        for coord in self.coords:
            goal_status = self.goal_config.get(coord)
            if goal_status is None:
                # This coordinate is not mentioned in the goal, so we don't
                # care about its state.
                continue

            # Get the fact object corresponding to the goal state for this lamp
            goal_fact = self.on_facts_map[coord].get(goal_status)

            if goal_fact is None:
                continue

            # Check if this goal fact is true in the current state
            is_in_goal_state = (state_bitwise >> goal_fact.global_id) & 1
            
            if not is_in_goal_state:
                mismatch_count += 1
        
        return mismatch_count

    def __repr__(self):
        return f"LampsGemini25ProHeuristic11(grid_size={len(self.coords)})"

    def __str__(self):
        return f"Lamps Heuristic (Mismatched Count) for a {len(self.coords)}-lamp grid."

    def __output__(self):
        return (
            f"Heuristic '{self.name}' Info:\n"
            f"\tType: Mismatched Lamps Count (Admissible)\n"
            f"\tGrid Size (num coords): {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
            f"\tMin H-Value Encountered: {self.min_hvalue}\n"
        )
