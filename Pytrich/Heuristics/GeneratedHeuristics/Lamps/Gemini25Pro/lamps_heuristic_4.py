from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict
import time

class LampsGemini25ProHeuristic4(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    This heuristic uses a simple, admissible goal-count approach to estimate the
    remaining cost to solve the puzzle.

    1. Bottleneck Analysis:
    The primary difficulty in the Lamps domain is the cascading, non-local effect of a
    single 'turn'. Choosing to flip one lamp can trigger a chain reaction that flips
    many others, making it hard to reason about the direct consequences of an action.
    The core measure of progress is the number of lamps in their correct goal state.

    2. Admissible Lower Bound:
    The heuristic calculates the number of lamps that are not in their goal state. This is
    called a "goal-count" heuristic, and it is admissible.
    - Admissibility Sketch: Let 'k' be the number of lamps in an incorrect state.
      Each of these 'k' lamps must have its state changed at least once to reach the
      goal. The primitive action `flip` changes the state of exactly one lamp.
      Therefore, at least 'k' `flip` actions are required in total to fix all
      mismatched lamps. This provides a lower bound on the total plan cost.
    """

    def __init__(self, lamps_gemini25pro_heuristic_4=None, name="LampsGemini25ProHeuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed data
        self.coords = set()                 # Set of (x, y) tuples for all lamps
        self.on_facts = defaultdict(dict)   # Maps (x,y) -> {'Y': fact, 'N': fact}
        self.goal_config = {}               # Maps (x,y) -> 'Y' or 'N' representing the goal state

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search begins. Preprocesses the domain to build
        fast lookup tables for use in the heuristic calculation.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Compute heuristic for the initial state
        h_value = self._compute(initial_node.state, initial_node.task_network)

        # Pass the initial h-value to the base class
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This method is called for every expanded node and must be fast.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _parse_fact_name(self, name: str):
        """
        Parses a fact name into a (predicate, [args...]) tuple.
        Handles formats like `+predicate[arg1,arg2]`.
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
        Extracts static information from the model, such as goal states and
        fact mappings for all lamp coordinates.
        """
        # 1. Parse goal configuration from model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    self.goal_config[(x, y)] = status

        # 2. Parse all facts to map coordinates to 'on' fact objects
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord = (x, y)
                self.coords.add(coord)
                self.on_facts[coord][status] = fact

    def _compute(self, state_bitwise: int, task_network) -> int:
        """
        Computes the heuristic value by counting mismatched lamps.
        """
        mismatches = 0
        if not self.coords:
            return 0

        for coord in self.coords:
            goal_status = self.goal_config.get(coord)
            if goal_status is None:
                # This case should not happen in a valid problem
                continue

            # Get the fact object corresponding to the GOAL state for this lamp
            fact_for_goal_status = self.on_facts[coord].get(goal_status)
            if fact_for_goal_status is None:
                continue

            # Check if the goal fact is true in the current state using a bitwise AND.
            # If (state >> id) & 1 is 0, the fact is false, meaning the lamp is not
            # in its goal state.
            is_in_goal_state = (state_bitwise >> fact_for_goal_status.global_id) & 1

            if not is_in_goal_state:
                mismatches += 1

        return mismatches

    def __repr__(self):
        return f"LampsGemini25ProHeuristic4(lamps={len(self.coords)})"

    def __str__(self):
        return f"LampsGemini25ProHeuristic4 with {len(self.coords)} lamps"

    def __output__(self):
        return (
            f"Lamps Gemini25Pro Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tGrid Size (num lamps): {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
