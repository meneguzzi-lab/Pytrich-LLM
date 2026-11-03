import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class LampsGemini25ProHeuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Lamps (Lights Out) domain.

    This heuristic is admissible and calculates the "Manhattan distance" to the goal.

    - **Domain Analysis & Bottleneck:** The Lamps domain is a "Lights Out" style puzzle.
      A single player action ('turn') involves flipping one lamp, which then triggers a
      state-dependent cascade of flips along its row and column. The cost of a plan is the
      total number of primitive 'flip' actions. The bottleneck is the complex interaction
      between flips, making it hard to predict the full effect of a single turn.

    - **Heuristic Strategy:** The chosen heuristic is `h(n) = number of mismatched lamps`.
      This counts how many lamps on the grid are currently in a state different from
      their specified goal state.

    - **Admissibility Justification:**
      Each primitive `flip` action changes the state of exactly one lamp.
      - If a mismatched lamp is flipped, the number of mismatches decreases by 1.
      - If a correctly placed lamp is flipped, the number of mismatches increases by 1.
      Therefore, a single `flip` action can reduce the number of mismatched lamps by at most 1.
      To fix `N` mismatched lamps, we need at least `N` `flip` actions where each action
      corrects a mismatch. Thus, the number of mismatched lamps is a lower bound on the
      minimum number of `flip` actions required to reach the goal, making the heuristic admissible.
    """

    def __init__(self, lamps_gemini25pro_heuristic_5=None, name="LampsGemini25ProHeuristic5"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0
        # Preprocessed data structures
        self.on_facts = {}      # (x, y) -> {'Y': fact_obj, 'N': fact_obj}
        self.goal_config = {}   # (x, y) -> 'Y' or 'N'
        self.coords = set()     # Set of all (x, y) coordinates

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocesses the domain model to build fast lookup tables for the heuristic.
        This is called once before the search begins.
        """
        start_time = time.time()
        self.model = model

        # 1. Parse all facts to identify lamp coordinates and their state facts.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == 'on' and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord = (x, y)
                self.coords.add(coord)
                if coord not in self.on_facts:
                    self.on_facts[coord] = {}
                self.on_facts[coord][status] = fact

        # 2. Parse the goal state to create a target configuration map.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            predicate, args = self._parse_fact_name(name)
            if predicate == 'on' and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                coord = (x, y)
                self.goal_config[coord] = status
        
        self.preprocessing_time = time.time() - start_time

        # Compute and return the initial h-value
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into a tuple.
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

    def _compute(self, state_bitwise: int) -> int:
        """
        Calculates the number of lamps that are not in their goal state.
        """
        mismatched_count = 0
        for coord in self.coords:
            goal_status = self.goal_config.get(coord)
            if goal_status is None:
                # This lamp is not constrained by the goal definition, so we ignore it.
                continue

            # Get the fact object corresponding to the state this lamp *should* be in.
            fact_at_goal = self.on_facts[coord].get(goal_status)
            if fact_at_goal is None:
                # This case should not be reached with valid PDDL files.
                continue

            # Check if this fact is true in the current state using a fast bitwise operation.
            is_in_goal_state = (state_bitwise >> fact_at_goal.global_id) & 1

            if not is_in_goal_state:
                mismatched_count += 1
        
        return mismatched_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This is called for every expanded node, so it must be fast.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return float(h_value)

    def __repr__(self):
        return f"{self.name}(coords={len(self.coords)})"

    def __output__(self):
        return (
            f"{self.name} Info:\n"
            f"\tName: {self.name}\n"
            f"\tTotal Lamps (coords): {len(self.coords)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
        )
