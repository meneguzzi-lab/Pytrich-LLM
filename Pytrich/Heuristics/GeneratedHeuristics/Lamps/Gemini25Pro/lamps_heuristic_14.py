from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict
import time


class LampsGemini25ProHeuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Lamps (Lights Out) domain.

    - **Domain Bottleneck:** The primary bottleneck is the complex, cascading effect of a single `flip` action. A single `turn` (initiated by a `flip`) can alter the state of many lamps through propagation, making it difficult to predict the outcome of an action and reach a specific goal configuration.

    - **Heuristic Strategy:** The heuristic is a single-component admissible lower bound based on the state difference to the goal.

    - **Admissible Lower Bound (h_misplaced):**
        - The heuristic value is the number of lamps whose current status ('Y' or 'N') does not match their goal status.
        - **Admissibility Sketch:** To reach the goal, every lamp that is currently in an incorrect state must have its state changed at least once. The only primitive action that can change a lamp's state is `flip`, which has a cost of 1. Therefore, the total number of required `flip` actions must be at least the number of incorrectly set lamps. This holds even with propagation, as a propagated flip is still a `flip` action with a cost of 1. h(n) is the number of misplaced lamps, and h*(n) is the minimum number of flips to solve the problem. Since each misplaced lamp must be flipped, h*(n) >= h(n). Thus, the heuristic is admissible.
    """

    def __init__(self, lamps_gemini25pro_heuristic_14=None, name="LampsGemini25ProHeuristic14"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed data structures
        self.goal_states = {}  # (x, y) -> status ('Y' or 'N')
        self.on_facts = defaultdict(dict)  # (x, y) -> {status: Fact}
        self.coords = []  # List of all (x, y) tuples

    def _parse_fact_name(self, name: str):
        """
        Parse a grounded fact name like '+on[l0,l1,Y]' into ('on', ['l0', 'l1', 'Y']).
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

    def _preprocess(self):
        """
        Extracts domain structure from the model facts and goals.
        This method is called once during initialization.
        """
        # 1. Parse goal state to build the self.goal_states map
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 3:
                    x, y, status = args[0], args[1], args[2]
                    self.goal_states[(x, y)] = status

        # 2. Parse all model facts to build the self.on_facts map
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_facts[(x, y)][status] = fact

        # 3. Create a list of all unique coordinates
        self.coords = sorted(list(self.on_facts.keys()))

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic, preprocess the model, and calculate the initial h-value.
        """
        start_time = time.time()
        self.model = model

        # Perform all expensive preprocessing
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate heuristic for the initial state
        h_value = self._calculate_heuristic(initial_node.state)

        # Call parent initializer with the calculated h-value
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value for a given state by counting misplaced lamps.
        This must be very fast.
        """
        misplaced_count = 0
        if not self.coords:
            return 0

        for x, y in self.coords:
            goal_status = self.goal_states.get((x, y))
            if goal_status is None:
                continue  # This coordinate is not part of the goal state

            # Determine the current status of the lamp at (x, y)
            current_status = None
            facts_for_coord = self.on_facts.get((x, y), {})
            y_fact = facts_for_coord.get('Y')
            n_fact = facts_for_coord.get('N')

            # Efficiently check which status fact is true using bitwise operations
            if y_fact and (state_bitwise >> y_fact.global_id) & 1:
                current_status = 'Y'
            elif n_fact and (state_bitwise >> n_fact.global_id) & 1:
                current_status = 'N'
            
            # Compare current status to goal status
            if current_status is not None and current_status != goal_status:
                misplaced_count += 1
        
        return misplaced_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate the heuristic value for a node during search.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return float(h_value)

    def __output__(self):
        """
        Returns a string with statistics about the heuristic's performance.
        """
        return (
            f"Heuristic Info for {self.name}:\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tGoal Lamps: {len(self.goal_states)}\n"
            f"\tTotal Coordinates: {len(self.coords)}\n"
            f"\tInitial H-Value: {self.initial_h}\n"
            f"\tTotal Calls: {self.calls}\n"
            f"\tMin H-Value: {self.min_hvalue}\n"
            f"\tAverage H-Value: {self.total_hvalue / self.calls if self.calls > 0 else 0:.2f}\n"
        )
