from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsGemini25ProHeuristic12(Heuristic):
    """
    Heuristic for the Lamps (Lights Out) domain.

    Domain Analysis:
    - The domain models a "Lights Out" style puzzle on a grid.
    - The only primitive action is `flip`, which changes a lamp's state.
    - The `turn` task initiates a `flip` at a chosen coordinate, which then propagates
      to adjacent lamps of the same original state, flipping them as well.
    - Cost is the total number of `flip` actions.

    Heuristic Design:
    1.  Bottleneck: The cascading effect of a single `turn` makes it difficult to predict
        the outcome and fix individual lamps without affecting others. The cost of a plan
        is the sum of all `flip` actions, including propagated ones.

    2.  Admissible Lower Bound:
        - h(n) = Number of lamps in an incorrect state.
        - Admissibility Sketch: Let C be the number of lamps whose current state differs
          from their goal state. Each of these C lamps must have its state changed at
          least once. A `flip` action is what changes a lamp's state. The best-case
          scenario for a plan is that each `flip` action contributes to fixing one
          incorrect lamp without breaking any correct ones. Therefore, at least C
          `flip` actions are required to reach the goal. This is a relaxed model
          that ignores the interdependent effects of propagation, ensuring it remains a lower bound.
    """

    def __init__(self, lamps_gemini25pro_heuristic_12=None, name="LampsGemini25ProHeuristic12"):
        super().__init__(name=name)
        # Preprocessed data
        # self.goal_config: maps (x, y) coordinates to their goal status ('Y' or 'N')
        self.goal_config = {}
        # self.coord_facts: maps (x, y) coordinates to a dict of status -> fact object
        self.coord_facts = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search begins. Preprocesses the domain to build fast lookup tables.
        """
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]' into (predicate, [args]).
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
        Extracts goal states and creates a mapping of coordinates to their 'on' facts.
        This is done once to make the heuristic calculation in __call__ very fast.
        """
        # 1. Parse the goal state from the model's goal bitmask.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.goal_config[(x, y)] = status

        # 2. Map all possible 'on' facts for every coordinate for efficient state checking.
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                if (x, y) not in self.coord_facts:
                    self.coord_facts[(x, y)] = {}
                self.coord_facts[(x, y)][status] = fact

    def _compute(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value: the number of lamps in an incorrect state.
        """
        if not self.goal_config:
            return 0

        wrong_lamps_count = 0
        for coords, goal_status in self.goal_config.items():
            # This check ensures we only evaluate lamps specified in the goal
            if coords not in self.coord_facts:
                continue

            fact_y = self.coord_facts[coords].get('Y')
            fact_n = self.coord_facts[coords].get('N')

            current_status = None
            # Use fast bitwise check to determine the current status of the lamp
            if fact_y and (state_bitwise >> fact_y.global_id) & 1:
                current_status = 'Y'
            elif fact_n and (state_bitwise >> fact_n.global_id) & 1:
                current_status = 'N'

            # If the lamp's current state doesn't match its goal state, increment the count.
            if current_status is not None and current_status != goal_status:
                wrong_lamps_count += 1

        return wrong_lamps_count

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This method is called for every node expansion and must be fast.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return float(h_value)
