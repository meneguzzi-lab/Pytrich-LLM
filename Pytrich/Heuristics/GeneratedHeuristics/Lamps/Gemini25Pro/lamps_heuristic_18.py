import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic is designed for the Lamps domain, which models a grid of lamps that
can be flipped, causing a chain reaction of flips along rows and columns.

**1. Bottleneck Analysis**
The primary bottleneck is the cascading 'flip' propagation. A single player
action ('turn') initiates a flip at one coordinate, which then propagates to
all adjacent lamps of the same color, which in turn also flip. This makes it
difficult to make localized changes. The cost of a plan is the total number
of primitive 'flip' actions, and a single 'turn' can have a large and
unpredictable cost.

**2. Heuristic Design: Mismatch Count**
The heuristic implemented is an admissible lower bound based on a delete-relaxation.
It calculates the number of lamps whose current status ('Y' or 'N') does not
match the goal status.

-   Heuristic Term: h(n) = count of all (x, y) where current_status(x, y) != goal_status(x, y)

**3. Admissibility Justification**
To change the status of a mismatched lamp at coordinate (x, y), the lamp must be
subject to a `flip` action at some point. Each primitive `flip` action costs 1.
Therefore, at least one `flip` action is required for each lamp that is currently
in the wrong state. The total number of primitive actions must be at least the
number of mismatched lamps. This heuristic ignores that a single `turn`
can cause multiple `flip`s (potentially fixing multiple mismatches) and can also
change correctly set lamps to incorrect states (negative effects). By ignoring
all effects except the one flip needed to fix one mismatch, we get an admissible
lower bound on the total number of primitive `flip` actions.
"""


class LampsGemini25ProHeuristic18(Heuristic):
    """
    An admissible heuristic for the Lamps domain.

    It computes the number of lamps that are not in their goal state.
    This is a lower bound on the number of primitive 'flip' actions required,
    as each mismatched lamp must be flipped at least once.
    """

    def __init__(self, lamps_gemini25pro_heuristic_18=None, name="LampsGemini25ProHeuristic18"):
        super().__init__(name=name)
        self.preprocessing_time = 0.0

        # Preprocessed data
        self.coords = []           # List of (x, y) tuples
        self.goal_map = {}         # (x, y) -> goal_status ('Y' or 'N')
        self.on_fact_map = {}      # (x, y, status) -> fact object

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Preprocess the domain to build maps of goals and facts.
        This is called once before search begins.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses a grounded fact name like '+predicate[arg1,arg2]'
        into a tuple (predicate, [arg1, arg2]).
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
        Extracts goal states and all possible 'on' facts from the model.
        """
        coords_set = set()

        # 1. Parse goal state from model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            predicate, args = self._parse_fact_name(name)
            if predicate == 'on' and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.goal_map[(x, y)] = status
                coords_set.add((x, y))

        # 2. Parse all facts to map coordinates and statuses to fact objects
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == 'on' and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.on_fact_map[(x, y, status)] = fact
                coords_set.add((x, y))

        # Store coordinates in a sorted list for deterministic iteration
        self.coords = sorted(list(coords_set))

    def _compute(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value: the number of mismatched lamps.
        """
        mismatches = 0
        for x, y in self.coords:
            goal_status = self.goal_map.get((x, y))

            # If a coordinate is not in the goal, we don't care about its state.
            if goal_status is None:
                continue

            # Determine the current status of the lamp at (x, y)
            current_status = None
            y_fact = self.on_fact_map.get((x, y, 'Y'))
            if y_fact and (state_bitwise >> y_fact.global_id) & 1:
                current_status = 'Y'
            else:
                # If it's not 'Y', it must be 'N' (assuming a complete state)
                n_fact = self.on_fact_map.get((x, y, 'N'))
                if n_fact and (state_bitwise >> n_fact.global_id) & 1:
                    current_status = 'N'

            if current_status is not None and current_status != goal_status:
                mismatches += 1

        return mismatches

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculate heuristic value for a search node.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return float(h_value)
