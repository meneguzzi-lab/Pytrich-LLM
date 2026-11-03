from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class LampsGemini25ProHeuristic7(Heuristic):
    """
    Domain-dependent heuristic for the Lamps domain.

    This heuristic calculates the number of lamps that are not in their target goal state.

    **Domain Analysis & Bottleneck**
    The Lamps domain models a "Lights Out" style puzzle where flipping a lamp triggers a
    cascading flip of adjacent lamps with the same initial state. The key bottleneck
    is this **interdependence of lamp states due to the propagation mechanism.** A single
    action can have widespread, hard-to-predict side effects, making it difficult to
    solve local goals (fix one lamp) without disturbing the rest of the board.

    **Heuristic Design: Admissible Goal Count**
    The heuristic is a simple, admissible goal-count:
    h(n) = number of lamps (x, y) where current_status(x, y) != goal_status(x, y)

    **Admissibility Sketch:**
    This heuristic is admissible because it represents a relaxation of the problem.
    It assumes each mismatched lamp can be fixed with a single action of cost 1, without
    any negative side effects on other lamps. In the real problem, the minimum cost to
    perform any change is one 'turn' action, which has a cost of at least 1 (for the initial flip).
    Therefore, the number of mismatched lamps is a lower bound on the number of actions
    required to reach the goal.

    **Implementation Details:**
    - **P3 Preprocessing:** The `initialize` method preprocesses all static information.
      It builds maps from coordinates to goal states and from coordinates/states to fact IDs.
    - **P2 State-Aware & Fast `__call__`: ** The `__call__` method iterates through all known coordinates,
      using the precomputed maps and fast bitwise operations to check the current state of each
      lamp against its goal state. The complexity is proportional to the number of lamps on the grid,
      not the number of facts or operators in the model.
    """

    def __init__(self, lamps_gemini25pro_heuristic_7=None, name="LampsGemini25ProHeuristic7"):
        super().__init__(name=name)
        # Map from (x, y) coordinate to the goal status ('Y' or 'N')
        self.goal_state_map = {}

        # Map from (x, y) to a dict of statuses and their fact IDs
        # e.g., {(x, y): {'Y': fact_id_Y, 'N': fact_id_N}}
        self.on_fact_map = {}

        # A sorted list of all unique (x, y) coordinates for deterministic iteration
        self.coords = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search. Preprocesses the domain to build maps for fast lookups.
        """
        self.model = model
        self._preprocess()
        h_value = self._compute(initial_node.state)
        return super().initialize(model, h_value)

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

    def _preprocess(self):
        """
        Builds internal data structures for fast heuristic calculation.
        - Populates `self.on_fact_map` with fact IDs for each lamp state.
        - Populates `self.coords` with all lamp coordinates.
        - Populates `self.goal_state_map` with the target state for each lamp.
        """
        all_coords = set()
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                all_coords.add((x, y))
                if (x, y) not in self.on_fact_map:
                    self.on_fact_map[(x, y)] = {}
                self.on_fact_map[(x, y)][status] = fact.global_id
        self.coords = sorted(list(all_coords))

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for goal_name in goal_fact_names:
            predicate, args = self._parse_fact_name(goal_name)
            if predicate == "on" and len(args) == 3:
                x, y, status = args[0], args[1], args[2]
                self.goal_state_map[(x, y)] = status

    def _compute(self, state_bitwise: int) -> int:
        """
        Calculates the number of lamps whose current state mismatches the goal state.
        """
        mismatched_lamps = 0
        for x, y in self.coords:
            goal_status = self.goal_state_map.get((x, y))
            if goal_status is None:
                continue

            # Determine the current status using fast bitwise checks
            fact_id_y = self.on_fact_map.get((x, y), {}).get('Y')

            # If there's no fact for (on x y Y), we assume the status is 'N',
            # as Y and N are the only two statuses.
            if fact_id_y is None:
                current_status_is_y = False
            else:
                current_status_is_y = (state_bitwise >> fact_id_y) & 1

            current_status = 'Y' if current_status_is_y else 'N'

            if current_status != goal_status:
                mismatched_lamps += 1

        return mismatched_lamps

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This method must be very fast.
        """
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return float(h_value)
