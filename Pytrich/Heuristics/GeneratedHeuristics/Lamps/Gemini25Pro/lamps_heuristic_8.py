from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time

"""
Domain-dependent heuristic for the Lamps domain.

This heuristic is designed for the "Lamps" (or "Lights Out") style of problem where flipping one lamp can trigger a cascade of flips on adjacent lamps.

**Bottleneck Analysis:**
The primary bottleneck is the non-local, cascading effect of a single `turn`. A single chosen `flip` can cause a chain reaction, flipping many other lamps. This makes it difficult to predict the full consequences of an action and can lead to undoing previous progress. The search space is large because any lamp can be chosen for a `turn` at any step.

**Heuristic Design (P1: Multi-component Max):**
This heuristic returns the maximum of two independent, admissible lower bounds on the remaining cost (number of primitive `flip` actions).

1.  **State-based term (h_state):** Counts the number of lamps whose current state ('Y' or 'N') does not match their goal state.
    *   **Admissibility:** Each mismatched lamp must be flipped at least once to reach its goal state. A primitive `flip` action changes the state of exactly one lamp. Therefore, the minimum number of `flip` actions required is at least the number of mismatched lamps.

2.  **Task-network-based term (h_tn):** Estimates the cost of the current plan prefix stored in the task network.
    *   For each `(turn)` task, it adds a cost of 1, representing the mandatory initial `flip` action.
    *   For each `(propagate_*)` task, it calculates the *exact* number of `flip` actions that will be triggered by simulating the propagation chain on the current state.
    *   **Admissibility:** This term sums the cost of actions that are guaranteed to occur if the current partial plan is pursued. It is a precise count of a subset of future actions, making it an admissible lower bound.

The final heuristic `h(n) = max(h_state, h_tn)` is admissible because both of its components are admissible. It combines a goal-oriented view (state distance) with a plan-oriented view (cost of current tasks) to provide a more informed estimate.
"""


class LampsGemini25ProHeuristic8(Heuristic):
    """
    A max-based heuristic for the Lamps domain combining state-distance and task-cost estimates.
    """

    def __init__(self, lamps_gemini25pro_heuristic_8=None, name="LampsGemini25ProHeuristic8"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Preprocessed data structures
        self.goal_on_status = {}  # (x, y) -> status_str ('Y' or 'N')
        self.on_facts = {}        # (x, y, status) -> fact_obj
        self.next_map = {}        # coord_str -> coord_str
        self.prev_map = {}        # coord_str -> coord_str
        self.all_coords = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Called once before search starts. Preprocesses the domain to build fast lookup tables.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _parse_name(self, name: str):
        """
        Parses a fact or task name into (predicate, args) tuple.
        Handles both "+predicate[arg1,arg2]" and "(predicate arg1 arg2)" formats.
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
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """
        Extracts static information from the model's facts and goals for fast lookups.
        """
        # 1. Parse all static and dynamic facts from the model
        for fact in self.model.facts:
            pred, args = self._parse_name(fact.name)
            if pred == "on" and len(args) == 3:
                x, y, s = args
                self.on_facts[(x, y, s)] = fact
                self.all_coords.add((x, y))
            elif pred == "next" and len(args) == 2:
                self.next_map[args[0]] = args[1]
                self.prev_map[args[1]] = args[0]

        # 2. Parse the goal state to create a goal map
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_name(name)
            if pred == "on" and len(args) >= 3:
                x, y, s = args[0], args[1], args[2]
                self.goal_on_status[(x, y)] = s

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node.
        This must be fast as it is called for every expanded node.
        """
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state_bitwise: int, task_network: list) -> float:
        """
        Computes the final heuristic value as max(h_state, h_tn).
        """
        h_state = self._compute_state_heuristic(state_bitwise)
        h_tn = self._compute_tn_heuristic(state_bitwise, task_network)
        return float(max(h_state, h_tn))

    def _compute_state_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates the number of lamps in an incorrect state compared to the goal.
        """
        mismatches = 0
        if not self.goal_on_status:
            return 0

        for (x, y), goal_s in self.goal_on_status.items():
            # Determine current status by checking which 'on' fact is true
            current_s = None
            y_fact = self.on_facts.get((x, y, 'Y'))

            if y_fact and (state_bitwise >> y_fact.global_id) & 1:
                current_s = 'Y'
            else:
                # If not 'Y', it must be 'N', as per domain constraints
                current_s = 'N'

            if current_s != goal_s:
                mismatches += 1
        return mismatches

    def _compute_tn_heuristic(self, state_bitwise: int, task_network: list) -> int:
        """
        Calculates the minimum cost to execute the tasks in the current task network.
        """
        cost = 0
        for task in task_network:
            pred, args = self._parse_name(task.name)
            if pred == "turn":
                cost += 1  # A turn always involves at least one flip action.
            elif pred and pred.startswith("propagate_"):
                direction = pred.split('_')[1]
                if len(args) == 3:
                    x, y, s = args
                    cost += self._get_propagation_cost(direction, x, y, s, state_bitwise)
        return cost

    def _get_propagation_cost(self, direction: str, x: str, y: str, s1: str, state_bitwise: int) -> int:
        """
        Simulates a propagation chain to find the exact number of flips it will cause.
        """
        prop_cost = 0
        curr_x, curr_y = x, y

        while True:
            next_x, next_y = None, None
            if direction == 'up':
                next_x, next_y = self.next_map.get(curr_x), curr_y
            elif direction == 'down':
                next_x, next_y = self.prev_map.get(curr_x), curr_y
            elif direction == 'left':  # 'left' in domain means 'next' y
                next_x, next_y = curr_x, self.next_map.get(curr_y)
            elif direction == 'right': # 'right' in domain means 'prev' y
                next_x, next_y = curr_x, self.prev_map.get(curr_y)

            if next_x is None or next_y is None:
                break  # Reached edge of the grid

            fact_to_check = self.on_facts.get((next_x, next_y, s1))
            if fact_to_check and ((state_bitwise >> fact_to_check.global_id) & 1):
                prop_cost += 1
                curr_x, curr_y = next_x, next_y
            else:
                break
        return prop_cost
