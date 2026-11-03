import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic24(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic is based on the concept of the "largest misplaced disk".
    The Tower of Hanoi problem has an optimal substructure. To move a stack of k
    disks to a new tower, it takes a minimum of 2^k - 1 moves. This heuristic
    leverages this fact by identifying the largest disk that is not in its final
    goal position. Let this disk be the k-th largest disk (where the smallest is 1).

    The heuristic then returns 2^k - 1 as the estimated cost to solve the
    subproblem of getting this disk and all smaller disks into their correct goal
    positions. This is an admissible and highly informative heuristic for this
    domain, as moving the k-th disk correctly requires solving the subproblem of
    moving the k-1 disks above it at least once.

    The initialization phase pre-processes the model to:
    1. Identify all rings and sort them by size (e.g., r1 < r2 < ...).
    2. Parse all `goal_on` predicates to build a map of the goal configuration.
    3. Create a mapping from fact indices to `(on ?ring ?object)` information
       for efficient decoding of the state bitmask.
    """

    def __init__(self, towers_gemini25pro_heuristic_24=None):
        super().__init__(name="towers_gemini25pro_heuristic_24")
        self.model = None
        self.goal_config = {}  # Maps ring_name -> goal_object_name
        self.rings_sorted = []  # List of ring names, smallest to largest
        self.on_predicate_idx_to_info = {}  # Maps fact_id -> (ring, on_obj)

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by preprocessing the planning model.
        It identifies rings, goal states, and creates mappings for fast state evaluation.
        """
        start_time = time.time()
        self.model = model

        rings = set()
        for fact in self.model.facts:
            fact_name = fact.name
            fact_id = fact.global_id

            # Parse goal predicates to build the goal configuration map
            if fact_name.startswith("(goal_on "):
                parts = fact_name.strip("()").split()
                ring_name, goal_obj_name = parts[1], parts[2]
                self.goal_config[ring_name] = goal_obj_name
                rings.add(ring_name)
            # Parse 'on' predicates to build a map for quick state decoding
            elif fact_name.startswith("(on "):
                parts = fact_name.strip("()").split()
                ring_name, on_obj_name = parts[1], parts[2]
                self.on_predicate_idx_to_info[fact_id] = (ring_name, on_obj_name)

        # Sort rings by size, assuming names like 'r1', 'r2', etc.
        if rings:
            self.rings_sorted = sorted(list(rings), key=lambda r: int(r[1:]))

        self.preprocessing_time = time.time() - start_time

        # Calculate the heuristic value for the initial node
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _get_current_on_config(self, node_state: int) -> dict:
        """
        Decodes the bitwise state into a dictionary mapping each ring to its location.
        """
        current_on = {}
        # Iterate through the pre-processed 'on' predicate map
        for idx, info in self.on_predicate_idx_to_info.items():
            if (node_state >> idx) & 1:  # Check if the bit for this 'on' fact is set
                ring, on_obj = info
                current_on[ring] = on_obj
        return current_on

    def _calculate_node_heuristic(self, node: HTNNode) -> int:
        """
        Calculates the heuristic value for a given node.
        Finds the largest misplaced disk and returns the cost of the corresponding subproblem.
        """
        if not node or self.model is None or not self.rings_sorted:
            return 0

        current_on_config = self._get_current_on_config(node.state)

        # Iterate from the largest ring downwards
        for i in range(len(self.rings_sorted) - 1, -1, -1):
            ring = self.rings_sorted[i]

            current_location = current_on_config.get(ring)
            goal_location = self.goal_config.get(ring)

            # If a ring is not in its goal location, we've found the largest misplaced ring.
            if current_location != goal_location:
                # The size of this subproblem is k = i + 1 (1-based index).
                # The minimum cost to solve a Hanoi problem of size k is 2^k - 1.
                k = i + 1
                return (2 ** k) - 1

        # If the loop completes, all rings are in their goal positions.
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
