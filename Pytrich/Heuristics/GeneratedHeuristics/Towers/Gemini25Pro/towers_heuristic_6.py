import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic6(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic calculates the remaining cost to reach the goal state by adopting
    the standard optimal cost formula for Towers of Hanoi subproblems. It operates
    under the crucial assumption that the current world state is accessible from the
    search node provided to the heuristic.

    Heuristic Logic:
    1.  **Preprocessing (`initialize`):**
        -   Identifies all rings and sorts them by size using the `smallerThan` predicate.
        -   Parses all `goal_on` predicates to build a representation of the goal state.
        -   Creates a mapping from `(on ring obj)` predicates to their internal fact IDs
            to allow for efficient checking of the current state.

    2.  **Evaluation (`__call__`):**
        -   It assumes the input `node` has a `.state` attribute containing a bitmask of the current state.
        -   It determines the current position of all rings from the state bitmask.
        -   It finds the largest ring (`r_k`) that is not in its correct goal position (i.e., not on the correct object as specified by the goal).
        -   The heuristic value is then calculated as `2^k - 1`, which is the exact minimum number
            of moves required to solve a Towers of Hanoi subproblem of size `k` (moving `r_k` and all smaller rings to their correct final configuration).
        -   If all rings are in their correct goal positions, the heuristic value is 0.

    This state-based approach is highly effective and admissible for this domain, as it directly
    estimates the cost of the largest remaining subproblem.
    """

    def __init__(self, towers_gemini25pro_heuristic_6=None):
        super().__init__(name="towers_gemini_heuristic_6")
        self.model = None
        self.rings_sorted = []
        self.goal_pos = {}
        self.on_fact_map = {}
        self.preprocessing_time = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model

        # 1. Identify all rings from the domain facts.
        rings = set()
        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            if parts[0] in ["on", "goal_on", "smallerthan"] and len(parts) > 1:
                rings.add(parts[1])

        # 2. Sort rings by size using 'smallerThan' facts.
        smaller_counts = {ring: 0 for ring in rings}
        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            if parts[0] == "smallerthan" and len(parts) == 3:
                larger_ring = parts[2]
                if larger_ring in smaller_counts:
                    smaller_counts[larger_ring] += 1
        
        self.rings_sorted = sorted(list(rings), key=lambda r: smaller_counts.get(r, 0))

        # 3. Parse goal configuration from 'goal_on' facts.
        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            if parts[0] == "goal_on" and len(parts) == 3:
                ring, obj_below = parts[1], parts[2]
                self.goal_pos[ring] = obj_below

        # 4. Create a map for 'on' facts for efficient state checking.
        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            if parts[0] == "on" and len(parts) == 3:
                ring, obj_below = parts[1], parts[2]
                self.on_fact_map[(ring, obj_below)] = fact.global_id

        self.preprocessing_time = time.time() - start_time
        
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _get_current_positions(self, state: int) -> dict:
        """
        Constructs a dictionary of current ring positions from a bitwise state.
        """
        current_pos = {}
        for (ring, obj_below), fact_id in self.on_fact_map.items():
            if (state >> fact_id) & 1:
                current_pos[ring] = obj_below
        return current_pos

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """
        Calculates the heuristic value based on the largest misplaced ring.
        
        ASSUMPTION: This heuristic critically relies on the `node` object having a 
        `.state` attribute that provides the current world state as a bitmask. 
        Without access to the state, this heuristic cannot function as designed.
        """
        if not hasattr(node, 'state'):
            # This heuristic is state-dependent. If state is unavailable, 
            # it cannot provide guidance. Return 0 as a neutral value.
            return 0
        
        state = node.state
        current_pos = self._get_current_positions(state)
        
        num_rings = len(self.rings_sorted)
        
        # Iterate from the largest ring downwards.
        for i in range(num_rings - 1, -1, -1):
            ring = self.rings_sorted[i]
            
            # If the ring's current support object doesn't match the goal, we've found our subproblem.
            if current_pos.get(ring) != self.goal_pos.get(ring):
                # The size of the tower to move is k = i + 1.
                k = i + 1
                # The optimal number of moves is 2^k - 1.
                h_value = (2 ** k) - 1
                super().update_info(h_value)
                return h_value
                
        # If the loop completes, all rings are in their goal positions.
        h_value = 0
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic6(rings={len(self.rings_sorted)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 6 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of rings: {len(self.rings_sorted)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
