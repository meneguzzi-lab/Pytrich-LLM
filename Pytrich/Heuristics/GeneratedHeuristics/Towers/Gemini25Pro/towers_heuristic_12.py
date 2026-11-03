import time
import re
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic12(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic calculates the remaining cost based on the classic Towers of Hanoi problem.
    It identifies the largest ring that is not in its final, "settled" position.
    A ring is considered settled if it is in its goal location, and all larger rings
    are also in their goal locations.

    The heuristic value is calculated as 2^k - 1, where 'k' is the size of the
    largest unsettled ring. This formula represents the minimum number of moves required
    to move a tower of 'k' rings from a source peg to a destination peg. This provides
    an admissible and highly accurate estimate of the remaining work, making it very
    effective for guiding the search in this domain.
    """

    def __init__(self, towers_gemini25pro_heuristic_12=None):
        """
        Initializes the TowersGemini25ProHeuristic12.
        """
        super().__init__(name="towers_gemini25pro_heuristic_12")
        self.model = None
        self.goal_config = {}
        self.ring_sizes = {}
        self.sorted_rings = []
        self.id_to_fact_name = {}
        self.on_predicate_fact_ids = set()
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by preprocessing the domain model to extract
        ring sizes, goal configurations, and relevant predicate information.
        This is done once at the beginning of the search.
        """
        start_time = time.time()
        self.model = model
        all_rings = set()

        # 1. Map fact IDs to names and identify all 'on' predicates to track ring locations
        for fact in self.model.facts:
            self.id_to_fact_name[fact.global_id] = fact.name
            if fact.name.startswith("(on "):
                self.on_predicate_fact_ids.add(fact.global_id)
                match = re.match(r'^\(on\s+(r\d+)\s+.*\)$', fact.name)
                if match:
                    all_rings.add(match.group(1))

        # 2. Parse goal conditions from the model's goal bitmask to build the target config map
        goal_fact_names = set()
        if hasattr(self.model, 'goals'):
            for i in range(len(self.model.facts)):
                if (self.model.goals >> i) & 1:
                    goal_fact_names.add(self.id_to_fact_name.get(i))

        for fact_name in goal_fact_names:
             if fact_name and fact_name.startswith("(on "):
                    match = re.match(r'^\(on\s+(r\d+)\s+(.*)\)$', fact_name)
                    if match:
                        ring, on_obj = match.groups()
                        self.goal_config[ring] = on_obj
        
        # 3. Determine ring sizes from their names (e.g., 'r10' -> 10)
        for ring in all_rings:
            try:
                size = int(ring[1:])
                self.ring_sizes[ring] = size
            except (ValueError, IndexError):
                self.ring_sizes[ring] = 0 # Fallback for non-standard ring names

        # 4. Create a list of rings sorted by size, from largest to smallest
        self.sorted_rings = sorted(list(all_rings), key=lambda r: self.ring_sizes.get(r, 0), reverse=True)

        self.preprocessing_time = time.time() - start_time
        
        # Calculate heuristic for the initial state
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value for a given state.
        """
        # 1. Decode the current positions of all rings from the bitwise state
        current_positions = {}
        for fact_id in self.on_predicate_fact_ids:
            if (state_bitwise >> fact_id) & 1:
                fact_name = self.id_to_fact_name[fact_id]
                match = re.match(r'^\(on\s+(r\d+)\s+(.*)\)$', fact_name)
                if match:
                    ring, on_obj = match.groups()
                    current_positions[ring] = on_obj
        
        # 2. Find the largest ring 'k' that is not 'settled'
        largest_unsettled_ring_size = 0
        for ring in self.sorted_rings:
            if current_positions.get(ring) != self.goal_config.get(ring):
                largest_unsettled_ring_size = self.ring_sizes.get(ring, 0)
                break
        
        # 3. Compute heuristic value: 2^k - 1
        if largest_unsettled_ring_size == 0:
            return 0  # All rings are settled.
        else:
            return (2 ** largest_unsettled_ring_size) - 1

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        The main heuristic evaluation function called by the planner.
        """
        if not hasattr(node, 'state'):
            return 1

        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic12(rings={len(self.sorted_rings)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 12 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings Tracked: {len(self.sorted_rings)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
            f"\tGoal Configuration Loaded: {'Yes' if self.goal_config else 'No'}"
        )
