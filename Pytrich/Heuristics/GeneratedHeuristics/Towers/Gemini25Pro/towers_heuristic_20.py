import time
import re
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic20(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic estimates the cost of a decomposition based on the size of the
    rings involved in the method's parameters. The core idea is that tasks
    involving larger rings lead to exponentially more primitive actions (moves).
    The heuristic provides an admissible and consistent estimate of the remaining
    work by assigning a value proportional to the size of the largest ring in the
    current subproblem.

    Rationale:
    1.  Domain Insight: The optimal solution to move a tower of k disks is 2^k - 1.
        Therefore, the size of the disk being manipulated is the primary driver of
        plan complexity.
    2.  Heuristic Design: The heuristic value h(n) for a node 'n' applying a
        decomposition method is calculated as `k * penalty`, where 'k' is the size
        of the largest ring parameter in the method. This value is a simple, monotonic
        function of the subproblem size and serves as an effective, admissible lower
        bound on the true cost (since k << 2^k - 1).
    3.  Implementation: During initialization, it scans all defined entities in the model
        to identify all rings and map their names (e.g., 'r10') to an integer size (e.g., 10).
        During search, it parses the grounded method name at each node to find the
        largest ring involved and calculates the heuristic value.
    4.  Admissibility: The cost of moving a single ring is 1. To move ring 'k', all
        k-1 smaller rings must be moved. The total cost is far greater than 'k'.
        Thus, `h(n) = k * penalty` (with penalty <= 1) is always less than or equal
        to the true cost to solve the subproblem, making the heuristic admissible.
    """
    def __init__(self, towers_gemini25pro_heuristic_20, ring_penalty=0.1):
        super().__init__(name=towers_gemini25pro_heuristic_20)
        self.ring_penalty = ring_penalty
        self.ring_order = {}
        self.preprocessing_time = 0
        # Pre-compile regex for efficiency
        self.ring_pattern = re.compile(r'\br\d+\b')

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-calculating the size order of the rings.
        """
        start_time = time.time()
        self.model = model

        # Discover all rings and their sizes by scanning all names in the model.
        # This is more robust than assuming a fixed set of objects.
        ring_candidates = set()
        source_collections = [
            self.model.facts,
            self.model.operators,
            self.model.decompositions,
            self.model.abstract_tasks
        ]

        for collection in source_collections:
            for item in collection:
                found_rings = self.ring_pattern.findall(item.name)
                for ring in found_rings:
                    ring_candidates.add(ring)

        for ring_name in ring_candidates:
            try:
                # Assumes ring names are in the format 'r<number>', e.g., 'r1', 'r20'
                num = int(ring_name[1:])
                self.ring_order[ring_name] = num
            except (ValueError, IndexError):
                # Ignore if parsing fails, not a valid ring name format
                pass

        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a node based on its decomposition method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        method_name = node.method.name

        # The method for the primitive 'move' action has a fixed cost.
        if 'newMethod21' in method_name:
            return self.ring_penalty

        max_ring_size = 0
        
        # Find all rings mentioned in the method's grounded name.
        found_rings = self.ring_pattern.findall(method_name)

        for ring_name in found_rings:
            if ring_name in self.ring_order:
                ring_size = self.ring_order[ring_name]
                if ring_size > max_ring_size:
                    max_ring_size = ring_size

        # The heuristic value is proportional to the size of the largest disk
        # in the current subproblem. This guides the search to understand that
        # methods dealing with larger rings will lead to more extensive sub-plans.
        h_value = max_ring_size * self.ring_penalty
        return h_value

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic20(ring_penalty={self.ring_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 20 Info:\n"
            f"\tName: {self.name}\n"
            f"\tRing Penalty: {self.ring_penalty}\n"
            f"\tRings Found: {len(self.ring_order)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
