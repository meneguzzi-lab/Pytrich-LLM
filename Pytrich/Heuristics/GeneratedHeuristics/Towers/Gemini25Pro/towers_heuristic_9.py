import time
import re
from Pytrich.Heuristics.heuristic import Heuristic

class TowersGemini25ProHeuristic9(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic calculates the cost to reach the goal by identifying the largest
    disk that is not in its final, "settled" position. A disk is considered
    settled if it is on its goal object (either a tower or another disk), and
    all disks supporting it are also settled.

    The core idea is that to move the largest unsettled disk (which has a size index k),
    one must first move the entire tower of k disks on top of it to an auxiliary peg.
    The cost of this operation is 2^k - 1. Then, moving the disk itself costs 1 move.
    The total cost to properly place the stack of k+1 disks is therefore 2^(k+1) - 1.

    This heuristic is admissible because it calculates the exact minimum number
    of moves required to solve the subproblem defined by the largest misplaced disk,
    which is a lower bound on the total moves remaining.

    NOTE: This heuristic is state-dependent and relies on accessing the current
    state from the search node (i.e., `node.state`). This is based on the task
    requirement to "not ignore dynamic state-dependent factors". If the planning
    framework does not provide state information in the node, this heuristic will
    return 0.
    """

    def __init__(self, towers_gemini25pro_heuristic_9=None):
        super().__init__(name="towers_gemini25pro_heuristic_9")
        self.model = None
        self.goal_config = {}
        self.ring_sizes = {}
        self.sorted_rings = []
        self.on_pred_map = {}
        self.ring_map = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by preprocessing the domain model to extract
        static information like ring sizes and goal configuration.
        """
        start_time = time.time()
        self.model = model
        self._preprocess_domain()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _preprocess_domain(self):
        """
        Parses the model's facts to build data structures for efficient heuristic calculation.
        - Identifies all rings.
        - Calculates the size of each ring (0 for smallest, N-1 for largest).
        - Stores the final goal configuration from `goal_on` predicates.
        - Maps `(on ...)` predicate strings to their bitwise IDs.
        """
        # 1. Identify all rings from `smallerThan` predicates.
        rings = set()
        smaller_than_facts = [f.name for f in self.model.facts if f.name.startswith("(smallerThan")]
        for fact_str in smaller_than_facts:
            match = re.match(r"\(smallerThan\s(.*?)\s(.*?)\)", fact_str)
            if match:
                rings.add(match.group(1))
        self.ring_map = {r: "RING" for r in rings}

        # 2. Determine ring sizes: size[r] = number of rings smaller than r.
        for r in self.ring_map:
            smaller_rings = {
                m.group(1) for f in smaller_than_facts
                if (m := re.match(rf"\(smallerThan\s(.*?)\s{r}\)", f))
            }
            self.ring_sizes[r] = len(smaller_rings)

        # 3. Sort rings by size, largest first, for efficient processing.
        self.sorted_rings = sorted(self.ring_map.keys(), key=lambda r: self.ring_sizes[r], reverse=True)

        # 4. Build goal configuration from `goal_on` predicates in the initial state.
        initial_state_facts = self.model.state_explicit_repr(self.model.initial_state)
        for fact_str in initial_state_facts:
            if fact_str.startswith("(goal_on"):
                match = re.match(r"\(goal_on\s(.*?)\s(.*?)\)", fact_str)
                if match:
                    ring, on_obj = match.groups()
                    self.goal_config[ring] = on_obj

        # 5. Map 'on' predicates to their bitwise index for quick lookup from state.
        for fact in self.model.facts:
            if fact.name.startswith("(on "):
                self.on_pred_map[fact.name] = fact.global_id

    def _get_current_on_config(self, state_bitwise):
        """Extracts the current `(on ring object)` configuration from a bitwise state."""
        config = {}
        for fact_str, fact_id in self.on_pred_map.items():
            if (state_bitwise >> fact_id) & 1:
                match = re.match(r"\(on\s(.*?)\s(.*?)\)", fact_str)
                if match:
                    ring, on_obj = match.groups()
                    config[ring] = on_obj
        return config

    def _is_settled(self, ring, current_on_config, settled_cache):
        """Recursively checks if a ring is settled, using memoization."""
        if ring in settled_cache:
            return settled_cache[ring]
            
        goal_on_obj = self.goal_config.get(ring)
        current_on_obj = current_on_config.get(ring)
        
        if goal_on_obj is None or current_on_obj != goal_on_obj:
            settled_cache[ring] = False
            return False
            
        if goal_on_obj not in self.ring_map:
            settled_cache[ring] = True
            return True
            
        below_ring_is_settled = self._is_settled(goal_on_obj, current_on_config, settled_cache)
        settled_cache[ring] = below_ring_is_settled
        return below_ring_is_settled

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for the state in the given node.
        """
        if not hasattr(node, 'state'):
            return 0

        current_on_config = self._get_current_on_config(node.state)
        settled_cache = {}
        
        largest_unsettled_stack_size = 0
        for ring in self.sorted_rings:
            if not self._is_settled(ring, current_on_config, settled_cache):
                largest_unsettled_stack_size = self.ring_sizes[ring] + 1
                break
        
        h_value = (2 ** largest_unsettled_stack_size) - 1 if largest_unsettled_stack_size > 0 else 0
            
        super().update_info(h_value)
        return h_value
