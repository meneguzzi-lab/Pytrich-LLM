import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

# Domain-dependent heuristic for the Towers of Hanoi (towers) domain.
class TowersGemini25ProHeuristic2(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic estimates the cost to reach the goal by counting the number of "unsettled" rings.
    A ring is considered "settled" if it is in its final goal position (i.e., on top of the correct
    object, which can be another ring or a tower), AND the object it is resting on is also settled.
    This check is performed recursively from the largest ring down to the smallest.

    The rationale is that to solve the puzzle, one must build the goal tower from the bottom up.
    Any ring that is not part of this correctly-built base is "unsettled" and must be moved at
    least once. Therefore, the number of unsettled rings is a consistent and admissible
    heuristic for a problem where each move action has a cost of 1.

    The heuristic is computationally efficient, requiring a single pass through the rings for each
    state evaluation after an initial preprocessing step.
    """
    def __init__(self, towers_gemini25pro_heuristic_2=1.0, name="towers_gemini25pro_heuristic_2"):
        super().__init__(name=name)
        # The penalty for each misplaced (unsettled) ring. Default is 1.0.
        self.misplaced_penalty = towers_gemini25pro_heuristic_2
        self.preprocessing_time = 0
        self.goal_map = {}
        self.rings = set()
        self.towers = set()
        self.sorted_rings_large_to_small = []

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by preprocessing domain information from the model.
        It parses all facts to identify rings, towers, and the goal configuration.
        """
        start_time = time.time()
        self.model = model
        self._preprocess_domain_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _preprocess_domain_info(self):
        """
        Parses the model's facts to extract goal positions, rings, and towers.
        This information is cached for efficient heuristic calculations later.
        """
        temp_rings = set()
        
        for fact in self.model.facts:
            fact_str = fact.name
            # Use regex to parse '(predicate p1 p2)'
            match = re.match(r'^\((\w+)\s+([\w-]+)\s+([\w-]+)\)$', fact_str)
            if not match:
                continue

            pred_name, p1, p2 = match.groups()

            # Infer object types from how they are used in predicates
            if pred_name == 'goal_on':
                self.goal_map[p1] = p2
                temp_rings.add(p1)
            elif pred_name == 'smallerthan':
                temp_rings.add(p1)
            elif pred_name == 'towertop':
                self.towers.add(p2)
                # If the top of a tower is not another tower, it must be a ring
                if not p2.startswith('t'):
                    temp_rings.add(p1)

        self.rings = temp_rings
        
        # Sort rings by size. This assumes the standard naming convention 'r1', 'r2', etc.,
        # where a larger number implies a larger ring.
        try:
            ring_list = sorted(list(self.rings), key=lambda r: int(r[1:]))
            self.sorted_rings_large_to_small = list(reversed(ring_list))
        except (ValueError, IndexError):
            # Fallback for non-standard ring names: alphabetical sort.
            self.sorted_rings_large_to_small = sorted(list(self.rings), reverse=True)


    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node's state.
        The heuristic counts the number of unsettled rings.
        """
        if not self.rings:
            return 0

        state_facts = self.model.state_explicit_repr(node.state)
        
        current_on_map = {}
        for fact_str in state_facts:
            if fact_str.startswith('(on '):
                match = re.match(r'^\(on\s+([\w-]+)\s+([\w-]+)\)$', fact_str)
                if match:
                    ring, support = match.groups()
                    current_on_map[ring] = support
        
        # If state has no 'on' facts, assume worst case (all rings unsettled)
        if not current_on_map:
             return len(self.rings) * self.misplaced_penalty

        # A settled object is one that is in its final goal position, resting
        # on another settled object. Towers are settled by definition.
        settled_objects = set(self.towers)
        
        for ring in self.sorted_rings_large_to_small:
            goal_support = self.goal_map.get(ring)
            current_support = current_on_map.get(ring)
            
            # This ring is settled if it's on its correct support object,
            # and that support object has itself been settled.
            if current_support == goal_support and goal_support in settled_objects:
                settled_objects.add(ring)

        num_total_rings = len(self.rings)
        # The number of settled rings is the count of settled objects minus the base towers.
        num_settled_rings = len(settled_objects) - len(self.towers)
        
        num_unsettled = num_total_rings - num_settled_rings
        heuristic_value = num_unsettled * self.misplaced_penalty
        
        return heuristic_value

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given search node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic2(penalty={self.misplaced_penalty})"

    def __str__(self):
        return self.__repr__()
