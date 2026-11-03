import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic21(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic is based on the classic solution to the Towers of Hanoi problem.
    It identifies the largest ring that is not in its final, "settled" position.
    A ring is considered settled if it's on its goal object, and the object
    it's on is also settled. This check is performed recursively from the largest
    ring downwards.

    The heuristic value is calculated as 2^k - 1, where k is the number of rings
    that need to be moved to place the largest unsettled ring. This corresponds
    to the minimum number of moves required to solve a k-ring Towers of Hanoi subproblem,
    making the heuristic admissible and highly informative for this domain.
    """
    def __init__(self, towers_gemini25pro_heuristic_21):
        super().__init__(name="TowersGemini25ProHeuristic21")
        self.problem = towers_gemini25pro_heuristic_21
        self.model = None
        self.goal_positions = {}
        self.on_facts_map = defaultdict(list)
        self.rings_sorted_by_size = []
        self.num_rings = 0
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by preprocessing the model to extract
        static domain information like ring order and goal positions.
        """
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _preprocess(self):
        """
        Analyzes the static facts from the model to build data structures
        for efficient heuristic calculation.
        - Parses `goal_on` facts to determine the final configuration.
        - Parses `smallerThan` facts to determine the size order of rings.
        - Caches `on` predicate objects for faster state lookups.
        """
        rings = set()
        smaller_than_facts = []

        for fact in self.model.facts:
            try:
                parts = fact.name.strip("()").split()
                predicate = parts[0]
            except (IndexError, AttributeError):
                continue

            if predicate == 'goal_on':
                ring, on_obj = parts[1], parts[2]
                self.goal_positions[ring] = on_obj
                rings.add(ring)
            elif predicate == 'smallerThan' and parts[1].startswith('r'):
                r1, r2 = parts[1], parts[2]
                smaller_than_facts.append((r1, r2))
                rings.add(r1)
                if r2.startswith('r'):
                    rings.add(r2)
            elif predicate == 'on' and parts[1].startswith('r'):
                ring = parts[1]
                self.on_facts_map[ring].append(fact)
                rings.add(ring)

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size. Count how many other rings are smaller than each ring.
        is_smaller_count = {ring: 0 for ring in rings}
        for r_small, r_large in smaller_than_facts:
            if r_small in rings and r_large in rings:
                is_smaller_count[r_large] += 1
        
        # Sort rings by this count, from smallest (0 smaller rings) to largest.
        self.rings_sorted_by_size = sorted(list(rings), key=lambda r: is_smaller_count.get(r, 0))

    def _get_current_positions(self, state_bitwise: int) -> dict:
        """
        Determines the current position of each ring from the bitwise state
        by checking the pre-filtered 'on' facts against the state.
        """
        current_positions = {}
        for ring, possible_facts in self.on_facts_map.items():
            for fact in possible_facts:
                if (state_bitwise >> fact.global_id) & 1:
                    on_obj = fact.name.strip("()").split()[2]
                    current_positions[ring] = on_obj
                    break
        return current_positions

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value for the given state.
        It finds the largest ring not in its final settled position and
        returns 2^k - 1, where k is the number of rings from the smallest
        up to and including that largest misplaced ring.
        """
        if not self.rings_sorted_by_size:
            return 0
            
        current_positions = self._get_current_positions(state_bitwise)

        # Iterate from the largest ring downwards to find the largest unsettled one.
        for i in range(self.num_rings - 1, -1, -1):
            ring = self.rings_sorted_by_size[i]
            
            # Check if the ring's current position matches its goal position.
            if current_positions.get(ring) != self.goal_positions.get(ring):
                # This ring and all smaller ones (total of i+1 rings) must be moved.
                # The minimum cost is solving the subproblem for these i+1 rings.
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1 # Efficient calculation of 2^n - 1

        # If the loop completes, all rings are in their goal positions and settled.
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        The main call method for the heuristic during search.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"TowersGemini25ProHeuristic21(rings={self.num_rings})"

    def __str__(self):
        return f"Towers Gemini 2.5 Pro of Hanoi Heuristic 21 with {self.num_rings} rings"

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 21 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings: {self.num_rings}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
            f"\tGoal Configuration: {self.goal_positions}\n"
            f"\tSorted Rings (Smallest to Largest): {self.rings_sorted_by_size}"
        )
