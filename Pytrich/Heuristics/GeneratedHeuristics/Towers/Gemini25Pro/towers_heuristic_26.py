import re
import time
from collections import deque
from Pytrich.Heuristics.heuristic import Heuristic

class TowersGemini25ProHeuristic26(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic calculates an admissible estimate of the remaining cost (number of moves)
    to reach the goal state. It is based on the classic optimal solution strategy for
    the Towers of Hanoi problem.

    Rationale:
    1.  **Domain Analysis**: The 'towers' domain is a direct implementation of the
        Towers of Hanoi puzzle. The key challenge is the exponential complexity
        (2^N - 1 moves for N rings) and the strict `smallerThan` constraint. A good
        heuristic must capture this exponential nature to be effective.

    2.  **Core Idea**: The heuristic identifies the largest ring that is not in its
        final, correct position as defined by the goal state. Let this be the k-th
        largest ring (out of N total rings). To move this k-th ring to its goal
        destination, all k-1 smaller rings must first be moved to the single
        auxiliary tower. This subproblem requires exactly 2^(k-1) - 1 moves. Then,
        the k-th ring is moved (1 move), and finally, the k-1 smaller rings are
        moved from the auxiliary tower on top of the k-th ring (another 2^(k-1) - 1
        moves). The total number of moves involving the k-th ring and smaller ones is
        therefore (2^(k-1) - 1) + 1 + (2^(k-1) - 1) = 2^k - 1.

    3.  **Admissibility**: Since moving the k-th largest misplaced ring *requires* at
        least 2^k - 1 moves in an optimal plan, this value is a lower bound on the
        true remaining cost. Therefore, the heuristic is admissible.

    4.  **Implementation**:
        - **Initialization**: During the `initialize` phase, the heuristic performs
          preprocessing:
            - It parses all `(smallerThan ...)` facts to build a dependency graph
              of rings.
            - It performs a topological sort on this graph to get a list of all rings
              ordered from smallest to largest.
            - It parses the `(goal ...)` predicates (which are `on` predicates in the goal section)
              to build a map of the final desired configuration for each ring.
            - It creates a mapping from fact IDs of `(on ...)` predicates to the
              ring and object involved, for efficient state decoding.
        - **Evaluation (`__call__`)**: For any given state node:
            - It decodes the bitwise state to determine the current location of each ring.
            - It iterates from the largest ring down to the smallest.
            - The first ring it finds that is not in its goal location determines the
              heuristic value. If this is the k-th ring (from smallest, 1-indexed),
              the value is `2^k - 1`.
            - If all rings are in their correct places, the heuristic value is 0.

    This state-based approach provides a highly accurate and admissible heuristic,
    drastically improving search performance compared to a generic or method-based
    heuristic for this specific domain.
    """
    def __init__(self, towers_gemini25pro_heuristic_26=1.0, name="towers_gemini25pro_heuristic_26"):
        super().__init__(name=name)
        self.model = None
        self.sorted_rings = []
        self.goal_config = {}
        self.on_fact_info = {}
        self.cost_weight = towers_gemini25pro_heuristic_26
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        # Preprocessing: parse facts, sort rings, and build goal configuration
        self._parse_model_info()

        self.preprocessing_time = time.time() - start_time

        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _parse_pddl_atom(self, atom_string):
        """Helper to parse a string like '(predicate obj1 obj2)' into a list."""
        atom_string = atom_string.strip()
        if atom_string.startswith('(') and atom_string.endswith(')'):
            return atom_string[1:-1].lower().split()
        return []

    def _parse_model_info(self):
        """Extracts ring order, goal state, and fact info from the model."""
        all_rings = set()
        smaller_than_pairs = []

        # 1. Collect ring and relationship info from all facts
        for fact in self.model.facts:
            parts = self._parse_pddl_atom(fact.name)
            if not parts:
                continue
            
            predicate = parts[0]
            if predicate == 'smallerthan' and len(parts) == 3:
                r1, r2 = parts[1], parts[2]
                all_rings.add(r1)
                all_rings.add(r2)
                smaller_than_pairs.append((r1, r2))
            elif predicate == 'on' and len(parts) == 3:
                r, o = parts[1], parts[2]
                if r.startswith('r'): # Assuming rings are named 'r...'
                    all_rings.add(r)
                    self.on_fact_info[fact.global_id] = (r, o)

        # 2. Build goal configuration from model goals
        for goal_fact in self.model.goals:
             parts = self._parse_pddl_atom(goal_fact.name)
             if parts and parts[0] == 'on' and len(parts) == 3:
                 self.goal_config[parts[1]] = parts[2]
        
        # 3. Topologically sort the rings by size
        in_degree = {ring: 0 for ring in all_rings}
        adj = {ring: [] for ring in all_rings}

        for r1, r2 in smaller_than_pairs:
            if r1 in all_rings and r2 in all_rings and r1.startswith('r') and r2.startswith('r'):
                adj[r1].append(r2)
                in_degree[r2] += 1
        
        # Rings are objects that can be smaller than others, so they have outgoing edges.
        # Start with rings that are not smaller than any other ring (in-degree 0 in the context of 'smaller-than').
        queue = deque([ring for ring in all_rings if in_degree[ring] == 0 and ring.startswith('r')])
        
        while queue:
            u = queue.popleft()
            self.sorted_rings.append(u)
            
            if u in adj:
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        
        # Filter out any non-ring objects that might have been added
        self.sorted_rings = [r for r in self.sorted_rings if r in self.goal_config or any(r == info[0] for info in self.on_fact_info.values())]


    def _get_current_config(self, state_bitwise):
        """Decodes the bitwise state into a map of ring locations."""
        current_config = {}
        for fact_id, (ring, obj) in self.on_fact_info.items():
            if (state_bitwise >> fact_id) & 1:
                current_config[ring] = obj
        return current_config

    def __call__(self, parent_node, node):
        if not hasattr(node, 'state'):
            return 0
        
        state = node.state
        
        # If task network is empty and goals are met, cost is 0
        if self.model.goal_reached(state, node.task_network):
             return 0

        current_config = self._get_current_config(state)
        h = 0
        
        # Find the largest ring that is not in its goal position
        # The list is sorted smallest to largest, so we iterate in reverse
        num_rings = len(self.sorted_rings)
        for i, ring in enumerate(reversed(self.sorted_rings)):
            ring_size_index = num_rings - i # 1-based index of ring size
            
            goal_obj = self.goal_config.get(ring)
            current_obj = current_config.get(ring)
            
            # If a ring's position is not its goal position, we've found our k.
            if goal_obj is not None and current_obj != goal_obj:
                h = (2**ring_size_index) - 1
                break
        
        h_value = self.cost_weight * h
        super().update_info(h_value)
        return h_value
    
    def __repr__(self):
        return f"TowersGemini25ProHeuristic26(cost_weight={self.cost_weight})"

    def __str__(self):
        return f"TowersGemini25ProHeuristic26(cost_weight={self.cost_weight})"

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 26 Info:\n"
            f"\tName: {self.name}\n"
            f"\tSorted Rings Found: {len(self.sorted_rings)}\n"
            f"\tGoal Predicates Parsed: {len(self.goal_config)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
