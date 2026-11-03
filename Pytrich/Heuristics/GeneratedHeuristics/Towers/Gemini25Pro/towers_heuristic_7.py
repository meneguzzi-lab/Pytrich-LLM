import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic7(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic leverages the well-known optimal solution cost for the Towers of Hanoi problem.
    The core idea is to identify the largest ring that is not in its final, "settled" position.
    A ring is considered settled if it is on its goal object (either the destination tower or a larger ring)
    and all larger rings beneath it are also settled.

    The estimated cost to solve the problem is then based on the cost to move the stack of rings
    down to and including this largest unsettled ring. The cost to move a stack of 'k' rings from
    one peg to another is 2^k - 1. This value serves as a highly informative and admissible heuristic
    because any valid plan must, at a minimum, perform the actions required to move this entire sub-tower.

    During initialization, the heuristic pre-processes the domain information to:
    1. Identify all rings.
    2. Determine the size order of the rings based on 'smallerThan' predicates.
    3. Parse the goal state to map each ring to its final destination object.
    4. Create efficient lookup tables to quickly determine the position of each ring from a state's bitmask,
       avoiding slow string parsing during search.
    """

    def __init__(self, towers_gemini25pro_heuristic_7):
        super().__init__(name="towers_gemini25pro_heuristic_7")
        self.model = None
        self.sorted_rings = []
        self.ring_order_map = {}  # Maps ring name -> 1-based size index (1=smallest)
        self.goal_positions = {}  # Maps ring name -> target object name
        self.preprocessing_time = 0

        # For optimized state lookup
        self.on_pred_map = {}       # Maps fact_id -> (ring, on_object)
        self.ring_to_on_preds = {}  # Maps ring -> list of possible 'on' fact_ids

    def initialize(self, model: Model, initial_node: HTNNode):
        start_time = time.time()
        self.model = model
        
        self._parse_domain_info()
        
        self.preprocessing_time = time.time() - start_time
        
        h_value = self.__call__(None, initial_node)
        
        return super().initialize(model, h_value)

    def _parse_domain_info(self):
        all_rings = set()
        larger_than_counts = {}

        # First pass to identify rings and their relationships
        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            if len(parts) < 2:
                continue
            
            pred = parts[0].lower()
            if pred == 'smallerthan' and len(parts) == 3:
                r1, r2 = parts[1], parts[2]
                all_rings.update([r1, r2])
                if r1 not in larger_than_counts: larger_than_counts[r1] = 0
                if r2 not in larger_than_counts: larger_than_counts[r2] = 0
                larger_than_counts[r2] += 1
            elif pred == 'on' and len(parts) == 3:
                # Assume anything starting with 'r' is a ring
                if parts[1].startswith('r'):
                    all_rings.add(parts[1])
                    if parts[1] not in larger_than_counts: larger_than_counts[parts[1]] = 0

        # Sort rings by size (smallest to largest)
        self.sorted_rings = sorted(list(all_rings), key=lambda r: larger_than_counts.get(r, 0))
        self.ring_order_map = {ring: i + 1 for i, ring in enumerate(self.sorted_rings)}
        self.ring_to_on_preds = {ring: [] for ring in self.sorted_rings}

        # Second pass to build optimized lookup maps
        for fact in self.model.facts:
            parts = fact.name.strip("()").split()
            if len(parts) == 3 and parts[0].lower() == 'on':
                ring, on_object = parts[1], parts[2]
                if ring in self.ring_order_map:
                    self.on_pred_map[fact.global_id] = (ring, on_object)
                    self.ring_to_on_preds[ring].append(fact.global_id)

        # Parse goal positions from the goal bitmask
        goal_fact_ids = [i for i, fact in enumerate(self.model.facts) if (self.model.goals >> fact.global_id) & 1]
        for fact_id in goal_fact_ids:
            if fact_id in self.on_pred_map:
                ring, on_object = self.on_pred_map[fact_id]
                self.goal_positions[ring] = on_object

    def _get_current_positions(self, node: HTNNode):
        positions = {}
        state_bits = node.state
        for ring, possible_fact_ids in self.ring_to_on_preds.items():
            for fact_id in possible_fact_ids:
                # Check if the bit for this fact_id is set in the state
                if (state_bits >> fact_id) & 1:
                    _ring, on_object = self.on_pred_map[fact_id]
                    positions[ring] = on_object
                    break  # Each ring can only be in one place
        return positions

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        h_value = 0
        
        if not self.sorted_rings:
            super().update_info(0)
            return 0
            
        current_positions = self._get_current_positions(node)
        
        # Iterate from the largest ring downwards to find the first unsettled one
        for ring_name in reversed(self.sorted_rings):
            order_index = self.ring_order_map.get(ring_name)
            if order_index is None:
                continue

            goal_pos = self.goal_positions.get(ring_name)
            current_pos = current_positions.get(ring_name)
            
            # If the current position does not match the goal, this is the largest
            # unsettled ring. The cost is the number of moves to transfer a tower
            # of its size.
            if current_pos != goal_pos:
                h_value = (2**order_index) - 1
                break
                
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"TowersGemini25ProHeuristic7(rings={len(self.sorted_rings)})"

    def __str__(self):
        return self.name

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 7 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of Rings Identified: {len(self.sorted_rings)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.6f}s\n"
            f"\tGoal Positions Found: {len(self.goal_positions)}"
        )
