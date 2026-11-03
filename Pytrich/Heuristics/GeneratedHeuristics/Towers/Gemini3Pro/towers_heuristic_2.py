import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Towers of Hanoi) domain.
    
    This heuristic implements the optimal solution cost calculation for the Towers of Hanoi problem.
    The minimum number of moves to transfer a stack of n disks is 2^n - 1.
    
    Algorithm:
    1. Preprocessing: 
       - Identifies the size ordering of rings (smallest to largest).
       - Identifies the goal position for each ring.
       - Maps 'on' facts to specific rings for efficient state lookup.
    
    2. Evaluation:
       - Iterates through rings from Largest to Smallest.
       - Checks if the current ring is at its goal position.
       - The first (largest) ring found that is NOT at its goal position implies that it and all 
         smaller rings must be moved. 
       - If the k-th ring (0-indexed size) is the largest misplaced ring, the heuristic estimate 
         is 2^(k+1) - 1.
         
    This heuristic is admissible because to move a specific ring, all rings smaller than it 
    must be moved off of it (and off the target), representing a sub-problem of height k.
    """

    def __init__(self, towers_gemini3pro_heuristic_1=None, name="TowersGemini3ProHeuristic2"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain knowledge storage
        self.goal_positions = {}      # ring -> target object identifier
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact_obj, target_obj_name) tuples
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by analyzing the domain structure and initial state.
        """
        start_time = time.time()
        self.model = model

        self._preprocess_domain()

        self.preprocessing_time = time.time() - start_time
        
        # Calculate heuristic for the initial state
        h_value = self._calculate_heuristic(initial_node.state)
        
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Robustly parses fact names from Pytrich's internal representation.
        Handles '+predicate[arg1,arg2]' and '(predicate arg1 arg2)'.
        """
        name = name.strip()
        
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
                
        # Format: (predicate arg1 arg2)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
                
        return None, []

    def _preprocess_domain(self):
        """
        Extracts static hierarchy and goal definitions from the model.
        """
        rings = set()
        smaller_than_relations = []

        # 1. Parse Goals
        # Goals are stored as a bitwise integer. We need to convert to explicit names to parse.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # Typically (on ?r ?o) or (goal_on ?r ?o)
                # The domain definition uses (goal_on ?r ?o) in :init but checks (on ?r ?o) in :goal
                # We check for 'on' in the goal state representation
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    # Heuristic assumption: rings usually start with 'r'
                    if ring.startswith("r") or "ring" in ring.lower():
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2. Parse Facts to build hierarchy and lookup tables
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            
            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                # (smallerThan ?smaller ?larger)
                smaller, larger = args[0], args[1]
                # We track rings involved in size relations
                if smaller.startswith("r") or "ring" in smaller.lower():
                    rings.add(smaller)
                    smaller_than_relations.append((smaller, larger))
                if larger.startswith("r") or "ring" in larger.lower():
                    rings.add(larger)

            elif predicate == "on":
                # (on ?ring ?target)
                # We need to map rings to the specific Fact objects that represent their position
                # to enable fast bitwise checks during search.
                ring, target = args[0], args[1]
                if ring.startswith("r") or "ring" in ring.lower():
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        
        if self.num_rings == 0:
            return

        # 3. Sort rings by size (Smallest -> Largest)
        # We count how many objects a ring is smaller than.
        # The ring smaller than the MOST objects is the smallest ring.
        smaller_count = defaultdict(int)
        for smaller, larger in smaller_than_relations:
            # Only count relations where the 'smaller' is a known ring
            if smaller in rings:
                smaller_count[smaller] += 1
        
        # Sort descending by count. 
        # Example: r1 < r2, r1 < r3, r2 < r3.
        # r1 count=2, r2 count=1, r3 count=0.
        # Order: r1, r2, r3.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determines the current location of a ring using bitwise operations.
        """
        if ring not in self.on_facts_by_ring:
            return None

        for fact, target in self.on_facts_by_ring[ring]:
            # Check if the bit corresponding to this fact is set in the state
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates 2^k - 1 based on the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from Largest to Smallest
        # rings_by_size is [small, ..., large]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            goal_pos = self.goal_positions.get(ring)
            
            # If we don't know the goal for a ring, we assume it's fine or irrelevant
            if not goal_pos:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            
            if current_pos != goal_pos:
                # Found the largest ring not in its goal position.
                # This ring is at index 'i' in the size list (0=smallest).
                # We need to move this ring, which requires moving all i smaller rings.
                # Total rings involved = i + 1.
                # Cost = 2^(i+1) - 1.
                k = i + 1
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for the given node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic2(rings={self.num_rings})"
