import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    This heuristic implements the optimal cost estimation for the Towers of Hanoi problem.
    The minimum number of moves to solve a k-disk problem is 2^k - 1.
    
    Logic:
    1. Identifies the hierarchy of rings (smallest to largest) via static analysis.
    2. Identifies the goal position for each ring.
    3. During search, finds the LARGEST ring (k) that is not in its goal position.
    4. Estimates cost as 2^k - 1, because moving the k-th ring requires moving all k-1 smaller rings.
    
    This approach is admissible and highly efficient for this specific structure.
    """

    def __init__(self, towers_gemini3pro_heuristic_13=None, name="TowersGemini3ProHeuristic14"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain structures
        self.goal_positions = {}      # ring -> target object
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize heuristic with model analysis.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses fact names from grounded (+pred[args]) or HDDL ((pred args)) formats.
        """
        name = name.strip()

        # Format 1: +predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Format 2: (predicate arg1 arg2)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        """
        Extracts ring hierarchy and goal definitions from the model.
        """
        rings = set()
        smaller_than = []

        # 1. Parse Goal State from model.goals bitmask
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # We look for on(ring, target) in goals
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    # Heuristic assumption: Rings start with 'r'
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2. Parse all facts to identify rings and size relations
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller, larger = args[0], args[1]
                # We only care about ring vs ring/tower relations to establish size
                # The problem files show smallerThan(r_small, r_large) and smallerThan(r_any, tower)
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)
            
            elif predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    # Store the fact object to access global_id later
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3. Sort rings by size (smallest to largest)
        # Logic: The smallest ring is 'smallerThan' the most other rings.
        # The largest ring is 'smallerThan' only the towers (or nothing if only ring-ring recorded).
        # We count how many RINGS a ring is smaller than.
        ring_smaller_count = defaultdict(int)
        for s, l in smaller_than:
            if s in rings and l in rings:
                ring_smaller_count[s] += 1
        
        # Sort descending: r1 is smaller than (N-1) rings -> index 0 (smallest)
        self.rings_by_size = sorted(list(rings), key=lambda r: ring_smaller_count[r], reverse=True)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determines the current object a ring is 'on' using bitwise checks.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for fact, target in self.on_facts_by_ring[ring]:
            # Fast bitwise check: (state >> id) & 1
            if (state_bitwise >> fact.global_id) & 1:
                return target
                
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates h(n) = 2^k - 1 where k is the rank of the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from largest ring to smallest ring
        # self.rings_by_size is sorted [smallest, ..., largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If ring is not part of the goal specification, we skip it (or assume it's fine)
            if ring not in self.goal_positions:
                continue

            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]

            if current_pos != goal_pos:
                # Found the largest ring that is not in its correct position.
                # Rank k = i + 1 (since list is 0-indexed)
                # Minimum cost is 2^k - 1
                return (1 << (i + 1)) - 1
        
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for a node.
        """
        h_value = self._calculate_heuristic(node.state)
        
        # Update heuristic info stats if base class supports it
        super().update_info(h_value)
        
        return h_value
