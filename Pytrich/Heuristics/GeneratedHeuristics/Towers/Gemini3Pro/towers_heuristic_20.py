import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic20(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    This heuristic implements the optimal 2^k - 1 strategy for Towers of Hanoi.
    It estimates the cost to goal by identifying the largest ring that is not
    in its goal position. If the k-th smallest ring is the largest misplaced ring,
    it assumes a subproblem of size k must be solved, requiring 2^k - 1 moves.
    
    This approach effectively leverages the recursive structure of the domain
    where moving a larger disk requires moving all smaller disks out of the way.
    """

    def __init__(self, towers_gemini3pro_heuristic_19=None, name="TowersGemini3ProHeuristic20"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain knowledge structures
        self.goal_positions = {}      # ring -> target object
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize heuristic with domain analysis.
        """
        start_time = time.time()
        self.model = model

        # Perform static analysis
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial h-value
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse fact name into (predicate, args).
        Handles grounded format: +predicate[arg1,arg2]
        Handles standard format: (predicate arg1 arg2)
        """
        name = name.strip()

        # Handle grounded format: +predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove sign
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Handle standard format: (predicate arg1 arg2)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def _preprocess(self):
        """
        Analyze the domain to determine ring hierarchy and goal states.
        """
        rings = set()
        smaller_than_counts = defaultdict(int)

        # 1. Extract Goal Configuration from model.goals
        # model.goals is a bitwise integer. We need the explicit representation.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # In Towers domain, goals are typically (on ring target)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2. Analyze Facts to determine ring sizes and track positions
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == "smallerThan":
                # smallerThan(?r, ?o) implies ?r < ?o
                # The smallest ring is smaller than EVERYTHING else.
                # So we count how many things each ring is smaller than.
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_counts[smaller] += 1
                    # If larger is a ring, add it to set (if larger is tower, ignore for set)
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                # Map rings to the specific facts that define their position
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # Sort rings by size (smallest to largest).
        # The ring with the highest 'smaller_than_counts' is the smallest ring.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_counts.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determine the current object the ring is on using bitwise operations.
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
        Calculate 2^k - 1 heuristic.
        
        Scans rings from Largest to Smallest.
        Finds the largest ring that is NOT at its goal position.
        If that ring has rank k (1-based index from smallest), returns 2^k - 1.
        """
        if not self.rings_by_size:
            return 0

        # Iterate backwards: Largest -> Smallest
        # i is 0-based index. Rank k = i + 1.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]

            # If ring has no defined goal, ignore it
            if ring not in self.goal_positions:
                continue

            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]

            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # All rings smaller than this (indices 0 to i-1) plus this ring (index i)
                # are involved in the subproblem. Total count = i + 1.
                k = i + 1
                return (1 << k) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Evaluate the heuristic for a given node.
        """
        h_value = self._calculate_heuristic(node.state)
        
        # Attempt to update info in base class if supported
        try:
            super().update_info(h_value)
        except AttributeError:
            pass
            
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic20 with {self.num_rings} rings"
