from collections import defaultdict
import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic estimates the cost to solve the Towers of Hanoi problem by identifying
    the largest ring that is not in its goal position. Based on the optimal solution
    properties of the problem, if the k-th smallest ring needs to be moved to reach
    the goal state (and by definition all rings smaller than it must also move),
    the minimum number of moves required is 2^k - 1.

    The heuristic:
    1. Preprocesses the domain to establish ring sizes and goal dependencies.
    2. For a given state, identifies the largest ring 'k' that is misplaced relative to the goal.
    3. Returns 2^k - 1.
    """

    def __init__(self, towers_gemini3pro_heuristic_2=None, name="TowersGemini3ProHeuristic3"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain structures
        self.goal_positions = {}      # ring -> target_object
        self.rings_by_size = []       # List of rings sorted smallest (index 0) to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_object)
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by analyzing the domain structure and goal.
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
        Parses a fact name string into a (predicate, args) tuple.
        Handles both grounded format '+pred[arg1,arg2]' and HDDL format '(pred arg1 arg2)'.
        """
        name = name.strip()
        
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # strip +/- 
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

    def _preprocess(self):
        """
        Extracts static domain info: ring hierarchy and goal definitions.
        """
        rings = set()
        smaller_than_counts = defaultdict(int) # ring -> count of things it is smaller than
        
        # 1. Extract Goal Configuration
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    # We only care about ring positions in the goal
                    if ring.startswith("r") or "ring" in ring.lower(): 
                        self.goal_positions[ring] = target
                        rings.add(ring)
                        if target.startswith("r") or "ring" in target.lower():
                            rings.add(target)

        # 2. Extract Static Domain Hierarchy (smallerThan relations) and Fact Maps
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller = args[0]
                larger = args[1]
                # Detect rings based on naming convention or usage in smallerThan
                # Usually args[0] is a ring. args[1] can be ring or tower.
                if smaller.startswith("r") or "ring" in smaller.lower():
                    rings.add(smaller)
                    smaller_than_counts[smaller] += 1
                    if (larger.startswith("r") or "ring" in larger.lower()) and larger not in rings:
                        rings.add(larger)

            elif predicate == "on":
                # Map on facts to rings for fast state checking
                ring = args[0]
                target = args[1]
                if ring.startswith("r") or "ring" in ring.lower():
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        
        # 3. Sort Rings by Size
        # We sort based on 'smaller_than_counts'. The smallest ring is smaller than the most objects.
        # So we sort descending by count.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_counts.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determines the current support of a ring using bitwise operations.
        Returns the name of the object the ring is on, or None.
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
        Calculates 2^k - 1 where k is the rank of the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from largest ring to smallest ring
        # rings_by_size is [smallest, ..., largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]

            # If this ring has no specific goal position, we assume it's fine where it is
            # unless a larger ring needs it to move (handled by outer loops logic).
            if ring not in self.goal_positions:
                continue

            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]

            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # rank is i + 1 (since i is 0-indexed)
                # Heuristic cost = 2^(rank) - 1
                # This corresponds to moving this ring and all smaller rings.
                return (1 << (i + 1)) - 1

        # All rings with goals are in position
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini3ProHeuristic3(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGemini3ProHeuristic3 with {self.num_rings} rings"
