import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi (towers) domain.
    
    This heuristic implements the optimal cost estimation for the Towers of Hanoi problem.
    The minimum number of moves to solve a problem with the largest misplaced ring 'k'
    is 2^k - 1, where k is the 1-based index of the ring sorted by size (1=smallest).
    
    Features:
    1. Preprocesses ring sizes using 'smallerThan' predicates.
    2. Extracts goal positions efficiently from the bitwise goal state.
    3. Uses bitwise operations for fast state querying during search.
    4. Admissible lower bound logic: To move the k-th smallest ring, all k-1 smaller
       rings must be moved, requiring 2^(k-1)-1 moves, plus 1 move for ring k,
       plus 2^(k-1)-1 moves to stack them back.
    """

    def __init__(self, towers_gemini3pro_heuristic_8=None, name="TowersGemini3ProHeuristic9"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring -> target object (from goal state)
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model.
        """
        start_time = time.time()
        self.model = model

        # Preprocess static domain information
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles +predicate[args] and (predicate args) formats.
        """
        name = name.strip()

        # Format 1: +predicate[arg1,arg2] or -predicate[arg1,arg2]
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
        Extract ring hierarchy and goal definitions from the model.
        """
        rings = set()
        smaller_than_counts = defaultdict(int)  # Count how many objects a ring is smaller than

        # 1. Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # Look for 'on' predicates in the goal
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):  # Simple check for ring objects
                        self.goal_positions[ring] = target

        # 2. Analyze all facts to build static domain structures
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                # (smallerThan ?r ?o) - r is smaller than o
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_counts[smaller] += 1
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                # Map 'on' facts to rings to allow fast lookup of current position
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

        # 3. Sort rings by size (smallest to largest)
        # The ring that is smaller than the MOST other objects is the smallest ring.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_counts.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Identify the object the ring is currently on using bitwise operations.
        """
        if ring not in self.on_facts_by_ring:
            return None

        for fact, target in self.on_facts_by_ring[ring]:
            # Check if the bit for this fact is set in the state
            if (state_bitwise >> fact.global_id) & 1:
                return target

        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate h(n) = 2^k - 1, where k is the rank of the largest ring not at its goal.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from largest ring to smallest (reverse order of rings_by_size)
        # rings_by_size[0] is smallest, rings_by_size[len-1] is largest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]

            # We only care about rings that have a defined goal position
            if ring in self.goal_positions:
                current_pos = self._get_current_position(state_bitwise, ring)
                goal_pos = self.goal_positions[ring]

                if current_pos != goal_pos:
                    # Found the largest ring not in goal position.
                    # Rank k = i + 1 (since i is 0-indexed)
                    # Heuristic cost = 2^k - 1
                    return (1 << (i + 1)) - 1

        # All rings with goals are in position
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for a node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini3ProHeuristic9(rings={self.num_rings})"
