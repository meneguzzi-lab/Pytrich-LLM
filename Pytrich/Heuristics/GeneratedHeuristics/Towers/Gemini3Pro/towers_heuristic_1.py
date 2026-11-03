import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi (towers) domain.
    
    This heuristic implements the optimal solution cost calculation for the Towers of Hanoi problem.
    For a state where the largest ring k that is not in its goal position is identified,
    the minimum number of moves required is 2^k - 1, where k is the 1-based rank of the ring
    (1 being the smallest).
    
    The heuristic:
    1. Preprocesses the domain to identify rings, their size ordering, and goal positions.
    2. At runtime, checks ring positions using efficient bitwise operations.
    3. Returns (2^rank - 1) for the largest misplaced ring.
    """

    def __init__(self, towers_gemini3pro_heuristic_0=None, name="TowersGemini3ProHeuristic1"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring -> target object (from goal_on facts)
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0

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
        Handles both Grounded format (+pred[arg]) and HDDL format ((pred arg)).
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
        Extract domain structure: identify rings, sort them by size, and map 'on' facts.
        """
        rings = set()
        smaller_than = []  # List of (smaller_ring, larger_obj) pairs

        # 1. Extract goal positions from model.goals bitmask
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2. Iterate all facts to build static knowledge
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
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
        if smaller_than:
            # Count how many things each ring is smaller than
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than:
                if smaller in rings:
                    smaller_count[smaller] += 1
            # The more things a ring is smaller than, the smaller it is.
            # We want smallest first (highest count).
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Fallback: Infer from goals if explicit smallerThan is missing
            is_smaller_than = defaultdict(set)
            for ring, target in self.goal_positions.items():
                if target.startswith("r"):
                    is_smaller_than[ring].add(target)
            
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for ring in list(is_smaller_than.keys()):
                    for larger in list(is_smaller_than[ring]):
                        for even_larger in is_smaller_than.get(larger, set()):
                            if even_larger not in is_smaller_than[ring]:
                                is_smaller_than[ring].add(even_larger)
                                changed = True
            
            smaller_count = {r: len(is_smaller_than.get(r, set())) for r in rings}
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Find where a ring currently is by checking which 'on' fact is true.
        """
        if ring not in self.on_facts_by_ring:
            return None

        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target

        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate h(n) = 2^k - 1, where k is the index of the largest misplaced ring.
        
        Algorithm:
        1. Iterate rings from largest to smallest.
        2. If a ring is not at its goal position, it (and all smaller rings above it)
           must be moved. The cost is equivalent to solving a k-disk Tower of Hanoi problem.
        3. If a ring IS at its goal position, it effectively acts as a base for the smaller rings.
        """
        if not self.rings_by_size:
            return 0

        # Iterate reverse: largest to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]

            # If we don't know the goal for this ring, we can't judge it misplaced
            goal_pos = self.goal_positions.get(ring)
            if not goal_pos:
                continue

            current_pos = self._get_current_position(state_bitwise, ring)
            
            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # i is 0-based index. If i=0 (smallest), we need 1 move (2^1 - 1).
                # If i=k, we have k+1 rings involved.
                num_rings_to_move = i + 1
                return (1 << num_rings_to_move) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini3ProHeuristic1(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGemini3ProHeuristic1 with {self.num_rings} rings"
