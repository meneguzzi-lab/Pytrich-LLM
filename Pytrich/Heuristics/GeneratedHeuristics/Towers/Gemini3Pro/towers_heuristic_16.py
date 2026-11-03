import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic16(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    This heuristic calculates the classic 2^k - 1 lower bound for the Towers of Hanoi problem.
    It identifies the largest ring k that is not in its goal position and returns 2^k - 1,
    which represents the minimum number of moves required to move that ring (assuming all
    smaller rings must be moved off it and then back onto it).
    """

    def __init__(self, towers_gemini3pro_heuristic_15=None, name="TowersGemini3ProHeuristic16"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain knowledge extracted during preprocessing
        self.goal_positions = {}      # ring -> target object
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model to extract ring sizes and goals.
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
        Handles both +predicate[args] and (predicate args) formats.
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

    def _preprocess(self):
        """
        Extract domain structure: rings, their size ordering, and goal configurations.
        """
        rings = set()
        smaller_than = []  # List of (smaller_ring, larger_obj) pairs

        # Extract goal positions from model.goals (bitwise integer)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # We look for 'on' predicates in the goal state
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # Iterate through all static facts to find structure
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
                # Pre-calculate mapping from ring to the specific 'on' facts that describe its position
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

        # Sort rings by size (smallest to largest)
        # Method 1: Use smallerThan facts
        if smaller_than:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than:
                if smaller in rings:
                    smaller_count[smaller] += 1
            # The more things a ring is smaller than, the smaller it is. 
            # However, smaller_count gives # of things > ring.
            # So higher count = smaller ring.
            self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))
        else:
            # Method 2: Infer from goal topology if smallerThan is missing
            # If r1 is on r2, r1 < r2.
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
        Calculate h(n) = 2^k - 1, where k is the size-index of the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from largest ring to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]

            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions.get(ring)

            # If goal_pos is defined and ring is not there
            if goal_pos and current_pos != goal_pos:
                # This ring needs to move. In Towers of Hanoi, moving the k-th ring
                # implies moving all k-1 smaller rings. Minimum moves = 2^k - 1.
                # Here k is (i + 1) because i is 0-indexed.
                num_rings_involved = i + 1
                return (1 << num_rings_involved) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for the new node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic16 with {self.num_rings} rings"
