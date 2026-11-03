import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Towers domain (Towers of Hanoi).

    This heuristic leverages the optimal solution property of the Towers of Hanoi problem.
    The minimum number of moves to solve a sub-problem of size k is 2^k - 1.

    Strategy:
    1. Identifies the size ordering of rings based on 'smallerThan' predicates.
    2. Identifies the goal position for each ring from the goal state bitmask.
    3. During search, finds the largest ring that is NOT in its goal position.
    4. Returns 2^(k) - 1, where k corresponds to the size index of that ring (1-based).
       This accounts for the necessity of moving all smaller rings to access and move the largest misplaced ring.

    This is admissible because moving the largest misplaced ring requires strictly solving the
    recursion for all smaller rings.
    """

    def __init__(self, towers_gemini3pro_heuristic_9=None, name="TowersGemini3ProHeuristic10"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.goal_positions = {}      # ring -> target object identifier
        self.rings_by_size = []       # List of rings sorted from smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact_object, target_name) tuples
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing static domain info and parsing goals.
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
        Parse a fact name into (predicate, args) tuple.
        Handles grounded format "+pred[arg1,arg2]" and HDDL format "(pred arg1 arg2)".
        """
        name = name.strip()

        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
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
        Extract ring hierarchy and goal definitions from the model.
        """
        rings = set()
        smaller_than_count = defaultdict(int)

        # 1. Extract goal positions from model.goals (bitwise int)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # Look for (on ?r ?o) or +on[?r,?o]
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2. Extract static structure (smallerThan) and dynamic fact mappings (on)
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                # (smallerThan r1 r2) means r1 is smaller than r2.
                # We count how many things a ring is smaller than to determine size rank.
                # The smallest ring is smaller than everything else.
                smaller = args[0]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_count[smaller] += 1
                    # Ensure the larger object is tracked if it is a ring
                    if args[1].startswith("r"):
                        rings.add(args[1])

            elif predicate == "on":
                # Map 'on' facts to rings for fast state checking
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

        # Sort rings: The ring smaller than the MOST objects is the smallest ring (index 0).
        # Example: r1 < r2 < r3. r1 count=2, r2 count=1, r3 count=0.
        # List becomes [r1, r2, r3].
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determine the current object the ring is 'on' using bitwise state check.
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
        Calculate h(n) = 2^k - 1, where k is the 1-based index of the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from the largest ring down to the smallest
        # rings_by_size is [smallest, ..., largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If this ring has no specific goal, skip it
            goal_pos = self.goal_positions.get(ring)
            if not goal_pos:
                continue

            current_pos = self._get_current_position(state_bitwise, ring)

            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # 'i' is the 0-based index. If i=0 (smallest ring), we need 2^1 - 1 = 1 move.
                # If i=k, we need 2^(k+1) - 1 moves.
                num_rings_involved = i + 1
                return (1 << num_rings_involved) - 1

        # All rings with goals are in correct positions
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic value for the current node.
        """
        h_value = self._calculate_heuristic(node.state)
        
        # Try to update stats if the base class supports it
        try:
            super().update_info(h_value)
        except AttributeError:
            pass
            
        return h_value

    def __repr__(self):
        return f"TowersGemini3ProHeuristic10(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGemini3ProHeuristic10 with {self.num_rings} rings"
