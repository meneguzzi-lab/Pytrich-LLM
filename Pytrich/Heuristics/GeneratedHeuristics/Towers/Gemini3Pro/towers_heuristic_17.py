import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    This heuristic implements the optimal cost estimation for the Towers of Hanoi problem:
    h(n) = 2^k - 1
    where k is the size of the largest ring that is not in its goal position.
    
    Rationale:
    1. In the Towers domain, to move a ring 'k' (where 1 is smallest), all smaller rings
       1 to k-1 must be cleared from both the source and target pegs.
    2. This recursive structure implies that if ring k is misplaced, at least 2^k - 1 moves
       are required to effectively move the stack of height k to the target.
    3. By finding the largest misplaced ring, we find the dominant term in the cost-to-go.
    4. This is admissible because the constraint hierarchy (smallerThan) enforces this
       ordering of moves.

    The heuristic efficiently:
    - Preprocesses the domain to order rings by size using 'smallerThan' facts.
    - Maps 'on' predicates to bitwise global IDs for O(1) state checks.
    - Extracts goal states from the bitwise model.goals.
    """

    def __init__(self, towers_gemini3pro_heuristic_16=None, name="TowersGemini3ProHeuristic17"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain data structures
        self.rings_by_size = []       # List[str], sorted smallest to largest
        self.goal_positions = {}      # ring -> target_object
        self.on_facts_by_ring = {}    # ring -> list of (Fact, target_object)
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic with domain analysis.
        Executed once before search begins.
        """
        start_time = time.time()
        self.model = model

        self._preprocess_domain()

        self.preprocessing_time = time.time() - start_time
        
        # Initial heuristic calculation
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses Pytrich grounded fact names.
        Expected format: +predicate[arg1,arg2]
        """
        name = name.strip()
        
        # Handle +predicate[args] format
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
                
        # Fallback/Safety for standard format (predicate arg1)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
                
        return None, []

    def _preprocess_domain(self):
        """
        Extracts static domain structure: ring ordering and goal mapping.
        """
        rings = set()
        smaller_counts = defaultdict(int)

        # 1. Analyze Facts to find Rings and Hierarchy
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            # 'smallerThan' determines size. 
            # smallerThan[r1, r2] -> r1 is smaller. 
            # Count how many things a ring is smaller than to sort them.
            if predicate == "smallerThan":
                small_obj = args[0]
                if small_obj.startswith("r"): # It's a ring
                    rings.add(small_obj)
                    smaller_counts[small_obj] += 1
                    
            # Map 'on' facts for fast state lookup
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

        # 2. Sort Rings: Smallest -> Largest
        # The ring smaller than the MOST objects is the smallest ring.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_counts[r])

        # 3. Extract Goals
        # Goals are stored as a bitwise integer in model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 2:
                    ring, target = args[0], args[1]
                    if ring in rings:
                        self.goal_positions[ring] = target

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determines the current object a ring is 'on'.
        Uses bitwise operations for speed.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for fact, target in self.on_facts_by_ring[ring]:
            # Check if the bit for this fact is set
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates h(n) = 2^k - 1 for the largest misplaced ring k.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from Largest Ring down to Smallest Ring
        # rings_by_size is [smallest, ..., largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If this ring has no specific goal position, we can't judge it misplaced
            if ring not in self.goal_positions:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]
            
            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # The size index k is i + 1 (since i is 0-indexed)
                k = i + 1
                return (1 << k) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Returns the heuristic value for the given node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic17(rings={self.num_rings})"