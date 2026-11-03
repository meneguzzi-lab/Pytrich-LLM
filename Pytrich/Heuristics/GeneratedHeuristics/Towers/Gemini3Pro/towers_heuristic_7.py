import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic7(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    This heuristic implements the optimal solution cost estimation for the Towers of Hanoi problem.
    It identifies the largest ring that is not in its goal position and estimates the cost
    as 2^k - 1, where k is the rank of that ring (1-based index from smallest).
    
    This approach is admissible because to move the k-th ring to its correct destination,
    all smaller k-1 rings must be moved off of it (and off the destination), which requires
    solving the sub-problem of size k-1.
    """

    def __init__(self, towers_gemini3pro_heuristic_6=None, name="TowersGemini3ProHeuristic7"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.towers_gemini3pro_heuristic_6 = towers_gemini3pro_heuristic_6
        
        # Domain structures
        self.goal_positions = {}      # ring -> target object name
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (Fact object, target name)
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by analyzing the domain structure and goals.
        This preprocessing is performed once before the search begins.
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
        Parses a fact name string into a (predicate, args) tuple.
        Handles both Grounded '+pred[a,b]' and HDDL '(pred a b)' formats.
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

    def _preprocess_domain(self):
        """
        Extracts static domain info: rings, size ordering, and goal configurations.
        """
        rings = set()
        smaller_than_relations = []

        # 1. Parse Goals from bitwise goal state
        # The goal state is a bitmask. We convert it to explicit strings to find target positions.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                # Goal is usually (on ?ring ?target)
                if predicate == "on" and len(args) == 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2. Parse all facts to identify rings, size relations, and 'on' predicates
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == "smallerThan":
                # (smallerThan ?smaller ?larger)
                smaller, larger = args[0], args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    smaller_than_relations.append((smaller, larger))
                    # If the larger obj is also a ring, add it
                    if larger.startswith("r"):
                        rings.add(larger)
            
            elif predicate == "on":
                # Store 'on' facts to quickly lookup current position during search
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3. Sort rings by size (Smallest -> Largest)
        # We count how many objects a ring is smaller than.
        # The smallest ring is smaller than the most objects.
        smaller_count = defaultdict(int)
        for smaller, _ in smaller_than_relations:
            if smaller in rings:
                smaller_count[smaller] += 1
        
        # Sort descending by count (Smallest ring has highest count)
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_count.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determines the object a ring is currently on using bitwise operations.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for fact, target in self.on_facts_by_ring[ring]:
            # Check if bit is set in state integer
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates the heuristic value.
        Algorithm: Find the largest ring that is NOT on its goal target.
        If ring at index k (0-based) is misplaced, it implies a sub-problem of size k+1.
        Cost = 2^(k+1) - 1.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from Largest to Smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If this ring has no defined goal, skip it (assume satisfied or irrelevant)
            if ring not in self.goal_positions:
                continue

            goal_target = self.goal_positions[ring]
            current_target = self._get_current_position(state_bitwise, ring)

            if current_target != goal_target:
                # This is the largest misplaced ring.
                # Number of rings involved in the subproblem is i + 1.
                # Heuristic cost is 2^(i+1) - 1.
                n = i + 1
                return (1 << n) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Called during search to evaluate a node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini3ProHeuristic7(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGemini3ProHeuristic7 with {self.num_rings} rings"
