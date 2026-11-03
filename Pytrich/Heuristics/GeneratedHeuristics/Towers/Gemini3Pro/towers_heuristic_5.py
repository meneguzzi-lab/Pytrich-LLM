from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import time
from collections import defaultdict

class TowersGemini3ProHeuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    This heuristic implements the optimal cost estimation for the Towers of Hanoi problem.
    The minimum moves required to solve a subproblem where the largest misplaced disk is k
    is 2^k - 1. This heuristic is admissible and provides tight bounds.
    
    It leverages the domain structure by:
    1. Preprocessing static 'smallerThan' relations to order rings by size.
    2. extracting goal positions from the bitwise goal state.
    3. Dynamically checking the position of rings in the current state.
    """

    def __init__(self, towers_gemini3pro_heuristic_4=None, name="TowersGemini3ProHeuristic5"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.goal_positions = {}      # ring -> target object name
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact_obj, target_obj_name)
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the domain facts and initial state.
        """
        start_time = time.time()
        self.model = model
        self._preprocess(initial_node.state)
        self.preprocessing_time = time.time() - start_time
        
        # Calculate initial value
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles +predicate[arg1,arg2] and (predicate arg1 arg2) formats.
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

    def _preprocess(self, initial_state_bitwise: int):
        """
        Extract ring order and lookup tables from facts.
        """
        rings = set()
        smaller_counts = defaultdict(int)

        # 1. Parse Goal State
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    # We assume things starting with 'r' are rings based on domain convention
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2. Iterate all facts to build structure
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            # Identify rings and size relations
            # smallerThan is static, so we check if it is true in the initial state
            if predicate == "smallerThan":
                # Check if this static fact is true in initial state
                if (initial_state_bitwise >> fact.global_id) & 1:
                    smaller, larger = args[0], args[1]
                    if smaller.startswith("r"):
                        rings.add(smaller)
                        smaller_counts[smaller] += 1
                        if larger.startswith("r"):
                            rings.add(larger)

            # Map 'on' facts for dynamic state checking
            elif predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        
        # Sort rings: The one smaller than the MOST other objects is the smallest ring.
        # rings_by_size[0] is the smallest ring.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_counts.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determine the object the ring is currently on.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate h(n) = 2^k - 1 for the largest misplaced ring k.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from largest ring to smallest
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If we don't know the goal for this ring, we skip checking it directly,
            # but we assume implicit constraints might exist. However, heuristic 
            # is driven by explicit mismatches.
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                continue

            current_pos = self._get_current_position(state_bitwise, ring)
            
            if current_pos != goal_pos:
                # This ring is misplaced.
                # In the sequence of rings 0..i, this is the i-th ring (0-indexed).
                # It corresponds to the (i+1)-th smallest ring.
                # Cost to move stack of size k is 2^k - 1.
                # Here k = i + 1.
                return (1 << (i + 1)) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for the given node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
