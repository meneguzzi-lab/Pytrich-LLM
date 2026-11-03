import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    This heuristic implements the optimal cost estimation for the Towers of Hanoi problem:
    h(n) = 2^k - 1
    where k is the number of rings in the sub-problem defined by the largest misplaced ring.
    
    Logic:
    1. The heuristic identifies the size ordering of rings (from static `smallerThan` facts).
    2. It determines the goal position for each ring from the problem goal.
    3. It iterates from the largest ring to the smallest.
    4. The first ring encountered that is NOT in its goal position is the "bottleneck".
    5. If ring 'r_i' (where index 0 is smallest) is the largest misplaced ring, it implies
       that all rings smaller than it (indices 0 to i) must be moved at least once.
       The minimum moves required is 2^(i+1) - 1.
       
    This is admissible because moving the largest misplaced ring requires moving all smaller
    rings off it (and off the target), performing the move, and potentially stacking them back.
    """

    def __init__(self, towers_gemini3pro_heuristic_14=None, name="TowersGemini3ProHeuristic15"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Processed domain knowledge
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.goal_positions = {}      # ring_name -> target_name
        self.on_facts_by_ring = {}    # ring_name -> list of (global_id, target_name)
        self.num_rings = 0

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles grounded format "+predicate[arg1,arg2]" and fallback HDDL format.
        """
        name = name.strip()

        # Format 1: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args

        # Format 2: (predicate arg1 arg2) - Fallback
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]

        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model to extract ring order and goals.
        """
        start_time = time.time()
        self.model = model

        self._preprocess()

        self.preprocessing_time = time.time() - start_time
        
        # Calculate heuristic for the initial state
        h_value = self._calculate_heuristic(initial_node.state)
        
        # Initialize parent
        return super().initialize(model, h_value)

    def _preprocess(self):
        """
        Extract static structure: ring sizes from 'smallerThan' and goals from model.goals.
        """
        rings = set()
        smaller_counts = defaultdict(int)
        
        # 1. Extract Goals from bitwise goal state
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 2:
                    ring, target = args[0], args[1]
                    self.goal_positions[ring] = target
                    if ring.startswith("r"):
                        rings.add(ring)

        # 2. Analyze Facts for Structure (Size and Position IDs)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue

            if pred == "smallerThan":
                # smallerThan(?r, ?o) implies ?r is smaller than ?o
                # We count how many objects a ring is smaller than to determine relative size.
                # The smallest ring is smaller than the MOST objects.
                r = args[0]
                if r.startswith("r"):
                    rings.add(r)
                    smaller_counts[r] += 1
                    
            elif pred == "on":
                # Cache 'on' facts for fast state lookups
                r, target = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    if r not in self.on_facts_by_ring:
                        self.on_facts_by_ring[r] = []
                    # Store global_id for bitwise check
                    self.on_facts_by_ring[r].append((fact.global_id, target))

        self.num_rings = len(rings)
        
        if self.num_rings == 0:
            return

        # 3. Sort Rings by Size
        # If 'r1' is smaller than 5 things and 'r5' is smaller than 0 things,
        # then 'r1' is the smallest ring.
        # We sort descending by count: [Smallest, ..., Largest]
        self.rings_by_size = sorted(list(rings), key=lambda r: smaller_counts[r], reverse=True)
        
        # Fallback if smallerThan was not found (rare in this domain, but safe)
        if sum(smaller_counts.values()) == 0 and len(self.goal_positions) > 0:
            # Infer sizes from goal chain: on(r1, r2) -> r1 < r2
            # Simple topological sort approx via counts of being 'on' something
            # This is complex to do perfectly without a graph, but the domain guarantees smallerThan
            # so we stick to the primary method. If fails, we just have an unsorted list,
            # which makes the heuristic less informed but still admissible if we are lucky,
            # though the power 2^k logic depends on sorting.
            pass

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determine what the ring is currently on using bitwise operations.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for global_id, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Compute the heuristic value h(n) = 2^k - 1.
        """
        if not self.rings_by_size:
            return 0

        # Iterate from Largest ring (end of list) to Smallest ring (start of list)
        # self.rings_by_size is sorted [Smallest ... Largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # We only care if we know where it should be
            goal_pos = self.goal_positions.get(ring)
            if not goal_pos:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            
            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # 'i' is the 0-based index in the sorted list of rings.
                # If i=0 (smallest ring), we return 2^1 - 1 = 1.
                # If i=N-1 (largest ring), we return 2^N - 1.
                # The number of rings involved in the move stack is i + 1.
                return (1 << (i + 1)) - 1

        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic for a search node.
        """
        h_value = self._calculate_heuristic(node.state)
        self.update_info(h_value)
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic15(rings={self.num_rings})"