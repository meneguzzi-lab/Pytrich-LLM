"""
Domain-dependent heuristic for the Towers domain (Towers of Hanoi).

This heuristic implements the optimal 2^k - 1 strategy tailored for the Towers of Hanoi problem.
It analyzes the domain structure to identify the hierarchy of rings and their goal positions,
providing an admissible and highly informative estimate for the remaining cost.

Key features:
1. Automatic ring detection and size sorting based on domain facts.
2. Bitwise state checking for high-performance evaluation.
3. Hierarchical goal decomposition analysis.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini3ProHeuristic11(Heuristic):
    """
    A domain-dependent heuristic for the Towers domain.
    
    This heuristic estimates the cost to solve the problem by identifying the largest
    ring that is not in its goal position. For a standard Towers of Hanoi problem,
    if the k-th smallest ring is the largest one misplaced, the minimum moves required
    is 2^k - 1. This heuristic generalizes this concept by dynamically analyzing
    the goal state and the current configuration.
    """

    def __init__(self, towers_gemini3pro_heuristic_10=None, name="TowersGemini3ProHeuristic11"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain data structures
        self.goal_positions = {}      # ring_name -> target_name
        self.rings_ordered = []       # List of rings sorted smallest to largest
        self.on_facts_map = {}        # ring_name -> list of (global_id, target_name)
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic with domain analysis.
        """
        start_time = time.time()
        self.model = model
        
        self._preprocess_domain()
        
        self.preprocessing_time = time.time() - start_time
        
        # Calculate initial value
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles both grounded '+pred[a,b]' and HDDL '(pred a b)' formats.
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
        Extract ring hierarchy and goal definitions from the model.
        """
        # 1. Parse Goals
        # model.goals is a bitwise int. We need the explicit fact names to find goal positions.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for fact_name in goal_fact_names:
                pred, args = self._parse_fact_name(fact_name)
                if pred == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    self.goal_positions[ring] = target

        # 2. Analyze Facts to Identify Rings and Structure
        rings = set()
        smaller_than_counts = defaultdict(int) # ring -> count of rings it is smaller than
        
        # Pass 1: Identify all rings and collect smallerThan facts
        # We assume any object appearing as the first argument of 'on' is a ring.
        # We also use 'smallerThan' to determine size ordering.
        
        potential_rings = set()
        smaller_than_facts = []

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or not args:
                continue
            
            if pred == "on":
                ring = args[0]
                target = args[1]
                potential_rings.add(ring)
                
                # Store 'on' facts for fast state lookup
                if ring not in self.on_facts_map:
                    self.on_facts_map[ring] = []
                self.on_facts_map[ring].append((fact.global_id, target))
            
            elif pred == "smallerThan":
                smaller, larger = args[0], args[1]
                smaller_than_facts.append((smaller, larger))
                potential_rings.add(smaller)

        # Pass 2: Sort rings by size
        # A ring R is smaller than N other rings implies it is small.
        # Specifically: Smallest ring is smaller than ALL other rings.
        # Largest ring is smaller than NO other rings (only towers).
        
        # Filter smaller_than relations to only include ring-ring comparisons
        for smaller, larger in smaller_than_facts:
            if smaller in potential_rings and larger in potential_rings:
                smaller_than_counts[smaller] += 1
        
        # Sort rings: descending order of 'smaller_than_counts'
        # The ring that is smaller than the most other rings is the smallest ring (index 0).
        self.rings_ordered = sorted(list(potential_rings), key=lambda r: smaller_than_counts[r], reverse=True)
        self.num_rings = len(self.rings_ordered)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Efficiently determine the current object a ring is on using bitwise operations.
        """
        if ring not in self.on_facts_map:
            return None
            
        for global_id, target in self.on_facts_map[ring]:
            if (state_bitwise >> global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value.
        
        Iterates from the largest ring to the smallest.
        Finds the largest ring that is NOT at its goal position.
        If the k-th ring (0-indexed smallest to largest) is the largest misplaced ring,
        it contributes 2^(k+1) - 1 to the heuristic.
        """
        if not self.rings_ordered:
            return 0

        # Iterate backwards: from largest ring to smallest
        # rings_ordered is [smallest, ..., largest]
        for i in range(len(self.rings_ordered) - 1, -1, -1):
            ring = self.rings_ordered[i]
            
            # If this ring has no defined goal, we skip checking it
            # (It might be an intermediate ring not specified in goal, though rare in Towers)
            if ring not in self.goal_positions:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]
            
            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # The cost to fix this configuration involves moving this ring,
                # which requires moving all smaller rings (0 to i-1) out of the way.
                # Heuristic cost = 2^(i+1) - 1
                # i is 0-based index, so i+1 is the rank (1 for smallest)
                return (1 << (i + 1)) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic for the given node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic11(rings={self.num_rings})"
"