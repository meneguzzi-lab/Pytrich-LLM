"""
Domain-dependent heuristic for the Towers domain (Towers of Hanoi).

This heuristic implements the optimal 2^k - 1 strategy tailored for the Hierarchical
Towers of Hanoi domain. It identifies the largest ring that is not in its goal
position and estimates the cost based on the number of smaller rings that must be
moved to rectify it.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic4(Heuristic):
    """
    Domain-dependent heuristic for Towers of Hanoi.
    
    Calculates h(n) = 2^k - 1, where k is the 1-based rank (size) of the 
    largest ring that is not currently in its goal position.
    
    This is admissible because to move disk k to its destination, all smaller 
    disks 1..(k-1) must be moved off it and off the destination peg. This 
    recurrence relation yields the minimal move count.
    """

    def __init__(self, towers_gemini3pro_heuristic_3=None, name="TowersGemini3ProHeuristic4"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Domain maps
        self.goal_positions = {}      # ring -> target_obj
        self.rings_by_size = []       # Ordered list: [smallest_ring, ..., largest_ring]
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj)
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Precompute static domain structures from the model facts and goals.
        """
        start_time = time.time()
        self.model = model
        
        self._preprocess_domain()
        
        self.preprocessing_time = time.time() - start_time
        
        # Compute for initial state
        h_val = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        """
        Parses fact names in format '+predicate[arg1,arg2]' or '(predicate arg1 arg2)'.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess_domain(self):
        """
        Extracts ring hierarchy and goal definitions.
        """
        rings = set()
        smaller_counts = defaultdict(int)

        # 1. Analyze Facts to find rings and size relationships
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue
                
            if predicate == "smallerThan":
                # smallerThan r1 r2 -> r1 is smaller than r2
                # The ring that is smaller than the MOST other things is the smallest ring.
                if len(args) >= 2:
                    smaller = args[0]
                    if smaller.startswith("r"):
                        rings.add(smaller)
                        smaller_counts[smaller] += 1
                        
            elif predicate == "on":
                # Map ring to its 'on' facts for quick state lookup
                if len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        rings.add(ring)
                        if ring not in self.on_facts_by_ring:
                            self.on_facts_by_ring[ring] = []
                        self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        
        # 2. Sort rings by size (smallest to largest)
        # A ring smaller than many things (high count) is smaller physically.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_counts.get(r, 0))

        # 3. Extract Goals from model.goals (bitwise mask)
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                predicate, args = self._parse_fact_name(g_name)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    if ring in rings:
                        self.goal_positions[ring] = target

    def _get_current_position(self, state_bitwise: int, ring: str):
        """
        Returns the object the ring is currently on, using bitwise state check.
        """
        facts = self.on_facts_by_ring.get(ring)
        if not facts:
            return None
            
        for fact, target in facts:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Finds largest misplaced ring r_k (0-indexed k) and returns 2^(k+1) - 1.
        """
        if not self.rings_by_size:
            return 0
            
        # Iterate from largest ring down to smallest
        # self.rings_by_size is [smallest, ..., largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If this ring has no specific goal position, we assume it's fine unless
            # a smaller ring needs to move. But strict Hanoi usually targets specific configs.
            if ring not in self.goal_positions:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]
            
            # If the ring is not on its goal target, it's the bottleneck.
            if current_pos != goal_pos:
                # k = i + 1 (size rank)
                # Cost is 2^k - 1
                # This ring and all i smaller rings must participate in the move sequence.
                return (1 << (i + 1)) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Heuristic evaluation called during search.
        """
        h_val = self._calculate_heuristic(node.state)
        super().update_info(h_val)
        return h_val
"