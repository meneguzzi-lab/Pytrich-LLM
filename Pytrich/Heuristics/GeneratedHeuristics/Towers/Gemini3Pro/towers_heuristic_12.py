"""
Domain-dependent heuristic for the Towers of Hanoi (towers) domain.
Designed for the Pytrich Planner framework.
"""

import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini3ProHeuristic12(Heuristic):
    """
    Domain-dependent heuristic for Towers of Hanoi.
    
    This heuristic implements the optimal solution cost estimate (2^k - 1) for the 
    Towers of Hanoi problem. It identifies the largest ring that is not in its 
    goal position and calculates the minimum number of moves required to place it 
    and all smaller rings correctly.

    The heuristic is admissible because to move a ring k, all k-1 smaller rings 
    must be moved off it, and they must be moved back after ring k is placed.
    """

    def __init__(self, towers_gemini3pro_heuristic_11=None, name="TowersGemini3ProHeuristic12"):
        super().__init__(name=name)
        self.preprocessing_time = 0

        # Domain knowledge structures
        self.goal_positions = {}      # ring -> target object name
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact_global_id, target_obj_name)
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize the heuristic by preprocessing the model to extract static domain info.
        This is called once before the search begins.
        """
        start_time = time.time()
        self.model = model

        # Preprocess domain structure
        self._preprocess()

        self.preprocessing_time = time.time() - start_time

        # Calculate heuristic for the initial state
        h_value = self._calculate_heuristic(initial_node.state)

        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse a fact name into (predicate, args) tuple.
        Handles both grounded formats (+pred[arg]) and HDDL formats ((pred arg)).
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
        Extract domain structure from facts and goals.
        Identifies rings, their size ordering, and goal configurations.
        """
        rings = set()
        smaller_than_counts = defaultdict(int)

        # 1. Extract goal positions from model.goals bitmask
        # This ensures we target the actual problem goal, not just any valid state.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring = args[0]
                    target = args[1]
                    # Identify rings by 'r' prefix convention in this domain
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2. Analyze all model facts to build lookup tables and size ordering
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)

            if predicate is None or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller = args[0]
                larger = args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    # Count how many things this ring is smaller than.
                    # The smallest ring is smaller than the most other objects.
                    smaller_than_counts[smaller] += 1
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                ring = args[0]
                target = args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    # Optimization: Store global_id for fast bitwise checks
                    self.on_facts_by_ring[ring].append((fact.global_id, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3. Sort rings by size (Smallest -> Largest)
        # Ring with highest 'smallerThan' count is smallest.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_counts.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Find the object a ring is currently on using bitwise operations.
        Returns None if position cannot be determined.
        """
        if ring not in self.on_facts_by_ring:
            return None

        # Check pre-mapped facts for this ring
        for fid, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fid) & 1:
                return target

        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate the heuristic value.
        
        Logic:
        1. Iterate rings from Largest to Smallest.
        2. Find the largest ring that is NOT at its goal position.
        3. If ring at index i (0-indexed) is misplaced, it implies a stack of size i+1 
           must be moved. The optimal moves are 2^(i+1) - 1.
        """
        if not self.rings_by_size:
            return 0

        # Iterate backwards: largest (index N-1) to smallest (index 0)
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If we don't care where this ring ends up, skip it
            if ring not in self.goal_positions:
                continue

            goal_pos = self.goal_positions[ring]
            current_pos = self._get_current_position(state_bitwise, ring)

            if current_pos != goal_pos:
                # Found the largest misplaced ring.
                # 'i' is the 0-based index, so it represents the i-th smallest ring.
                # The number of rings in the stack up to this ring is i + 1.
                # Heuristic cost = 2^(i+1) - 1
                return (1 << (i + 1)) - 1

        # All goal rings are in position
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute the heuristic for a given node.
        Called during search expansion.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini3ProHeuristic12(rings={self.num_rings})"

    def __str__(self):
        return f"TowersGemini3ProHeuristic12 with {self.num_rings} rings"
"