import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi (towers) domain.
    
    This heuristic leverages the optimal solution structure of the Towers of Hanoi problem.
    The minimum moves to solve a sub-problem for a stack of k rings is 2^k - 1.
    
    Heuristic Logic:
    1. Identifies the hierarchy of rings based on 'smallerThan' predicates.
    2. Identifies the goal position for each ring from the goal state.
    3. Iterating from the largest ring to the smallest:
       - Finds the largest ring 'r' that is NOT currently on its goal target.
       - If such a ring is found at index k (0-indexed, where 0 is smallest), it implies
         that this ring and all k smaller rings must be moved.
       - The heuristic estimate is 2^(k+1) - 1.
       
    This approach is admissible and provides a tight lower bound for the standard Towers problem.
    """

    def __init__(self, towers_gemini3pro_heuristic_7=None, name="TowersGemini3ProHeuristic8"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Static domain structures
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.goal_positions = {}      # ring_name -> target_obj_name
        self.ring_on_facts = defaultdict(list) # ring_name -> list of (global_id, target_obj_name)
        self.num_rings = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize heuristic with domain analysis.
        Executed once before search begins.
        """
        start_time = time.time()
        self.model = model
        
        self._preprocess()
        
        self.preprocessing_time = time.time() - start_time
        
        # Initial heuristic value
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses fact names in format +predicate[arg1,arg2] or (predicate arg1 arg2).
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
            parts = name[1:-1].split()
            if parts:
                return parts[0], parts[1:]
                
        return None, []

    def _preprocess(self):
        """
        Analyze static facts to determine ring sizes and map fact IDs.
        Analyze goal state to determine target positions.
        """
        rings = set()
        smaller_than_counts = defaultdict(int)
        
        # 1. Parse Goal State
        # Goals are stored as a bitwise integer in model.goals
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2. Parse all facts to build ring hierarchy and fast lookup maps
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            
            if not pred or not args:
                continue
                
            if pred == "smallerThan":
                # smallerThan ?r1 ?r2/obj
                if len(args) == 2:
                    small, large = args[0], args[1]
                    if small.startswith("r"):
                        rings.add(small)
                        # Count how many things this ring is smaller than
                        smaller_than_counts[small] += 1
                        if large.startswith("r"):
                            rings.add(large)
            
            elif pred == "on":
                # Map 'on' facts for fast state checking
                if len(args) == 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        rings.add(ring)
                        # Store (global_id, target) for quick lookup
                        self.ring_on_facts[ring].append((fact.global_id, target))

        self.num_rings = len(rings)
        
        # 3. Sort rings from smallest to largest
        # The smallest ring is smaller than the most other objects.
        # So we sort by count descending.
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_counts.get(r, 0))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Finds where a ring is located in the current state using bitwise operations.
        """
        if ring not in self.ring_on_facts:
            return None
            
        # Check pre-cached fact IDs
        for global_id, target in self.ring_on_facts[ring]:
            if (state_bitwise >> global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates 2^k - 1 based on the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0
            
        # Iterate from largest ring to smallest ring
        # rings_by_size is [smallest, ..., largest]
        # We traverse backwards.
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If this ring has no specific goal, we assume it's fine unless
            # it blocks a larger ring. But here we are iterating largest down,
            # so if we are at this ring, larger ones are already satisfied.
            # If a ring has no goal, we skip it (don't force a move).
            if ring not in self.goal_positions:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]
            
            if current_pos != goal_pos:
                # Found the largest ring that is not in position.
                # To move this ring (index i), we effectively need to solve
                # the Hanoi problem for size i+1 (since i is 0-indexed).
                # Cost = 2^(i+1) - 1
                exponent = i + 1
                return (1 << exponent) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for a search node.
        """
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
