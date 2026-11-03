import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    This heuristic implements the optimal 2^k - 1 lower bound logic for Towers of Hanoi problems.
    It identifies the largest ring that is not in its goal position and estimates the cost
    based on the recursive structure of the problem: to move the k-th smallest ring, 
    one must move the k-1 smaller rings twice (off and back on).
    
    Design Rationale:
    1. **Admissibility**: The cost to move a ring 'k' given ToH constraints is strictly bounded 
       by 2^k - 1 (where k is the 1-based rank of the ring). This assumes all smaller rings 
       must be moved out of the way.
    2. **Efficiency**: Uses bitwise operations to check state facts, avoiding string manipulations 
       during the search loop. Static analysis is done once in initialize().
    3. **Robustness**: Handles explicit goal facts (common in HDDL) and 'smallerThan' predicates 
       to determine ring hierarchy.
    """

    def __init__(self, towers_gemini3pro_heuristic_5=None, name="TowersGemini3ProHeuristic6"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        
        # Static domain data computed in initialize
        self.rings_ordered = []  # List of rings sorted smallest to largest
        self.goal_map = {}       # Map: ring_name -> target_obj_name
        
        # Fast lookup data structures for runtime
        # List of tuples: (ring_name, goal_target_name, list_of_fact_tuples)
        # where list_of_fact_tuples contains (global_id, target_obj_name)
        self.fast_ring_data = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Precompute domain structure and goal states.
        """
        start_time = time.time()
        self.model = model

        self._preprocess_domain()
        
        self.preprocessing_time = time.time() - start_time
        
        # Calculate initial h-value
        h_val = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_val)

    def _parse_fact_name(self, name: str):
        """
        Parse fact names in the format +predicate[arg1,arg2] or (predicate arg1 arg2).
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
        Extract ring sizes, goal positions, and map facts to bitwise IDs.
        """
        rings = set()
        smaller_counts = defaultdict(int)
        on_facts_by_ring = defaultdict(list)
        
        # 1. Parse Goal State
        # Goals are stored in model.goals as a bitwise integer.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g_name in goal_fact_names:
                pred, args = self._parse_fact_name(g_name)
                if pred == "on" and len(args) == 2:
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_map[ring] = target
                elif pred == "goal_on" and len(args) == 2:
                    # Sometimes goals are defined via goal_on predicate in init
                    ring, target = args[0], args[1]
                    if ring.startswith("r"):
                        self.goal_map[ring] = target

        # 2. Parse All Facts for Structure (smallerThan and on)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred or len(args) < 2:
                continue
            
            if pred == "smallerThan":
                # smallerThan r1 t1 -> r1 is smaller than t1.
                # Count how many things a ring is smaller than.
                # The smallest ring is smaller than everything else (highest count).
                r = args[0]
                if r.startswith("r"):
                    rings.add(r)
                    smaller_counts[r] += 1
                    if args[1].startswith("r"):
                        rings.add(args[1])
            
            elif pred == "on":
                r, target = args[0], args[1]
                if r.startswith("r"):
                    rings.add(r)
                    # Store (global_id, target_name) for fast runtime lookup
                    on_facts_by_ring[r].append((fact.global_id, target))

        # 3. Sort Rings by Size
        # Sort descending by 'smaller_than' count. Smallest ring has most 'smallerThan' facts.
        if not rings:
            return
            
        self.rings_ordered = sorted(list(rings), key=lambda r: smaller_counts[r], reverse=True)
        
        # 4. Build Fast Runtime Data Structure
        # Structure: List ordered by ring size (smallest to largest)
        # Item: (ring_name, goal_target_or_None, list_of_on_facts)
        for ring in self.rings_ordered:
            goal_target = self.goal_map.get(ring)
            facts = on_facts_by_ring.get(ring, [])
            self.fast_ring_data.append((ring, goal_target, facts))

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Compute the 2^k - 1 heuristic.
        Iterate from largest ring to smallest. If a ring is misplaced, it dictates the cost.
        """
        # Iterate backwards (Largest -> Smallest)
        # fast_ring_data is sorted Smallest -> Largest
        for i in range(len(self.fast_ring_data) - 1, -1, -1):
            ring_name, goal_target, on_facts = self.fast_ring_data[i]
            
            # If this ring has no specific goal, we assume it's fine where it is
            if goal_target is None:
                continue
            
            # Find current position
            current_pos = None
            for global_id, target in on_facts:
                if (state_bitwise >> global_id) & 1:
                    current_pos = target
                    break
            
            # If misplaced, this is the largest misplaced ring 'k'
            # We need to move k (1 move) + move all smaller (i rings) twice roughly.
            # The formula is 2^(number of rings up to this one) - 1
            # number of rings up to this one = i + 1 (since 0-indexed)
            if current_pos != goal_target:
                return (1 << (i + 1)) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._calculate_heuristic(node.state)
        self.update_info(h_val)
        return h_val
