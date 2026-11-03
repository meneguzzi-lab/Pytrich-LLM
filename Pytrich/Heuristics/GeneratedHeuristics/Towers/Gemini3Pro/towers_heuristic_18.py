import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.

    This heuristic leverages the optimal solution property of the Towers of Hanoi problem:
    to move a stack of k disks, 2^k - 1 moves are required.
    
    The heuristic:
    1. Preprocesses the domain to establish a size ordering of rings (r1 < r2 < ... < rn).
    2. Identifies the goal position for each ring from the abstract goal description.
    3. During search, identifies the largest ring 'rk' that is NOT in its goal position.
    4. Returns 2^k - 1 as the heuristic estimate.
    
    This is admissible because moving the k-th largest ring to its correct destination 
    requires moving all k-1 smaller rings off it (and off the destination), moving the k-th ring,
    and moving the smaller rings back on top. This recursive structure forms the lower bound.
    """

    def __init__(self, towers_gemini3pro_heuristic_17=None, name="TowersGemini3ProHeuristic18"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.goal_positions = {}      # ring -> target object name
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.on_facts_by_ring = {}    # ring -> list of (fact, target_obj) tuples
        self.num_rings = 0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize heuristic with domain preprocessing.
        """
        start_time = time.time()
        self.model = model
        
        self._preprocess()
        
        self.preprocessing_time = time.time() - start_time
        
        # Compute initial h value
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parse fact name to (predicate, args).
        Handles "+pred[arg1,arg2]" and "(pred arg1 arg2)".
        """
        name = name.strip()
        # Format: +predicate[arg1,arg2] (Grounded)
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
                
        # Format: (predicate arg1 arg2) (HDDL)
        elif name.startswith("(") and name.endswith(")"):
            inner = name[1:-1]
            parts = inner.split()
            if parts:
                return parts[0], parts[1:]
                
        return None, []

    def _preprocess(self):
        """
        Extract static domain structure: Ring ordering and Goal definitions.
        """
        rings = set()
        smaller_than_counts = defaultdict(int)

        # 1. Extract Goal Configuration
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) == 2:
                    ring = args[0]
                    target = args[1]
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target
                        rings.add(ring)

        # 2. Analyze Facts for Ring Sizes and 'On' predicates
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue

            if predicate == "smallerThan" and len(args) == 2:
                r_small = args[0]
                r_large = args[1]
                # Only consider ring-ring relations for sorting
                if r_small.startswith("r"):
                    rings.add(r_small)
                    if r_large.startswith("r"):
                        rings.add(r_large)
                        # r_small is smaller than r_large
                        smaller_than_counts[r_small] += 1

            elif predicate == "on" and len(args) == 2:
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

        # 3. Sort Rings from Smallest to Largest
        # A ring is "smaller" if it appears as the first arg in more 'smallerThan' facts
        # r1 < r2 < r3 => r1 counts: {r2, r3} (2), r2 counts: {r3} (1), r3 counts: {} (0)
        # Sorting by count descending gives [r1, r2, r3]
        self.rings_by_size = sorted(list(rings), key=lambda r: -smaller_than_counts[r])

        # Fallback: If smallerThan was not explicitly given for rings (inference required)
        if all(c == 0 for c in smaller_than_counts.values()) and len(rings) > 1:
            # Infer from goal chain if possible: on(r1, r2) implies r1 < r2
            is_smaller = defaultdict(set)
            for r, target in self.goal_positions.items():
                if target.startswith("r"):
                    is_smaller[r].add(target)
            
            # Simple transitive closure to build counts
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller.keys()):
                    for larger in list(is_smaller[r]):
                        for even_larger in is_smaller.get(larger, set()):
                            if even_larger not in is_smaller[r]:
                                is_smaller[r].add(even_larger)
                                changed = True
                                
            self.rings_by_size = sorted(list(rings), key=lambda r: -len(is_smaller.get(r, set())))

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Efficiently find the object the ring is currently on using bitwise operations.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for fact, target in self.on_facts_by_ring[ring]:
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculate h(n) = 2^k - 1, where k is the rank of the largest misplaced ring.
        """
        if not self.rings_by_size:
            return 0
            
        # Iterate from Largest ring to Smallest ring
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # If this ring has no defined goal, skip it
            if ring not in self.goal_positions:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            goal_pos = self.goal_positions[ring]
            
            # Found the largest ring that is not where it belongs
            if current_pos != goal_pos:
                # k is the 1-based index/rank of the ring (Smallest is 1)
                # i is 0-based index
                k = i + 1
                return (1 << k) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Compute heuristic value for a search node.
        """
        h_value = self._calculate_heuristic(node.state)
        try:
            super().update_info(h_value)
        except AttributeError:
            pass
        return h_value
