import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic13(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    This heuristic exploits the optimal solution structure of the Towers of Hanoi problem.
    The minimum moves required to solve a k-ring subproblem is 2^k - 1.
    
    Key Logic:
    1. Preprocesses the domain to order rings by size (smallest to largest).
    2. Identifies the goal position for each ring.
    3. During search, scans rings from LARGEST to SMALLEST.
    4. The first ring found that is NOT in its goal position determines the lower bound.
       If ring k (where 1 is smallest) is misplaced, it and all smaller rings must move.
       This incurs a cost of at least 2^k - 1.
       
    This provides a tight, admissible lower bound that drastically prunes the search space.
    """

    def __init__(self, towers_gemini3pro_heuristic_12=None, name="TowersGemini3ProHeuristic13"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        # List of tuples: (fact_global_id, heuristic_weight)
        # Index i corresponds to the i-th smallest ring (0=smallest)
        self.ordered_goal_checks = []

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initialize heuristic with domain analysis.
        """
        start_time = time.time()
        self.model = model
        self._preprocess()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_heuristic(initial_node.state)
        return super().initialize(model, h_value)

    def _parse_fact_name(self, name: str):
        """
        Parses grounded fact names like '+on[r1,r2]' or '(on r1 r2)'.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args = name[bracket_idx+1:-1].split(",") if name[bracket_idx+1:-1] else []
                return predicate, args
        elif name.startswith("(") and name.endswith(")"):
            parts = name[1:-1].split()
            if parts:
                return parts[0], parts[1:]
        return None, []

    def _preprocess(self):
        """
        Analyzes static domain info to build efficient lookup tables.
        """
        rings = set()
        smaller_counts = defaultdict(int)
        fact_map = {f.name: f.global_id for f in self.model.facts}
        
        # 1. Determine Ring Hierarchy via smallerThan
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan" and len(args) == 2:
                r, o = args
                # In this domain, the first argument is the smaller ring
                if r.startswith("r") or "ring" in str(type(r)).lower(): 
                    rings.add(r)
                    smaller_counts[r] += 1
        
        # Sort rings: Smallest (smaller than many things) -> Largest
        # r1 < r2 < ... < t1. r1 counts high, r20 counts low.
        sorted_rings = sorted(list(rings), key=lambda r: -smaller_counts[r])
        
        # 2. Map Goals
        goal_map = {} # ring -> target_obj
        if hasattr(self.model, 'goals') and self.model.goals:
            # Get explicit fact names from the goal bitmask
            goal_names = self.model.state_explicit_repr(self.model.goals)
            for name in goal_names:
                pred, args = self._parse_fact_name(name)
                if (pred == "on" or pred == "goal_on") and len(args) == 2:
                    goal_map[args[0]] = args[1]

        # 3. Build Optimized Check List
        # We store checks ordered by ring size (smallest to largest)
        # But we will iterate backwards (largest to smallest) at runtime
        self.ordered_goal_checks = []
        
        for i, ring in enumerate(sorted_rings):
            # Heuristic weight for the i-th ring (1-based index k=i+1) is 2^k - 1
            weight = (1 << (i + 1)) - 1
            
            if ring in goal_map:
                target = goal_map[ring]
                # Construct potential grounded fact names
                # Standard format: +on[r1,r2]
                target_fact_name = f"+on[{ring},{target}]"
                fid = fact_map.get(target_fact_name)
                
                if fid is None:
                    # Fallback for HDDL format if necessary
                    target_fact_name_alt = f"(on {ring} {target})"
                    fid = fact_map.get(target_fact_name_alt)
                
                # Store (global_id, weight). If fid is None, we can't check it.
                self.ordered_goal_checks.append((fid, weight))
            else:
                # If a ring has no specific goal, it contributes 0 to the cost check 
                # (or we can't determine it is misplaced)
                self.ordered_goal_checks.append((None, weight))

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Calculates 2^k - 1 based on the largest misplaced ring.
        Using bitwise operations for max speed.
        """
        if not self.ordered_goal_checks:
            return 0
            
        # Iterate backwards from Largest Ring to Smallest Ring
        for i in range(len(self.ordered_goal_checks) - 1, -1, -1):
            fid, weight = self.ordered_goal_checks[i]
            
            if fid is not None:
                # Check if the goal fact is TRUE in the current state
                # (state >> fid) & 1 checks the specific bit
                is_at_goal = (state_bitwise >> fid) & 1
                
                if not is_at_goal:
                    # Found the largest misplaced ring.
                    # To move this ring, all smaller rings must also be moved.
                    # Cost is 2^k - 1.
                    return weight
        
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._calculate_heuristic(node.state)
        super().update_info(h_value)
        return h_value
