import time
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3ProHeuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Hanoi) domain.
    
    This heuristic implements the optimal cost estimation for the Towers of Hanoi problem.
    The minimum number of moves to solve a stack of k disks is 2^k - 1.
    
    Strategy:
    1. Preprocess the domain to identify rings and their relative sizes (based on smallerThan predicates).
    2. Preprocess goal positions from the model's goal state.
    3. During evaluation, identify the largest ring 'r_k' (index k) that is NOT in its goal position.
    4. The heuristic value is 2^(k+1) - 1 (since indices are 0-based).
    
    This is admissible because to move a ring, all smaller rings must be moved off it, 
    and the recursive structure of the problem dictates this exponential cost.
    """

    def __init__(self, towers_gemini3pro_heuristic_18=None, name="TowersGemini3ProHeuristic19"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.rings_by_size = []       # List of rings sorted smallest to largest
        self.goal_positions = {}      # ring -> target object name
        self.on_facts_by_ring = {}    # ring -> list of (Fact, target_obj_name)
        self.num_rings = 0

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
        Parse fact names handling both grounded +pred[arg] and standard (pred arg) formats.
        """
        name = name.strip()
        # Format 1: +predicate[arg1,arg2]
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
        Analyze static domain structure: ring hierarchy and goal definitions.
        """
        rings = set()
        smaller_than_relations = []

        # 1. Parse Goal State
        # model.goals is a bitwise integer. We parse the explicit representation.
        if hasattr(self.model, 'goals') and self.model.goals:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for goal_name in goal_fact_names:
                predicate, args = self._parse_fact_name(goal_name)
                if predicate == "on" and len(args) >= 2:
                    ring, target = args[0], args[1]
                    # We assume things starting with 'r' are rings as per domain convention
                    if ring.startswith("r"):
                        self.goal_positions[ring] = target

        # 2. Parse Facts for Structure (smallerThan) and Dynamics (on)
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate or len(args) < 2:
                continue

            if predicate == "smallerThan":
                smaller, larger = args[0], args[1]
                if smaller.startswith("r"):
                    rings.add(smaller)
                    # Collect relations to sort rings later
                    smaller_than_relations.append((smaller, larger))
                    if larger.startswith("r"):
                        rings.add(larger)

            elif predicate == "on":
                ring, target = args[0], args[1]
                if ring.startswith("r"):
                    rings.add(ring)
                    if ring not in self.on_facts_by_ring:
                        self.on_facts_by_ring[ring] = []
                    # Store the Fact object (for global_id) and the target name
                    self.on_facts_by_ring[ring].append((fact, target))

        self.num_rings = len(rings)
        if self.num_rings == 0:
            return

        # 3. Sort Rings from Smallest to Largest
        # We calculate a score: how many objects is ring R smaller than?
        # The smallest ring (top) is smaller than the most other rings/objects.
        if smaller_than_relations:
            smaller_count = defaultdict(int)
            for smaller, larger in smaller_than_relations:
                if smaller in rings:
                    smaller_count[smaller] += 1
            self.rings_by_size = sorted(list(rings), key=lambda r: smaller_count[r], reverse=True)
        else:
            # Fallback: Infer hierarchy from goal chain if smallerThan facts are missing/abstracted
            # If goal is (on r1 r2), then r1 < r2.
            is_smaller = defaultdict(set)
            for r, t in self.goal_positions.items():
                if t.startswith("r"):
                    is_smaller[r].add(t)
            
            # Transitive closure
            changed = True
            while changed:
                changed = False
                for r in list(is_smaller.keys()):
                    for larger in list(is_smaller[r]):
                        for even_larger in is_smaller.get(larger, set()):
                            if even_larger not in is_smaller[r]:
                                is_smaller[r].add(even_larger)
                                changed = True
                                
            self.rings_by_size = sorted(list(rings), key=lambda r: len(is_smaller[r]), reverse=True)

    def _get_current_position(self, state_bitwise: int, ring: str) -> str:
        """
        Determine the current object the ring is 'on'.
        Uses efficient bitwise check against the state mask.
        """
        if ring not in self.on_facts_by_ring:
            return None
            
        for fact, target in self.on_facts_by_ring[ring]:
            # Check bit at global_id
            if (state_bitwise >> fact.global_id) & 1:
                return target
        return None

    def _calculate_heuristic(self, state_bitwise: int) -> int:
        """
        Compute the heuristic value: 2^k - 1 for the largest misplaced ring k.
        """
        if not self.rings_by_size:
            return 0
            
        # Iterate from largest ring (end of list) to smallest ring (start of list)
        # self.rings_by_size is sorted [smallest, ..., largest]
        for i in range(len(self.rings_by_size) - 1, -1, -1):
            ring = self.rings_by_size[i]
            
            # Check if this ring is in its goal position
            goal_pos = self.goal_positions.get(ring)
            if goal_pos is None:
                continue
                
            current_pos = self._get_current_position(state_bitwise, ring)
            
            if current_pos != goal_pos:
                # Found largest misplaced ring at index i.
                # This ring represents the base of a sub-problem of size k = i + 1.
                # Cost to move a stack of size k is 2^k - 1.
                # (e.g. if i=0, it's the smallest ring, cost 2^1 - 1 = 1)
                return (1 << (i + 1)) - 1
                
        return 0

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        """
        Calculate heuristic for a search node.
        """
        h_value = self._calculate_heuristic(node.state)
        # Update heuristic statistics in parent class
        super().update_info(h_value)
        return h_value

    def __str__(self):
        return f"TowersGemini3ProHeuristic19 with {self.num_rings} rings"
