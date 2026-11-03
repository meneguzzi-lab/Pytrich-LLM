from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic2(Heuristic):
    """
    Domain: Towers (Hanoi-style ring stacking).
    
    1. Bottleneck: 
       The exponential nature of moving a stack of N rings. Moving the i-th largest ring 
       to its target requires clearing all i-1 smaller rings above it, moving it, and then 
       moving the i-1 rings back on top of it. This creates a serialized bottleneck where 
       cost grows as 2^n - 1.

    2. Admissible Lower Bounds:
       - h_hanoi: The standard Towers of Hanoi distance. We sum 2^(size-1) for every ring 
         that is not 'correctly placed'. A ring is correctly placed only if it is on its 
         goal support AND that support is itself correctly placed. 
         Admissibility: This is a mathematically proven lower bound for the minimal 
         number of moves required in any Tower of Hanoi problem from a given state.
       - h_tn: Count of 'move' and 'move_abstract' tasks in the current task network.
         Admissibility: Each such task represents at least one unique primitive action 
         in the resulting plan.
    """
    def __init__(self, towers_gemini3flash_heuristic_2="towers_gemini3flash_heuristic_2"):
        super().__init__(name=towers_gemini3flash_heuristic_2)
        self.on_fact_ids = {}      # (ring, support) -> global_id
        self.ring_goals = {}       # ring -> target_support
        self.ring_sizes = {}       # ring -> int (1=smallest)
        self.sorted_rings = []     # rings sorted largest to smallest

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        rings = set()
        smaller_than_relations = []
        
        # Parse grounded facts
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
            elif pred == "smallerThan":
                smaller_than_relations.append((args[0], args[1]))
                # Only track rings; ignore towers in the ring set
                # Type detection: we know ?r is a RING in (smallerThan ?r ?o)
                rings.add(args[0])
        
        # Calculate ring sizes (rank by count of smaller rings below it)
        for r in rings:
            # size = 1 + count of other rings smaller than this ring
            count = sum(1 for r_small, r_large in smaller_than_relations 
                        if r_large == r and r_small in rings)
            self.ring_sizes[r] = count + 1
            
        # Parse goal state from model bitwise representation
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                self.ring_goals[args[0]] = args[1]
                
        # Order rings for top-down checking (largest first)
        self.sorted_rings = sorted(self.ring_sizes.keys(), key=lambda r: self.ring_sizes[r], reverse=True)
        
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state: int, task_network) -> int:
        # 1. Calculate Hanoi state-distance
        # A ring is 'stable' if it and all rings below it in the goal are satisfied
        correctly_placed = set()
        for ring in self.sorted_rings:
            goal_support = self.ring_goals.get(ring)
            if not goal_support:
                continue
            
            # Check if current state has ring on its goal support
            fact_id = self.on_fact_ids.get((ring, goal_support))
            is_on_goal_support = fact_id is not None and (state >> fact_id) & 1
            
            if is_on_goal_support:
                # If goal support is a tower (base), it's satisfied
                if goal_support not in self.ring_sizes:
                    correctly_placed.add(ring)
                # If goal support is a ring, that ring must also be correctly placed
                elif goal_support in correctly_placed:
                    correctly_placed.add(ring)
        
        h_hanoi = 0
        for ring in self.sorted_rings:
            if ring not in correctly_placed:
                # Standard Hanoi additive term: 2^(size-1)
                h_hanoi += (1 << (self.ring_sizes[ring] - 1))
        
        # 2. Count moves in Task Network
        h_tn = 0
        for task in task_network:
            if hasattr(task, 'name'):
                # Any move-related task contributes at least one move
                if "move" in task.name:
                    h_tn += 1
                
        return max(h_hanoi, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
