from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic0(Heuristic):
    """
    Bottleneck Analysis:
    The primary bottleneck in the Towers domain (Hanoi) is the strict recursive dependency of ring placement.
    A ring cannot be moved unless it is the 'towerTop' of its current tower and the destination is either a 
    tower base or a larger ring. This serializes all movements based on ring size.

    Admissibility Sketches:
    1. Misplaced Rings (h_state): Every ring that is not currently at its goal position, or is resting on 
       a ring that itself must be moved, must be moved at least once. This is a standard lower bound for Hanoi.
    2. Task Network (h_tasks): Each task currently in the HTN network represents a committed unit of work 
       that must eventually decompose into at least one primitive action to be cleared.
    """

    def __init__(self, towers_gemini3flash_heuristic_0="towers_heuristic_0"):
        super().__init__(towers_gemini3flash_heuristic_0)
        self.on_map = {} # (ring, obj) -> global_id
        self.goal_on = {} # ring -> target_obj
        self.rings = set()
        self.tower_bases = set()

    def initialize(self, model: Model, initial_node: HTNNode):
        # Precompute fact IDs and goal requirements
        goal_names = model.state_explicit_repr(model.goals)
        for name in goal_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                # args[0] is ring, args[1] is target
                self.goal_on[args[0]] = args[1]

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_map[(args[0], args[1])] = fact.global_id
                self.rings.add(args[0])
                if args[1].startswith('t'): # Simple heuristic for tower names
                    self.tower_bases.add(args[1])

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state_bitwise, task_network) -> int:
        # 1. State-based Term: Misplaced Rings
        # A ring is misplaced if it's not on its goal target, or if its target (if a ring) is misplaced.
        # We iterate in an order that respects the stack (bottom-up would be ideal, 
        # but checking goal conditions is a good proxy).
        misplaced_count = 0
        
        # Track which rings are 'stable' (correct and on a stable base)
        stable_rings = set()
        
        # We sort rings to process roughly bottom-to-top if names allow, 
        # but the logic below works regardless by iterating until convergence or simple pass.
        # For Towers, goal_on typically forms a chain: r1 on r2, r2 on r3, r3 on t3.
        
        # A simple pass: a ring is stable if (on r target) is true AND (target is tower OR target is stable)
        changed = True
        while changed:
            changed = False
            for ring, target in self.goal_on.items():
                if ring in stable_rings:
                    continue
                
                # Check if current state has (on ring target)
                fact_id = self.on_map.get((ring, target))
                if fact_id is not None and (state_bitwise >> fact_id) & 1:
                    if target in self.tower_bases or target in stable_rings:
                        stable_rings.add(ring)
                        changed = True

        misplaced_count = len(self.rings) - len(stable_rings)

        # 2. Task-based Term: Task Network Size
        # Every task in the network is a commitment to do work.
        task_count = len(task_network)

        # Total heuristic: sum of work required in state and work promised in network.
        # We add a small tie-breaker for task network length to prefer deeper/more-decomposed paths.
        return misplaced_count + task_count

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
