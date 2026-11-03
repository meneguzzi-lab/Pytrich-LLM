import collections
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic13(Heuristic):
    """
    Bottleneck: The Tower of Hanoi constraint (only top-most and smaller rings can move) 
    creates a serial bottleneck where moving a large ring requires all rings above it to 
    be moved twice (once to clear the source and once to clear the destination).

    Admissible Lower Bounds:
    1. Misplaced Ring Cost: If a ring of size k (where k=1 is smallest) is not in its goal 
       position, it must move at least once. However, to move it, all k-1 rings smaller than it 
       must also have been moved at some point to clear the stack. Thus, if the largest misplaced 
       ring has index k, at least k moves are required.
    2. Task Network Cost: Every 'move_abstract' task remaining in the task network corresponds 
       to a physical move action that must be performed. Every other complex task represents 
       at least one eventual move action.
    """

    def __init__(self, towers_gemini3flash_heuristic_13="towers_gemini3flash_heuristic_13"):
        super().__init__(towers_gemini3flash_heuristic_13)
        self.ring_sizes = {}
        self.on_ids = {}
        self.ring_goals = {}
        self.ring_sizes_ordered = []

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
        # 1. Identify all rings and map 'on' predicates
        all_rings = set()
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                all_rings.add(args[0])
            if pred == "on" and len(args) == 2:
                self.on_ids[(args[0], args[1])] = fact.global_id

        # 2. Compute ring sizes (k-th smallest)
        # r1 is smallerThan r2, r3... so r1 has 19 larger rings in a 20-ring set. 
        # size = total_rings - larger_rings_count
        ring_to_larger_rings = collections.defaultdict(set)
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                # Only count larger objects that are also rings to determine relative rank
                if args[0] in all_rings and args[1] in all_rings:
                    ring_to_larger_rings[args[0]].add(args[1])
        
        num_rings = len(all_rings)
        for r in all_rings:
            # Rank r1 as size 1, rN as size N
            self.ring_sizes[r] = num_rings - len(ring_to_larger_rings[r])
        
        self.ring_sizes_ordered = sorted(self.ring_sizes.items(), key=lambda x: x[1], reverse=True)

        # 3. Extract goal configuration
        goal_fact_names = model.state_explicit_repr(model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                # args[0] is the ring, args[1] is the target it should be on
                self.ring_goals[args[0]] = args[1]
        
        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

    def _compute(self, state, task_network):
        # Term 1: State cost based on largest misplaced ring
        # Moving ring of size k requires at least k moves (admissible)
        h_state = 0
        for ring, size in self.ring_sizes_ordered:
            goal_obj = self.ring_goals.get(ring)
            if goal_obj:
                on_goal_id = self.on_ids.get((ring, goal_obj))
                if on_goal_id is not None:
                    # Bitwise check if ring is on its goal
                    if not ((state >> on_goal_id) & 1):
                        h_state = size
                        break
        
        # Term 2: Task network cost
        # Every remaining task is at least one move action
        h_tn = len(task_network)
        
        # Both are admissible lower bounds on remaining moves
        # Max of two lower bounds is admissible
        return max(h_state, h_tn)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
