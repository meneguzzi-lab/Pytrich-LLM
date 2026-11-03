from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic14(Heuristic):
    """
    Bottleneck: The Tower of Hanoi serial constraint. Moving a ring requires clearing all rings above it 
    and ensuring the target location is clear and has a larger base.
    
    Admissible Lower Bounds:
    1. Hanoi Distance: 2^k - 1, where k is the size of the largest ring that is not currently in its 
       goal-satisfied position. This is the minimum number of actions to solve a k-ring Hanoi subproblem.
    2. Task Network Size: Each task in the network must eventually decompose into actions. 
       A network of length L implies at least L steps remain in the hierarchy.
    3. Misplaced Ring Count: Each ring that is not in its goal state must be moved at least once.
    """

    def __init__(self, towers_gemini3flash_heuristic_14="towers_gemini3flash_heuristic_14"):
        super().__init__(towers_gemini3flash_heuristic_14)
        self.on_id_map = {}
        self.goal_on = {}
        self.rings_sorted = []
        self.ring_to_size = {}
        self.all_objs = []

    def initialize(self, model: Model, initial_node: HTNNode):
        # 1. Map goal state
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fname in goal_fact_names:
            pred, args = self._parse_fact_name(fname)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # 2. Identify rings and calculate sizes via smallerThan relations
        rings = set()
        smaller_than_counts = {}
        objs = set()
        
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                rings.add(args[0])
                smaller_than_counts[args[0]] = smaller_than_counts.get(args[0], 0) + 1
            for arg in args:
                objs.add(arg)
        
        self.all_objs = list(objs)
        # A smaller ring is smallerThan more things. 
        # Sort rings by count of things they are smaller than (descending) to get smallest to largest.
        # Then reverse to get largest to smallest for satisfaction checking.
        self.rings_sorted = sorted(list(rings), key=lambda r: smaller_than_counts.get(r, 0), reverse=False)
        # ring_to_size: smallest ring (last in counts) is size 1, largest is size N.
        sorted_by_size = sorted(list(rings), key=lambda r: smaller_than_counts.get(r, 0), reverse=True)
        self.ring_to_size = {r: i+1 for i, r in enumerate(sorted_by_size)}

        # 3. Pre-map all 'on' global IDs
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_id_map[(args[0], args[1])] = fact.global_id

        h_val = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_val)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_val = self._compute(node.state, node.task_network)
        super().update_info(h_val)
        return h_val

    def _compute(self, state, task_network):
        misplaced = {}
        max_k = 0
        num_misplaced = 0

        # Evaluate rings from largest to smallest to check recursive satisfaction
        for ring in self.rings_sorted:
            size = self.ring_to_size[ring]
            target = self.goal_on.get(ring)
            current_on = None
            
            # Fast identification of current location
            for obj in self.all_objs:
                oid = self.on_id_map.get((ring, obj))
                if oid is not None and (state >> oid) & 1:
                    current_on = obj
                    break
            
            # A ring is misplaced if it's on the wrong thing OR its target is also misplaced
            is_mis = (current_on != target) or (target in misplaced and misplaced[target])
            misplaced[ring] = is_mis
            
            if is_mis:
                max_k = max(max_k, size)
                num_misplaced += 1

        # h_hanoi = 2^k - 1 moves for a k-sized stack
        h_hanoi = (1 << max_k) - 1 if max_k > 0 else 0
        
        # Tie-breakers: HTN structure and count of misplaced items
        h_total = h_hanoi + len(task_network) + num_misplaced
        return h_total

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
