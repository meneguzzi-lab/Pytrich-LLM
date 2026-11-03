from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic8(Heuristic):
    """
    Bottleneck: The exponential recursive dependency of ring placement.
    To move any ring i, all smaller rings (1 to i-1) must be moved out of the way to an auxiliary peg.

    Admissibility Sketches:
    1. State-based Hanoi Cost (h_hanoi): A ring of size 'k' that is not on its goal peg requires at least
       2^(k-1) moves to clear smaller rings and move the ring itself. This is a standard lower bound
       for the Towers of Hanoi problem.
    2. Task Network Move Count (h_tn): Every 'move' or 'move_abstract' task currently in the HTN
       decomposition represents exactly one physical operator execution. Summing these is admissible.
    """
    def __init__(self, towers_gemini3flash_heuristic_8="towers_gemini3flash_heuristic_8"):
        super().__init__(name=towers_gemini3flash_heuristic_8)
        self.ring_sizes = {}  # ring_name -> size (1 to N)
        self.ring_goal = {}   # ring_name -> goal_obj_name
        self.on_fact_ids = {} # (ring, obj) -> global_id

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Identify all rings and their relative sizes
        rings = set()
        smaller_than_pairs = []
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                # First argument is always a ring based on domain types
                rings.add(args[0])
                smaller_than_pairs.append((args[0], args[1]))

        # Size is rank-based: count how many rings 'r' is smaller than
        num_rings = len(rings)
        for r in rings:
            count_larger_rings = sum(1 for src, dest in smaller_than_pairs if src == r and dest in rings)
            self.ring_sizes[r] = num_rings - count_larger_rings

        # 2. Extract goal positions for rings from model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_fact_names:
            pred, args = self._parse_fact_name(fact_name)
            if pred == "on" and len(args) == 2:
                self.ring_goal[args[0]] = args[1]

        # 3. Map (ring, target) to fact IDs for fast state checking
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on" and len(args) == 2:
                self.on_fact_ids[(args[0], args[1])] = fact.global_id

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

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

    def _compute(self, state_bitwise: int, task_network: list) -> int:
        # Term 1: Hanoi State Lower Bound
        # Find the largest ring (size k) that is not on its goal peg
        h_hanoi = 0
        sorted_rings = sorted(self.ring_sizes.items(), key=lambda x: x[1], reverse=True)
        for ring_name, size in sorted_rings:
            goal_obj = self.ring_goal.get(ring_name)
            if goal_obj:
                fact_id = self.on_fact_ids.get((ring_name, goal_obj))
                # If ring is not on its goal object, it requires 2^(size-1) moves
                if fact_id is None or not ((state_bitwise >> fact_id) & 1):
                    h_hanoi = (1 << (size - 1))
                    break

        # Term 2: Task Network Moves
        # Count pending physical moves in the network
        tn_move_count = 0
        for task in task_network:
            tname = task.name.lower()
            if 'move' in tname: # Catches 'move' and 'move_abstract'
                tn_move_count += 1

        # Combine with a small tie-breaker for HTN progress (len(TN))
        # Use max for admissibility, but add sub-unit penalty for guidance
        h_base = max(h_hanoi, tn_move_count)
        return h_base + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
