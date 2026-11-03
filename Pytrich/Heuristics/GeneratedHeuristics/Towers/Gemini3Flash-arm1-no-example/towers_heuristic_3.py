from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    Bottleneck Analysis:
    The bottleneck in the Towers domain is the physical constraint that rings must be moved 
    in a strictly recursive, exponential sequence. To move a large ring, all smaller rings 
    currently on top of it or on its target must be cleared out of the way. This creates 
    a serial constraint where the largest misplaced ring dictates the minimum remaining work.

    Admissibility Sketch:
    - Term 1 (State-based Exponential): If the largest misplaced ring is at index k (0-indexed),
      moving it to its goal position requires at least 2^k moves in the classic Towers of Hanoi 
      setup. If the largest unstable ring is at index k (meaning it or something below it in 
      the goal stack is incorrect), 2^(k+1)-1 is the exact cost to complete that tower from scratch,
      but 2^k is a conservative lower bound for a single misplaced ring that must be cleared.
      We use h = (2^(max_unstable_idx + 1) - 1) as our primary guidance.
    - Term 2 (Task-based): Each 'move' or 'move_abstract' task in the task network represents 
      exactly one physical action. This is a simple additive lower bound for the immediate network.
    """

    def __init__(self, towers_gemini3flash_heuristic_3="towers_gemini3flash_heuristic_3"):
        super().__init__(name=towers_gemini3flash_heuristic_3)
        self.on_fact_ids = {}  # ring -> target_obj -> global_id
        self.ring_goals = {}   # ring -> goal_target_obj
        self.ring_size = {}    # ring -> integer (0 is smallest)
        self.sorted_rings_desc = []

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
        all_rings = set()
        smaller_than_pairs = set()

        # 1. Map 'on' facts and collect all rings
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                ring, target = args[0], args[1]
                all_rings.add(ring)
                if ring not in self.on_fact_ids:
                    self.on_fact_ids[ring] = {}
                self.on_fact_ids[ring][target] = fact.global_id
            elif pred == "smallerThan":
                smaller_than_pairs.add((args[0], args[1]))

        # 2. Determine ring sizes by counting how many other rings are smaller than them
        for r in all_rings:
            # Size = number of other rings 'x' such that smallerThan(x, r) is true
            size_count = sum(1 for x in all_rings if (x, r) in smaller_than_pairs)
            self.ring_size[r] = size_count

        # Sorted list of rings from largest to smallest for stability checking
        self.sorted_rings_desc = sorted(list(all_rings), key=lambda r: self.ring_size[r], reverse=True)

        # 3. Parse goals to find the target position for each ring
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fact_name in goal_fact_names:
            pred, args = self._parse_fact_name(fact_name)
            if pred == "on":
                self.ring_goals[args[0]] = args[1]

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state_bitwise, task_network):
        # 1. State-based exponential cost
        # Stability check: a ring is stable if it's on its goal AND the target it's on is stable.
        is_stable = True
        max_unstable_idx = -1

        for ring in self.sorted_rings_desc:
            goal_target = self.ring_goals.get(ring)
            if goal_target is None:
                continue

            # Find current target of the ring in the current state
            current_target = None
            for target_obj, fid in self.on_fact_ids.get(ring, {}).items():
                if (state_bitwise >> fid) & 1:
                    current_target = target_obj
                    break
            
            if is_stable and current_target == goal_target:
                # This ring is perfectly placed and its base is stable
                continue
            else:
                # This ring or something below it is wrong, so it must eventually move
                is_stable = False
                max_unstable_idx = max(max_unstable_idx, self.ring_size[ring])
                # Since we iterate from largest down, the first unstable we hit is the max
                break

        # Classic Hanoi lower bound: 2^(k+1)-1 moves for a stack of size k+1
        h_state = (1 << (max_unstable_idx + 1)) - 1 if max_unstable_idx != -1 else 0

        # 2. Task-based additive cost
        # We count move-related tasks as a simple linear bound
        h_task = sum(1 for task in task_network if "move" in task.name)

        # Tie-breaker: small penalty for task network size to favor decomposition progress
        tie_breaker = len(task_network) * 0.01

        return max(h_state, h_task) + tie_breaker

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = int(self._compute(node.state, node.task_network))
        super().update_info(h_value)
        return h_value
