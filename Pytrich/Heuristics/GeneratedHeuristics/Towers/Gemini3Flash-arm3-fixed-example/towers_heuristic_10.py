from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic10(Heuristic):
    """
    Domain-dependent heuristic for the Tower of Hanoi (Towers) domain.

    1. Bottleneck Analysis:
       The core bottleneck in the Towers domain is the recursive disk stacking constraint:
       a disk r can only be moved if it is the top of its current stack and the target is
       the top of its stack and is larger than r. This forces a serial execution where
       to move disk N, all disks 1..N-1 must be moved to the auxiliary tower.

    2. Admissible Lower Bounds:
       - Term 1: Foundation Stability Count (Admissible).
         A disk is considered 'stable' only if it is currently on its goal object and
         that goal object is also stable (towers are always stable). Any disk that is 
         not stable must move at least once. This is a standard lower bound for stacking.
       - Term 2: Task Network Pending Moves (Admissible).
         Each 'move' or 'move_abstract' task in the task network represents a physical
         action that must be taken. Counting these provides a lower bound on the remaining
         work in the current decomposition branch.

    3. Composite Heuristic:
       h = max(unstable_rings, pending_moves_in_tn) + 0.001 * total_tasks_in_tn
       The tie-breaker (0.001) helps the search prioritize shallower task networks.
    """

    def __init__(self, towers_gemini3flash_heuristic_10="towers_gemini3flash_heuristic_10"):
        super().__init__(name=towers_gemini3flash_heuristic_10)
        self.rings = []
        self.towers = []
        self.on_ids = {}
        self.goal_map = {}
        self.towers_gemini3flash_heuristic_10_initialized = False

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
        self.rings = []
        self.towers = []
        self.on_ids = {}
        self.goal_map = {}

        # Precompute IDs and Object lists
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                # args[0] is RING, args[1] is RING or TOWER
                self.on_ids[(args[0], args[1])] = fact.global_id
                if args[0] not in self.rings:
                    self.rings.append(args[0])
            elif pred == "towerTop":
                # args[1] is TOWER
                if args[1] not in self.towers:
                    self.towers.append(args[1])

        # Parse Goal State
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gfn in goal_fact_names:
            pred, args = self._parse_fact_name(gfn)
            if pred == "on":
                self.goal_map[args[0]] = args[1]

        self.towers_gemini3flash_heuristic_10_initialized = True
        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _is_stable(self, ring, state, memo):
        if ring in memo:
            return memo[ring]
        if ring in self.towers:
            return True
        
        target = self.goal_map.get(ring)
        if not target:
            memo[ring] = True
            return True

        # Check if current state has ring on its target
        fid = self.on_ids.get((ring, target))
        if fid is not None and (state >> fid) & 1:
            # Ring is on target. Stability depends on target stability.
            res = self._is_stable(target, state, memo)
            memo[ring] = res
            return res
        else:
            # Ring is not on target or target is not stable
            memo[ring] = False
            return False

    def _compute(self, state, task_network):
        if not self.towers_gemini3flash_heuristic_10_initialized:
            return 0

        # Term 1: Unstable rings
        memo = {}
        unstable_count = 0
        for r in self.rings:
            if not self._is_stable(r, state, memo):
                unstable_count += 1

        # Term 2: Task network complexity
        pending_moves = 0
        for task in task_network:
            name_lower = task.name.lower()
            if "move" in name_lower:
                pending_moves += 1

        # Composite value with tie-breaker
        h_base = max(unstable_count, pending_moves)
        h_tie = 0.001 * len(task_network)
        
        return h_base + h_tie

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
