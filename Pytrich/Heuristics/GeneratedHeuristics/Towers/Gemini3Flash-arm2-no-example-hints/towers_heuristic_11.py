from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic11(Heuristic):
    """
    Domain-Dependent Heuristic for the Towers of Hanoi domain.

    1. Bottleneck: The serial constraint of the Hanoi puzzle where only the top ring can move
       and must be placed on a larger ring or empty tower peg.
    2. Admissible Lower Bound (Term 1): Recursive Hanoi distance to reach the goal peg assignment.
       This is the exact optimal cost (h*) for a 3-peg Hanoi configuration.
    3. Tie-breaking (Term 2): Small penalty for task network length to guide HTN decomposition.
    """

    def __init__(self, towers_gemini3flash_heuristic_11="towers_gemini3flash_heuristic_11"):
        super().__init__(towers_gemini3flash_heuristic_11)
        self.rings_ordered = []
        self.tower_names = set()
        self.on_facts_by_ring = {}  # ring -> {target_obj: global_id}
        self.goal_peg_map = {}
        self.memo = {}

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
        # 1. Identify all rings and towers
        rings = set()
        smaller_than_pairs = set()
        for i in range(model.ifacts_init, model.ifacts_end + 1):
            fact = model.get_component(i)
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                # args[0] is ring, args[1] is object (ring or tower)
                rings.add(args[0])
                if not args[1].startswith('t'): # Simple tower heuristic check
                    smaller_than_pairs.add((args[0], args[1]))
            if pred == "towerTop":
                self.tower_names.add(args[1])
            if pred == "on":
                if args[0] not in self.on_facts_by_ring:
                    self.on_facts_by_ring[args[0]] = {}
                self.on_facts_by_ring[args[0]][args[1]] = fact.global_id

        # 2. Order rings from smallest to largest using smallerThan relationships
        ring_counts = {r: 0 for r in rings}
        for r1, r2 in smaller_than_pairs:
            if r1 in ring_counts and r2 in rings:
                ring_counts[r1] += 1
        # smallest ring will be smaller than most others
        self.rings_ordered = sorted(rings, key=lambda x: ring_counts[x], reverse=True)

        # 3. Precompute goal peg for every ring
        goal_fact_names = model.state_explicit_repr(model.goals)
        goal_on_map = {}
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                goal_on_map[args[0]] = args[1]
        
        # Trace goal pegs from largest to smallest
        for i in range(len(self.rings_ordered)-1, -1, -1):
            r = self.rings_ordered[i]
            target = goal_on_map.get(r)
            if target in self.tower_names:
                self.goal_peg_map[r] = target
            else:
                self.goal_peg_map[r] = self.goal_peg_map.get(target)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _get_pegs(self, state):
        peg_map = {}
        # Process largest to smallest
        for i in range(len(self.rings_ordered)-1, -1, -1):
            r = self.rings_ordered[i]
            fid_dict = self.on_facts_by_ring.get(r, {})
            found = False
            for target, gid in fid_dict.items():
                if (state >> gid) & 1:
                    if target in self.tower_names:
                        peg_map[r] = target
                    else:
                        peg_map[r] = peg_map.get(target)
                    found = True
                    break
            if not found:
                # Fallback - should not happen in valid state
                peg_map[r] = list(self.tower_names)[0] if self.tower_names else "t1"
        return peg_map

    def _other_peg(self, p1, p2):
        for p in self.tower_names:
            if p != p1 and p != p2:
                return p
        return p1

    def _compute_h_to_peg(self, ring_idx, target_peg, current_peg_map):
        if ring_idx < 0: return 0
        ring = self.rings_ordered[ring_idx]
        curr = current_peg_map[ring]
        if curr == target_peg:
            return self._compute_h_to_peg(ring_idx - 1, target_peg, current_peg_map)
        else:
            other = self._other_peg(curr, target_peg)
            return (1 << ring_idx) + self._compute_h_to_peg(ring_idx - 1, other, current_peg_map)

    def _compute_h(self, ring_idx, target_peg, current_peg_map):
        if ring_idx < 0: return 0
        ring = self.rings_ordered[ring_idx]
        curr = current_peg_map[ring]
        if curr == target_peg:
            # Disk is at target, solve for disks below
            if ring_idx > 0:
                next_target = self.goal_peg_map.get(self.rings_ordered[ring_idx-1], target_peg)
                return self._compute_h(ring_idx - 1, next_target, current_peg_map)
            return 0
        else:
            # Move disk n to target, need smaller disks on 'other' peg
            other = self._other_peg(curr, target_peg)
            return (1 << ring_idx) + self._compute_h_to_peg(ring_idx - 1, other, current_peg_map)

    def _compute(self, state, task_network):
        if state in self.memo:
            h_base = self.memo[state]
        else:
            if not self.rings_ordered:
                h_base = 0
            else:
                peg_map = self._get_pegs(state)
                top_ring_idx = len(self.rings_ordered) - 1
                target = self.goal_peg_map.get(self.rings_ordered[top_ring_idx])
                h_base = self._compute_h(top_ring_idx, target, peg_map)
            self.memo[state] = h_base

        return h_base + (0.001 * len(task_network))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return int(h_value)
