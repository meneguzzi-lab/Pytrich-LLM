from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

"""
Bottleneck Analysis:
In the Towers (Hanoi) domain, the primary bottleneck is the recursive ordering constraint: to move a ring r_k, 
all rings smaller than it (r_1...r_{k-1}) must be moved to a spare peg. This results in an exponential move 
requirement (2^k - 1) for a tower of size k. 

Admissibility Sketch:
1. Exact Hanoi Distance: We compute the exact number of moves required to reach the goal configuration for a 
   3-peg tower of rings from the current configuration. This is a well-known admissible lower bound (and often 
   the exact cost) for Hanoi problems.
2. Task Network: The number of remaining tasks provides a minimal tie-breaker to guide search when state-based 
   costs are equal.
"""

class TowersGemini3FlashHeuristic8(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_8="towers_gemini3flash_heuristic_8"):
        super().__init__(towers_gemini3flash_heuristic_8)
        self.rings_ordered = []
        self.towers = set()
        self.goal_pegs = {}
        self.ring_on_fact_ids = {} # ring -> list of (target_obj, global_id)
        self.spares = {} # (peg1, peg2) -> peg3

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
        # 1. Identify all rings
        rings = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                rings.add(args[0])
        
        # 2. Order rings by size using smallerThan facts
        # size(r) = count of objects ?x such that smallerThan(?x, r) is true
        ring_sizes = {r: 0 for r in rings}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'smallerThan' and args[0] in rings and args[1] in rings:
                ring_sizes[args[1]] += 1
        self.rings_ordered = sorted(list(rings), key=lambda r: ring_sizes[r])

        # 3. Identify all towers
        all_objs = set()
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            for a in args: all_objs.add(a)
        self.towers = all_objs - rings

        # 4. Precompute sparing logic for 3-peg Hanoi
        t_list = sorted(list(self.towers))
        if len(t_list) >= 3:
            for t1 in t_list:
                for t2 in t_list:
                    if t1 != t2:
                        spare = [t for t in t_list if t != t1 and t != t2][0]
                        self.spares[(t1, t2)] = spare

        # 5. Build lookup for 'on' facts
        self.ring_on_fact_ids = {r: [] for r in self.rings_ordered}
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on':
                r, obj = args[0], args[1]
                if r in self.ring_on_fact_ids:
                    self.ring_on_fact_ids[r].append((obj, f.global_id))

        # 6. Parse Goal Pegs
        goal_fact_names = model.state_explicit_repr(model.goals)
        goal_on_map = {}
        for f_name in goal_fact_names:
            pred, args = self._parse_fact_name(f_name)
            if pred == 'on':
                goal_on_map[args[0]] = args[1]
        
        self.goal_pegs = {}
        # Trace chains from largest to smallest to find target peg for each ring
        for r in reversed(self.rings_ordered):
            target = goal_on_map.get(r)
            if target in self.towers:
                self.goal_pegs[r] = target
            elif target in self.goal_pegs:
                self.goal_pegs[r] = self.goal_pegs[target]
            else:
                self.goal_pegs[r] = None

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise, task_network):
        if not self.rings_ordered:
            return 0

        # 1. Determine current peg for each ring (O(N^2) bit tests)
        current_pegs = {}
        for r in reversed(self.rings_ordered):
            for obj, fid in self.ring_on_fact_ids[r]:
                if (state_bitwise >> fid) & 1:
                    if obj in self.towers:
                        current_pegs[r] = obj
                    else:
                        current_pegs[r] = current_pegs.get(obj)
                    break

        # 2. Compute exact Hanoi distance using iterative target reassignment
        h_dist = 0
        target_pegs = self.goal_pegs.copy()
        for i in range(len(self.rings_ordered) - 1, -1, -1):
            r = self.rings_ordered[i]
            cur = current_pegs.get(r)
            tar = target_pegs.get(r)
            if cur and tar and cur != tar:
                # To move ring i from cur to tar, all smaller rings must move to spare
                spare = self.spares.get((cur, tar))
                if spare:
                    for j in range(i):
                        target_pegs[self.rings_ordered[j]] = spare
                h_dist += (1 << i) # Add 2^i
        
        # 3. Add small tie-breaker for task network progress
        h_total = h_dist + 0.001 * len(task_network)
        return int(h_total) if h_total > 1000000000 else h_total

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
