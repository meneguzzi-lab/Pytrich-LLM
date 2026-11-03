from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The Towers of Hanoi domain is constrained by the recursive dependency of rings.
# Moving a larger ring requires all smaller rings to be moved to an auxiliary peg,
# creating a bottleneck where work scales exponentially with the number of disks.
#
# Admissibility Sketches:
# Term 1: State-based Hanoi distance. In a 3-peg configuration, the minimum moves
# to relocate a stack of k disks is 2^k - 1. If the largest incorrectly placed
# ring is the k-th ring (1-indexed), the cost to settle it and all disks above
# it is at least 2^k - 1. This is a classic lower bound for Hanoi.
# Term 2: Task network size. Each abstract task or operator in the network must
# at least be processed once, contributing at least 1 unit of cost (or 1 move
# if primitive). Summing them is admissible as a low-level tie-breaker.

class TowersGemini3FlashHeuristic17(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_17="towers_gemini3flash_heuristic_17"):
        super().__init__(towers_gemini3flash_heuristic_17)
        self.ring_size_map = {}  # ring -> size_index (0 is smallest)
        self.ring_order = []     # rings sorted smallest to largest
        self.goal_map = {}       # ring -> target_object
        self.on_fact_id_map = {} # (ring, target) -> global_id

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
        # 1. Map all facts
        rings = set()
        smaller_than_pairs = []
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                rings.add(args[0])
                self.on_fact_id_map[(args[0], args[1])] = f.global_id
            elif pred == "smallerThan":
                smaller_than_pairs.append((args[0], args[1]))

        # 2. Build Transitive Closure for smallerThan to determine size index
        # In Towers, size index is the number of rings strictly smaller than a ring.
        is_smaller = {r: set() for r in rings}
        for r_sub, r_super in smaller_than_pairs:
            if r_sub in rings and r_super in rings:
                is_smaller[r_super].add(r_sub)

        changed = True
        while changed:
            changed = False
            for r in rings:
                new_set = is_smaller[r].copy()
                for r_small in is_smaller[r]:
                    new_set.update(is_smaller[r_small])
                if len(new_set) > len(is_smaller[r]):
                    is_smaller[r] = new_set
                    changed = True

        self.ring_order = sorted(list(rings), key=lambda r: len(is_smaller[r]))
        for i, r in enumerate(self.ring_order):
            self.ring_size_map[r] = i

        # 3. Parse Goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for goal_fact in goal_fact_names:
            pred, args = self._parse_fact_name(goal_fact)
            if pred == "on":
                self.goal_map[args[0]] = args[1]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        n = len(self.ring_order)
        if n == 0:
            return len(task_network)

        # Determine which rings are correctly placed
        # A ring is correct if on its goal and (if goal is a ring) that ring is correct.
        # Process from largest to smallest ring.
        correct = [False] * n
        for i in range(n - 1, -1, -1):
            r = self.ring_order[i]
            target = self.goal_map.get(r)
            if not target:
                correct[i] = True # No specific goal for this ring
                continue
            
            fid = self.on_fact_id_map.get((r, target))
            if fid is not None and (state >> fid) & 1:
                if target not in self.ring_size_map:
                    # On a tower, always correct if facts match
                    correct[i] = True
                else:
                    # On a ring, correct only if target ring is correct
                    target_idx = self.ring_size_map[target]
                    if correct[target_idx]:
                        correct[i] = True
            else:
                correct[i] = False

        # Term 1: Hanoi state cost (2^k - 1 for largest misplaced ring index k)
        h_state = 0
        for i in range(n - 1, -1, -1):
            if not correct[i]:
                h_state = (1 << (i + 1)) - 1
                break

        # Term 2: Task network cost (tie-breaker)
        h_tasks = len(task_network)

        return h_state + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
