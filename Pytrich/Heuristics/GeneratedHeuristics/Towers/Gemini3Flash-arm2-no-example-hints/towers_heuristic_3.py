from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
from collections import defaultdict

# Bottleneck Analysis:
# The primary bottleneck in the Towers domain is the recursive serial dependency of ring movements.
# Moving a ring k to its goal position requires all rings smaller than k to be positioned on 
# the third 'buffer' tower. This implies an exponential cost growth (2^k - 1).
# 
# Admissibility Sketches:
# 1. Hanoi Distance: The minimum number of moves to relocate a tower of size k in a 3-peg setup
#    is 2^k - 1. If k is the index of the largest misplaced ring (misplaced means not on its goal
#    or resting on a misplaced ring), then 2^k - 1 is an absolute lower bound on actions required.
# 2. Task Network progress: Each move_abstract or move task in the task network represents at 
#    least one physical action (move) that must be executed. This is an independent cost term.

class TowersGemini3FlashHeuristic3(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_3="TowersGemini3FlashHeuristic3"):
        super().__init__(towers_gemini3flash_heuristic_3)
        self.rings = set()
        self.towers = set()
        self.on_fact_map = defaultdict(dict)
        self.goal_on = {}
        self.ring_to_size = {}
        self.rings_large_to_small = []

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
        smaller_than_pairs = []

        # 1. Parse all facts to identify rings, towers, and 'on' mappings
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "on":
                self.on_fact_map[args[0]][args[1]] = f.global_id
                self.rings.add(args[0])
            elif pred == "smallerThan":
                self.rings.add(args[0])
                smaller_than_pairs.append((args[0], args[1]))
            elif pred == "towerTop":
                self.towers.add(args[1])

        # 2. Identify goal positions from bitwise goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_on[args[0]] = args[1]

        # 3. Determine ring sizes based on smallerThan relations
        # Size = count of other rings that are smaller than this one
        ring_larger_than_count = {r: 0 for r in self.rings}
        for r_small, r_large in smaller_than_pairs:
            if r_small in self.rings and r_large in self.rings:
                ring_larger_than_count[r_large] += 1
        
        self.ring_to_size = ring_larger_than_count
        # Sort rings so we can process them largest-to-smallest
        self.rings_large_to_small = sorted(self.rings, key=lambda r: self.ring_to_size[r], reverse=True)

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state_bitwise, task_network):
        # Step A: Identify where every ring is currently
        current_on = {}
        for r in self.rings:
            for obj, fid in self.on_fact_map[r].items():
                if (state_bitwise >> fid) & 1:
                    current_on[r] = obj
                    break

        # Step B: Find the largest misplaced ring k
        # A ring is stable if it is at its goal position AND its support is stable.
        # Anything else must eventually be moved.
        k = 0
        stable = set(self.towers)
        for r in self.rings_large_to_small:
            target = self.goal_on.get(r)
            curr = current_on.get(r)
            
            # A ring is correctly placed if its current support matches goal support AND support is stable
            is_at_goal = (target is None or curr == target)
            if is_at_goal and curr in stable:
                stable.add(r)
            else:
                # Found the largest misplaced ring
                k = self.ring_to_size[r] + 1
                break

        # Admissible Hanoi Distance Term
        h_hanoi = (2**k - 1) if k > 0 else 0

        # Step C: Guidance term from the task network
        # We add a small penalty per task to favor shallow/completed trees
        h_tasks = 0
        for task in task_network:
            # Every move_abstract or move task represents a future move action
            if task.name.startswith("move"):
                h_tasks += 1
            else:
                # Compound tasks (shiftTower, selectDirection) represent future decompositions
                h_tasks += 0.5

        # Combine terms: Hanoi distance dominates, task count provides tie-breaking guidance
        return int(h_hanoi + 0.1 * h_tasks)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
