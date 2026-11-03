from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Bottleneck Analysis:
# The bottleneck in the Towers domain (a variation of the Towers of Hanoi) is the exponential movement sequence.
# To move a stack of N rings, a specific recursive sequence of moves is required where the largest disk must 
# be cleared of all smaller disks before it can be moved to its target tower. This creates an unavoidable 
# cost of 2^k - 1 moves where k is the size of the misplaced sub-tower.

# Admissibility Sketches:
# Term 1 (State-based Exponential): We identify the largest ring 'k' (index 1 to N) that is not in its goal position
# or whose supporting structure is not yet stable. The number of moves needed to reach the goal state from 
# such a configuration is at least 2^k - 1. This is a classic lower bound for the Towers of Hanoi problem.
# Term 2 (Task-based progress): We add the count of remaining tasks in the task network. This is a non-negative 
# measure of the complexity remaining in the decomposition process, acting as a small guidance term.

class TowersGemini3FlashHeuristic5(Heuristic):
    def __init__(self, towers_gemini3flash_heuristic_5="TowersGemini3FlashHeuristic5"):
        super().__init__(towers_gemini3flash_heuristic_5)
        self.rings_ordered = []
        self.on_fact_ids = {} # (ring, target) -> global_id
        self.goal_target = {} # ring -> target
        self.num_rings = 0

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
        # 1. Map all relevant fact IDs
        rings_detected = set()
        smaller_than_pairs = set()

        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "on":
                self.on_fact_ids[(args[0], args[1])] = fact.global_id
                rings_detected.add(args[0])
            elif pred == "smallerThan":
                smaller_than_pairs.add((args[0], args[1]))
                rings_detected.add(args[0])

        # 2. Extract goals from model
        goal_fact_names = model.state_explicit_repr(model.goals)
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on" and len(args) == 2:
                self.goal_target[args[0]] = args[1]
                rings_detected.add(args[0])

        # 3. Order rings by size (smallest to largest)
        # A ring is smaller than all rings it has a smallerThan relation with.
        # In Hanoi pfiles, r1 is usually smaller than r2, r2 < r3, etc.
        smaller_counts = {}
        for r in rings_detected:
            count = 0
            for other in rings_detected:
                if (r, other) in smaller_than_pairs:
                    count += 1
            smaller_counts[r] = count
        
        # Sorting by count of smallerThan relations descending: 
        # r1 is smaller than 19 others (highest count), r20 is smaller than 0 others (lowest count).
        self.rings_ordered = sorted(list(rings_detected), key=lambda x: smaller_counts.get(x, 0), reverse=True)
        self.num_rings = len(self.rings_ordered)

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def _compute(self, state, task_network):
        # Identify the largest misplaced ring k (index 1 to N)
        k = 0
        for i in range(self.num_rings - 1, -1, -1):
            ring = self.rings_ordered[i]
            target = self.goal_target.get(ring)
            if not target:
                continue
            
            fid = self.on_fact_ids.get((ring, target))
            if fid is None:
                # Fact not even defined, must be misplaced
                k = i + 1
                break
            
            is_on = (state >> fid) & 1
            if not is_on:
                k = i + 1
                break
            # Even if it's on target, we must check the stack stability from bottom to top.
            # Because we iterate from largest down, if a larger ring below was misplaced,
            # we would have already broken and set k to that larger index.

        # Standard Hanoi exponential cost term
        # h = 2^k - 1
        h_state = (1 << k) - 1

        # Task network penalty: guides search through decomposition steps
        h_tasks = len(task_network)

        return h_state + h_tasks

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
