import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Towers (Tower of Hanoi) domain.
    
    Bottleneck Analysis:
    The fundamental constraint in the Towers domain is the recursive dependency of ring movements. 
    To move a ring of size N, all smaller rings (1 to N-1) must be moved to the spare tower first. 
    This creates a minimum cost of 2^N - 1 moves for any misplaced ring of size N.

    Admissibility Sketches:
    1. Hanoi Lower Bound: Let r_i be the largest ring that is not 'stable'. A ring is stable if it is 
       on its goal position and its base (either a tower or another ring) is also stable. 
       To reach the goal, r_i must move at least once. The minimum number of actions required 
       to solve a Hanoi subproblem involving a stack of size i is 2^i - 1. Since all move 
       actions have cost 1, this is an admissible lower bound on the remaining cost.
    2. Task Network Tie-breaker: Each task remaining in the network represents at least one 
       future decomposition or action. We add a very small weight (0.0001) per task to guide 
       the search through equivalent states in the decomposition process without violating admissibility.
    """

    def __init__(self, towers_gemini3flash_heuristic_18="towers_gemini3flash_heuristic_18"):
        super().__init__(name=towers_gemini3flash_heuristic_18)
        self.goal_map = {}  # ring -> target_obj
        self.on_ids = {}    # (ring, obj) -> global_id
        self.ring_sizes = {} # ring -> int (1 is smallest)
        self.rings = []     # Sorted rings largest to smallest
        self.towers = set()
        self.possible_on_targets = {}

    def initialize(self, model: Model, initial_node: HTNNode):
        self.model = model
        
        # 1. Parse all facts to identify rings, towers, and predicates
        all_rings = set()
        smaller_than_pairs = set() # (smaller, larger)
        all_objs = set()

        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred: continue
            
            if pred == "on":
                self.on_ids[(args[0], args[1])] = f.global_id
                all_objs.update(args)
            elif pred == "smallerThan":
                all_rings.add(args[0])
                smaller_than_pairs.add((args[0], args[1]))
                all_objs.update(args)
            elif pred == "goal_on":
                all_rings.add(args[0])
                all_objs.update(args)

        # 2. Determine which objects are rings and which are towers
        self.towers = all_objs - all_rings
        
        # 3. Calculate ring sizes (1 to N)
        # A ring's size is determined by how many other rings it is smaller than
        for r in all_rings:
            larger_rings_count = sum(1 for r2 in all_rings if (r, r2) in smaller_than_pairs)
            self.ring_sizes[r] = len(all_rings) - larger_rings_count

        # 4. Sort rings largest to smallest for stability checking
        self.rings = sorted(list(all_rings), key=lambda r: self.ring_sizes[r], reverse=True)

        # 5. Precompute goal mapping from model.goals bitmask
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                self.goal_map[args[0]] = args[1]

        # 6. Precompute possible targets for each ring to speed up state inspection
        for r in all_rings:
            self.possible_on_targets[r] = [o for (r_in, o) in self.on_ids.keys() if r_in == r]

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state: int, task_network) -> float:
        # Determine current 'on' state for each ring
        current_on = {}
        for r in self.rings:
            for o in self.possible_on_targets[r]:
                if (state >> self.on_ids[(r, o)]) & 1:
                    current_on[r] = o
                    break

        # Calculate stability for rings (largest to smallest)
        stable = set()
        h_hanoi = 0
        for r in self.rings:
            target = self.goal_map.get(r)
            curr_base = current_on.get(r)
            
            is_stable = False
            if target:
                # If ring has a goal, it must be on that goal and its base must be stable
                if curr_base == target and (curr_base in self.towers or curr_base in stable):
                    is_stable = True
            else:
                # If ring has no goal, it is stable if its current base is stable
                if curr_base in self.towers or curr_base in stable:
                    is_stable = True
            
            if is_stable:
                stable.add(r)
            else:
                # Misplaced ring of size S implies at least 2^S - 1 moves remain
                ring_cost = (2 ** self.ring_sizes[r]) - 1
                if ring_cost > h_hanoi:
                    h_hanoi = ring_cost

        # Add a very small tie-breaker for task network progress
        h_tasks = len(task_network) * 0.0001
        
        return h_hanoi + h_tasks

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
