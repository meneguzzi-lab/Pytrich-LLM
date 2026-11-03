from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import math

class TowersGemini3FlashHeuristic6(Heuristic):
    """
    Hanoi-specific state-based heuristic for the Towers domain.

    1. Bottleneck Analysis:
       The core constraint is the 3-peg Tower of Hanoi serial structure. Moving any ring k 
       requires moving all rings 1..k-1 to an intermediate peg. The total effort follows 
       the 2^n - 1 progression.

    2. Admissible Lower Bounds:
       - Term 1 (Hanoi Recursive Distance): Given the current tower positions of all rings,
         the minimum moves to reach the goal state in a 3-peg configuration is computable 
         recursively in O(n). This is the absolute minimum cost to satisfy ring ordering.
       - Term 2 (Misplaced Rings): A simple count of rings not on their target object.
         While weaker, it can break ties in early decomposition stages.

    3. Complexity:
       - Initialize: O(F) to map predicates, O(R^2) to rank rings.
       - Call: O(R) bit tests to find current ring positions + O(R) recursive cost calculation.
    """

    def __init__(self, towers_gemini3flash_heuristic_6="towers_gemini3flash_heuristic_6"):
        super().__init__(towers_gemini3flash_heuristic_6)
        self.rings_by_size = {}  # size (1..N) -> ring name
        self.ring_sizes = {}     # ring name -> size
        self.on_ids_by_ring = {} # ring name -> list of (object name, global_id)
        self.target_pegs = {}    # ring name -> tower name
        self.tower_set = set()
        self.num_rings = 0
        self.cache = {}

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
        ring_names = set()
        smaller_than_pairs = []
        all_objs = set()

        # Pre-parse all facts to identify objects and relationships
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred: continue
            
            all_objs.update(args)
            if pred == "smallerThan":
                smaller_than_pairs.append(args)
            elif pred == "on":
                if args[0] not in self.on_ids_by_ring:
                    self.on_ids_by_ring[args[0]] = []
                self.on_ids_by_ring[args[0]].append((args[1], fact.global_id))

        # Identify rings (those that are smaller than something else)
        for r, o in smaller_than_pairs:
            ring_names.add(r)
        
        self.tower_set = all_objs - ring_names
        self.num_rings = len(ring_names)

        # Rank rings by counting how many other rings they are smaller than
        ring_rank = {}
        for r in ring_names:
            count = sum(1 for other in ring_names if [r, other] in smaller_than_pairs)
            ring_rank[r] = count

        # Smallest ring has most 'smallerThan' relations (count N-1), size = 1
        sorted_rings = sorted(ring_names, key=lambda x: ring_rank[x], reverse=True)
        for i, r_name in enumerate(sorted_rings):
            size = i + 1
            self.rings_by_size[size] = r_name
            self.ring_sizes[r_name] = size

        # Parse Goal State to find target peg for each ring
        goal_fact_names = model.state_explicit_repr(model.goals)
        goal_on = {}
        for g_name in goal_fact_names:
            pred, args = self._parse_fact_name(g_name)
            if pred == "on":
                goal_on[args[0]] = args[1]

        for r_name in ring_names:
            curr = r_name
            visited = {curr}
            while curr in goal_on and curr not in self.tower_set:
                curr = goal_on[curr]
                if curr in visited: break
                visited.add(curr)
            self.target_pegs[r_name] = curr

        h_initial = self._compute(initial_node.state)
        return super().initialize(model, h_initial)

    def _compute(self, state_bitwise):
        if state_bitwise in self.cache:
            return self.cache[state_bitwise]

        # Determine current peg for every ring
        obj_on = {}
        for r_name, options in self.on_ids_by_ring.items():
            for o_name, fid in options:
                if (state_bitwise >> fid) & 1:
                    obj_on[r_name] = o_name
                    break

        current_pegs = {}
        for r_name in self.ring_sizes:
            curr = r_name
            visited = {curr}
            while curr in obj_on and curr not in self.tower_set:
                curr = obj_on[curr]
                if curr in visited: break
                visited.add(curr)
            current_pegs[r_name] = curr

        h_val = self._hanoi_dist(self.num_rings, current_pegs, self.target_pegs)
        self.cache[state_bitwise] = h_val
        return h_val

    def _hanoi_dist(self, size, current_pegs, targets):
        if size == 0:
            return 0
        
        ring = self.rings_by_size[size]
        c_peg = current_pegs.get(ring)
        t_peg = targets.get(ring)

        if c_peg == t_peg:
            return self._hanoi_dist(size - 1, current_pegs, targets)
        else:
            # Must move ring 'size' from c_peg to t_peg.
            # Requires all smaller rings to move to the auxiliary peg.
            aux_peg = next(p for p in self.tower_set if p != c_peg and p != t_peg)
            
            # Create effective targets for smaller subproblem
            new_targets = targets.copy()
            for i in range(1, size):
                new_targets[self.rings_by_size[i]] = aux_peg
            
            return (1 << (size - 1)) + self._hanoi_dist(size - 1, current_pegs, new_targets)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state)
        super().update_info(h_value)
        return h_value
