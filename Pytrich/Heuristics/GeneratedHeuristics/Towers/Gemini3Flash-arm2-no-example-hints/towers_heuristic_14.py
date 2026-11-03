from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic14(Heuristic):
    """
    Bottleneck: The recursive dependency of ring movements in the Towers of Hanoi problem, 
    where moving a larger ring requires all smaller rings to be moved to a spare peg.

    Admissibility Sketches:
    - Term 1: Optimal Hanoi distance (2^k - 1 moves). This is the exact minimum number of 
      actions required to move k rings between three pegs, making it a tightly admissible bound.
    - Memoization: Used to avoid re-computing the O(N) distance for the same state.
    - Efficiency: O(N) calculation per node using precomputed rank and peg-assignment chains.
    """

    def __init__(self, towers_gemini3flash_heuristic_14="towers_gemini3flash_heuristic_14"):
        super().__init__(towers_gemini3flash_heuristic_14)
        self.rings = set()
        self.towers = set()
        self.sorted_rings = []
        self.on_facts_by_ring = {}
        self.target_pegs = {}
        self.spare_peg_map = {}
        self.pow2 = []
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
        # 1. Identify Rings and Towers
        temp_rings = set()
        for fid in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(fid)
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                temp_rings.add(args[0])
        
        temp_towers = set()
        for fid in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(fid)
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                if args[1] not in temp_rings:
                    temp_towers.add(args[1])
            elif pred == "towerTop":
                if args[0] == args[1]:
                    temp_towers.add(args[1])
        
        self.rings = temp_rings
        self.towers = temp_towers

        # 2. Rank rings by size using smallerThan facts
        ring_smaller_counts = {r: 0 for r in self.rings}
        for fid in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(fid)
            pred, args = self._parse_fact_name(f.name)
            if pred == "smallerThan" and args[0] in self.rings and args[1] in self.rings:
                ring_smaller_counts[args[0]] += 1
        
        # Largest to smallest sorting (largest ring has 0 rings it is smaller than)
        self.sorted_rings = sorted(list(self.rings), key=lambda r: ring_smaller_counts.get(r, 0))
        
        # 3. Precompute powers of 2
        self.pow2 = [1 << i for i in range(len(self.sorted_rings) + 1)]

        # 4. Map 'on' facts by ring for fast lookup
        self.on_facts_by_ring = {r: {} for r in self.rings}
        for fid in range(model.ifacts_init, model.ifacts_end + 1):
            f = model.get_component(fid)
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and args[0] in self.rings:
                self.on_facts_by_ring[args[0]][args[1]] = f.global_id

        # 5. Determine target peg for each ring from goal state
        goal_on_map = {}
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on":
                goal_on_map[args[0]] = args[1]
        
        # Process largest-to-smallest to resolve chain targets O(N)
        self.target_pegs = {}
        for r in self.sorted_rings:
            if r in goal_on_map:
                dest = goal_on_map[r]
                if dest in self.towers:
                    self.target_pegs[r] = dest
                elif dest in self.target_pegs:
                    self.target_pegs[r] = self.target_pegs[dest]

        # 6. Precompute spare pegs (assumes 3-tower instances as per domain standard)
        towers_list = sorted(list(self.towers))
        if len(towers_list) >= 3:
            for i in towers_list:
                for j in towers_list:
                    if i == j: continue
                    for k in towers_list:
                        if k != i and k != j:
                            self.spare_peg_map[(i, j)] = k
                            break

        h_value = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_value)

    def _compute(self, state, task_network):
        if not task_network:
            return 0
        
        if state in self.memo:
            return self.memo[state]
        
        # Step 1: Trace current pegs for all rings O(N)
        curr_peg = {}
        for r in self.sorted_rings: # Largest to smallest
            found = False
            if r in self.on_facts_by_ring:
                for obj, fid in self.on_facts_by_ring[r].items():
                    if (state >> fid) & 1:
                        if obj in self.towers:
                            curr_peg[r] = obj
                        else:
                            # obj is a larger ring, we already have its peg
                            curr_peg[r] = curr_peg.get(obj)
                        found = True
                        break
            if not found: 
                # Safety fallback: assume it is on its current peg if not found in bitset
                pass

        # Step 2: Compute recursive Hanoi distance O(N)
        h = 0
        current_targets = self.target_pegs.copy()
        n = len(self.sorted_rings)
        for i in range(n):
            r = self.sorted_rings[i]
            c = curr_peg.get(r)
            d = current_targets.get(r, c) # Stay if no target
            
            if c and d and c != d:
                # Distance contribution is 2^(rank-1)
                # rank is n-i where r1 (smallest) is i=n-1
                rank = n - 1 - i
                h += self.pow2[rank]
                # To move ring i, all smaller rings (j > i) must move to the spare peg
                spare = self.spare_peg_map.get((c, d))
                if spare:
                    for j in range(i + 1, n):
                        current_targets[self.sorted_rings[j]] = spare

        self.memo[state] = h
        return h

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
