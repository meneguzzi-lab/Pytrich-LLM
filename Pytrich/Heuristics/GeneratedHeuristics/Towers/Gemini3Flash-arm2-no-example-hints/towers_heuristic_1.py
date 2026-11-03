from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.
    
    Bottleneck Analysis:
    The physical constraint that larger rings cannot be placed on smaller ones creates a strict 
    dependency. Moving the k-th ring requires all k-1 smaller rings to be relocated to a different peg.
    This recursive structure means the minimum cost from any state to the goal is exactly determined 
    by the current and target peg positions of every ring.

    Admissibility Sketch:
    The heuristic calculates the exact optimal number of 'move' actions required to reach the goal 
    using the standard iterative Hanoi distance algorithm (sum of 2^(i-1) for necessary moves). 
    Since every 'move' action in the domain has a cost of 1, and 'move' is the only primitive 
    action that changes state, this value is an admissible (and here, quite accurate) lower bound.
    """

    def __init__(self, towers_gemini3flash_heuristic_1="TowersGemini3FlashHeuristic1"):
        super().__init__(towers_gemini3flash_heuristic_1)
        self.ring_name_to_size = {}
        self.tower_name_to_idx = {}
        self.on_fact_ids = []
        self.on_fact_map = {} # fact_id -> (ring_name, parent_name)
        self.target_pegs = {} # size -> tower_idx (1, 2, or 3)
        self.num_rings = 0
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
        # 1. Identify all Rings and Towers
        ring_names = set()
        tower_names = set()
        smaller_than_pairs = []
        
        # First pass: find objects and predicates
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" or pred == "goal_on":
                ring_names.add(args[0])
            elif pred == "smallerThan":
                ring_names.add(args[0])
                smaller_than_pairs.append((args[0], args[1]))
            elif pred == "towerTop":
                tower_names.add(args[1])

        # 2. Map Rings to Sizes (1 to N)
        # size = count of rings smaller than this ring + 1
        for r in ring_names:
            size = 1
            for other_r in ring_names:
                if (other_r, r) in smaller_than_pairs:
                    size += 1
            self.ring_name_to_size[r] = size
        
        self.num_rings = len(ring_names)
        self.ring_indices = list(range(1, self.num_rings + 1))

        # 3. Map Towers to indices (1, 2, 3)
        sorted_towers = sorted(list(tower_names))
        self.tower_name_to_idx = {name: i+1 for i, name in enumerate(sorted_towers)}

        # 4. Map 'on' fact IDs for fast state checking
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and args[0] in self.ring_name_to_size:
                self.on_fact_ids.append(f.global_id)
                self.on_fact_map[f.global_id] = (args[0], args[1])

        # 5. Determine Target Pegs (where rings go in goal)
        goal_on_map = {}
        # Check initial state for 'goal_on' hints
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "goal_on":
                if (initial_node.state >> f.global_id) & 1:
                    goal_on_map[args[0]] = args[1]
        
        # Augment with bitmask goals
        goal_fact_names = model.state_explicit_repr(model.goals)
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "on":
                goal_on_map[args[0]] = args[1]

        # Trace target pegs
        for r_name in ring_names:
            curr = r_name
            for _ in range(self.num_rings + 1):
                parent = goal_on_map.get(curr)
                if not parent: break
                if parent in self.tower_name_to_idx:
                    self.target_pegs[self.ring_name_to_size[r_name]] = self.tower_name_to_idx[parent]
                    break
                curr = parent

        h_initial = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_initial)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value

    def _compute(self, state, task_network):
        if state in self.memo:
            return self.memo[state]
        
        if self.num_rings == 0:
            return 0

        # 1. Map current peg for each ring index (size)
        on_parent = {}
        for fid in self.on_fact_ids:
            if (state >> fid) & 1:
                r_name, p_name = self.on_fact_map[fid]
                on_parent[r_name] = p_name
        
        current_pegs = {}
        for r_name, size in self.ring_name_to_size.items():
            curr = r_name
            for _ in range(self.num_rings + 1):
                parent = on_parent.get(curr)
                if not parent: break
                if parent in self.tower_name_to_idx:
                    current_pegs[size] = self.tower_name_to_idx[parent]
                    break
                curr = parent

        # 2. Iterative Hanoi Distance Logic
        dist = 0
        temp_targets = dict(self.target_pegs)
        for size in range(self.num_rings, 0, -1):
            c_peg = current_pegs.get(size)
            t_peg = temp_targets.get(size)
            # If ring is not on its target peg, it must be moved
            if c_peg and t_peg and c_peg != t_peg:
                dist += (1 << (size - 1))
                # All smaller rings must be moved to the auxiliary peg
                other_peg = 6 - c_peg - t_peg
                for s in range(1, size):
                    temp_targets[s] = other_peg
        
        # Small tie-breaker for task network progress
        h = int(dist)
        self.memo[state] = h
        return h
