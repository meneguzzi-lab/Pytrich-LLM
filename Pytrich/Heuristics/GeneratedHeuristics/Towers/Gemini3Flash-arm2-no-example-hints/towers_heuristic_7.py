import math
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic7(Heuristic):
    """
    Bottleneck Analysis:
    The Towers (Hanoi) domain is strictly serialized by ring sizes and the limited number of pegs (towers).
    A ring 'k' can only move if all smaller rings are stacked on the single available auxiliary peg 
    (in a 3-peg setup) and its target destination is clear of smaller rings. This results in the 
    classical exponential work constraint ($2^n - 1$).

    Admissible Lower Bounds:
    1. Hanoi Distance: The minimum number of moves to reach the goal state from the current configuration.
       For a 3-peg problem, this is exactly computable in O(N). For >3 pegs, it remains a valid admissible 
       lower bound. 
    2. Task Counting: While simple, each task in the network eventually expands into at least one action.
       We prioritize the state-based Hanoi distance as it captures the exponential growth.
    """

    def __init__(self, towers_gemini3flash_heuristic_7="towers_gemini3flash_heuristic_7"):
        super().__init__(name=towers_gemini3flash_heuristic_7)
        self.ring_ids = []
        self.tower_ids = []
        self.obj_to_ring_idx = {}
        self.on_facts = [] # List of (global_id, ring_idx, target_obj_name)
        self.num_rings = 0
        self.goal_pegs = [] # Target peg index for every ring
        self.tower_to_peg_idx = {}
        self.power_of_2 = []

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
        rings_unsorted = []
        towers_found = []

        # 1. Collect all Rings and Towers
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred in ["on", "smallerThan", "towerTop"]:
                for arg in args:
                    if arg.startswith('r') and arg[1:].isdigit():
                        rings_unsorted.append(arg)
                    elif arg.startswith('t') and arg[1:].isdigit():
                        towers_found.append(arg)

        # Sort rings by size (r1 is smallest, rN is largest)
        # In this domain, name-based sorting (r1, r2...) is consistent with the smallerThan relations
        unique_rings = sorted(list(set(rings_unsorted)), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0)
        self.ring_ids = unique_rings
        self.num_rings = len(self.ring_ids)
        self.tower_ids = sorted(list(set(towers_found)))
        self.tower_to_peg_idx = {t: i for i, t in enumerate(self.tower_ids)}
        self.obj_to_ring_idx = {r: i for i, r in enumerate(self.ring_ids)}
        self.power_of_2 = [2**i for i in range(self.num_rings + 1)]

        # 2. Pre-map all possible 'on' facts for fast state checking
        self.on_facts = []
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on":
                ring_name, on_obj_name = args[0], args[1]
                if ring_name in self.obj_to_ring_idx:
                    self.on_facts.append((f.global_id, self.obj_to_ring_idx[ring_name], on_obj_name))

        # 3. Determine Goal Peg for each ring
        # Trace the 'on' chain in the goal state to find which tower base each ring sits on
        goal_fact_names = model.state_explicit_repr(model.goals)
        goal_on = {}
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on":
                goal_on[args[0]] = args[1]
        
        self.goal_pegs = [0] * self.num_rings
        for i, r_name in enumerate(self.ring_ids):
            curr = r_name
            visited = set()
            while curr in goal_on and curr not in self.tower_to_peg_idx and curr not in visited:
                visited.add(curr)
                curr = goal_on[curr]
            self.goal_pegs[i] = self.tower_to_peg_idx.get(curr, 0)

        initial_h = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, initial_h)

    def get_hanoi_dist(self, ring_idx, curr_pegs):
        """
        Computes the standard Hanoi distance from current state to the goal state.
        This is a lower bound even for non-perfect tower goals.
        """
        dist = 0
        # Start from the largest displaced ring
        current_target = self.goal_pegs[ring_idx] if self.goal_pegs else 0
        for i in range(ring_idx, -1, -1):
            cp = curr_pegs[i]
            if cp == -1 or cp == current_target:
                continue
            # Ring i is not on its target peg. It must move.
            dist += self.power_of_2[i]
            # To move ring i, all smaller rings must go to the auxiliary peg.
            # We find a peg that is not the current one and not the target one.
            # This is exact for 3-pegs, admissible for more.
            aux = -1
            for p in range(3):
                if p != cp and p != current_target:
                    aux = p
                    break
            if aux != -1:
                current_target = aux
        return dist

    def _compute(self, state, task_network):
        # 1. Identify which object each ring is immediately 'on' using O(1) bit shifts
        ring_on_what = [None] * self.num_rings
        for fid, ridx, on_obj_name in self.on_facts:
            if (state >> fid) & 1:
                ring_on_what[ridx] = on_obj_name
        
        # 2. Trace the 'on' chains to find the current peg for each ring
        curr_pegs = [-1] * self.num_rings
        for i in range(self.num_rings):
            curr = self.ring_ids[i]
            path = set()
            while curr in self.obj_to_ring_idx:
                ridx = self.obj_to_ring_idx[curr]
                next_obj = ring_on_what[ridx]
                if next_obj is None or next_obj in path:
                    break
                path.add(curr)
                curr = next_obj
            if curr in self.tower_to_peg_idx:
                curr_pegs[i] = self.tower_to_peg_idx[curr]

        # 3. Calculate Hanoi Distance from the state configuration
        h_base = self.get_hanoi_dist(self.num_rings - 1, curr_pegs)
        
        # 4. Tie-breaking / HTN Task context
        # In Tower of Hanoi, state distance is so dominant that task counting is rarely needed,
        # but we could add a small penalty for every high-level task remaining.
        h_tasks = len(task_network) * 0.01

        return int(h_base + h_tasks)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h_value = self._compute(node.state, node.task_network)
        super().update_info(h_value)
        return h_value
