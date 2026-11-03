import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini3FlashHeuristic3(Heuristic):
    """
    Analysis of the Towers Domain:
    1. Bottleneck: The Tower of Hanoi problem is constrained by a strict sequential dependency. 
       To move a ring to its goal, all smaller rings must first be moved to an auxiliary tower. 
       This creates a recursive structure where the cost grows exponentially with the number of rings.
    2. Admissible Bounds:
       - Term 1: Recursive Hanoi Distance. The optimal number of moves required to solve a Hanoi problem 
         from any state can be computed in O(N) using a recursive distance formula. Since the planner's 
         goal matches the standard Hanoi configuration, this is a perfectly admissible and tight lower bound.
       - Term 2: Task Network tie-breaker. Adding a small magnitude (sub-unit) penalty for the number 
         of tasks remaining in the network helps guide the search toward decomposing higher-level tasks.
    """

    def __init__(self, towers_gemini3flash_heuristic_3="towers_gemini3flash_heuristic_3"):
        super().__init__(name=towers_gemini3flash_heuristic_3)
        self.ordered_rings = []
        self.tower_names = []
        self.tower_names_set = set()
        self.ring_to_on_facts = {}  # Map: ring_name -> list of (target_name, fact_id)
        self.goal_tower_name = None
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
        self.model = model
        rings = set()
        towers = set()
        smaller_than_counts = {}

        # 1. Parse domain facts to extract structure
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "smallerThan":
                # smallerThan(smaller_obj, larger_obj)
                smaller, larger = args[0], args[1]
                smaller_than_counts[smaller] = smaller_than_counts.get(smaller, 0) + 1
                rings.add(smaller)
                if not larger.startswith('t'):  # If larger is a ring, add it
                    rings.add(larger)
            elif pred == "on":
                obj, target = args[0], args[1]
                if obj not in self.ring_to_on_facts:
                    self.ring_to_on_facts[obj] = []
                self.ring_to_on_facts[obj].append((target, fact.global_id))
            elif pred == "towerTop":
                towers.add(args[1])

        self.tower_names = list(towers)
        self.tower_names_set = towers
        self.num_rings = len(rings)

        # 2. Order rings by size (smallest to largest)
        # A ring is smaller than (N-k) other rings + 3 towers.
        # Smaller rings have higher counts.
        self.ordered_rings = sorted(list(rings), key=lambda r: smaller_than_counts.get(r, 0), reverse=True)

        # 3. Identify the target tower from the goal configuration
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        if self.ordered_rings:
            largest_ring = self.ordered_rings[-1]
            for gf in goal_fact_names:
                pred, args = self._parse_fact_name(gf)
                if pred == "on" and args[0] == largest_ring:
                    # Follow 'on' chain to find the tower base
                    curr = args[1]
                    for _ in range(self.num_rings + 1):
                        if curr in self.tower_names_set:
                            self.goal_tower_name = curr
                            break
                        found_next = False
                        for gf2 in goal_fact_names:
                            p2, a2 = self._parse_fact_name(gf2)
                            if p2 == "on" and a2[0] == curr:
                                curr = a2[1]
                                found_next = True
                                break
                        if not found_next: break

        # Safety default for goal tower
        if not self.goal_tower_name and self.tower_names:
            self.goal_tower_name = "t3" if "t3" in self.tower_names_set else self.tower_names[-1]

        h_init = self._compute(initial_node.state, initial_node.task_network)
        return super().initialize(model, h_init)

    def _get_current_tower(self, ring_name, state):
        """Traces the stack to find which tower a ring is currently on."""
        curr = ring_name
        for _ in range(self.num_rings + 1):
            if curr in self.ring_to_on_facts:
                found = False
                for target_name, fid in self.ring_to_on_facts[curr]:
                    if (state >> fid) & 1:
                        if target_name in self.tower_names_set:
                            return target_name
                        curr = target_name
                        found = True
                        break
                if not found: return None
            else:
                return curr if curr in self.tower_names_set else None
        return None

    def _hanoi_dist(self, ring_idx, target_tower, ring_towers):
        """Computes recursive optimal move distance for Hanoi state."""
        if ring_idx < 0: return 0
        ring_name = self.ordered_rings[ring_idx]
        curr_tower = ring_towers.get(ring_name)
        
        if curr_tower == target_tower or curr_tower is None:
            # Ring is on the correct path, look at the next smallest ring
            return self._hanoi_dist(ring_idx - 1, target_tower, ring_towers)
        else:
            # Ring is on the wrong tower. To move it:
            # 1. Move all smaller rings to the third (auxiliary) tower
            # 2. Move this ring to the target tower
            # 3. Move all smaller rings from aux to target
            # Standard formula: dist(k-1, aux) + 2^k moves (where k-1 is the index)
            aux = target_tower
            for t in self.tower_names:
                if t != target_tower and t != curr_tower:
                    aux = t
                    break
            return self._hanoi_dist(ring_idx - 1, aux, ring_towers) + (1 << ring_idx)

    def _compute(self, state, task_network):
        if not self.ordered_rings: return 0
        
        # Find towers for all rings in current state (O(N^2))
        ring_towers = {r: self._get_current_tower(r, state) for r in self.ordered_rings}
        
        # Main admissible term: Optimal move distance
        h_dist = self._hanoi_dist(len(self.ordered_rings)-1, self.goal_tower_name, ring_towers)
        
        # Tie-breaker: prefer nodes with shorter task networks
        h_tn = len(task_network) * 0.1
        
        return h_dist + h_tn

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> int:
        h = self._compute(node.state, node.task_network)
        super().update_info(h)
        return int(h)
