import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic8(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic calculates the remaining cost to solve the puzzle from a given state.
    It is based on a standard, admissible heuristic for the Tower of Hanoi problem.
    The core idea is to identify the largest ring that is not yet in its final 'settled' position
    and calculate the minimum number of moves required to place it correctly.

    The heuristic value h(n) is calculated as follows:
    1. Iterate from the largest ring (N) down to the smallest (1).
    2. For each ring 'i', if its current peg is not its target peg, it must be moved.
    3. The cost of moving ring 'i' is at least 2^(i-1) (1 move for ring 'i' itself, and 2^(i-1)-1
       moves to relocate the stack of i-1 smaller rings).
    4. When such a ring 'i' is found, we add 2^(i-1) to the heuristic value. We then update the
       target peg for all smaller rings to be the 'auxiliary' peg, as they must be moved there
       to allow ring 'i' to move.
    5. This process continues, summing the costs for all rings that are misplaced relative to the
       larger rings.

    This heuristic is admissible because it calculates the exact number of moves in a relaxed version
    of the problem where larger disks are ignored once they are placed on their target peg.
    """

    def __init__(self, towers_gemini25pro_heuristic_8):
        super().__init__(name="towers_gemini25pro_heuristic_8")
        self.model = None
        self.rings_by_size = []
        self.goal_peg = {}
        self.on_facts = {}
        self.peg_map = {}
        self.fact_name_to_id = {}
        self.all_towers = []

    def initialize(self, model, initial_node):
        """
        Pre-computes necessary information from the model, such as ring sizes,
        goal configuration, and fact IDs for efficient lookup during search.
        """
        self.model = model
        self.fact_name_to_id = {fact.name: fact.global_id for fact in model.facts}

        # 1. Discover all objects (rings, towers) by parsing fact names
        all_objects = set()
        for fact in model.facts:
            clean_name = fact.name.strip()
            if clean_name.startswith('(') and clean_name.endswith(')'):
                parts = clean_name[1:-1].split()
                for p in parts[1:]:
                    all_objects.add(p)

        rings = sorted([o for o in all_objects if o.startswith('r')], key=lambda r: int(r[1:]))
        towers = sorted([o for o in all_objects if o.startswith('t')])
        self.rings_by_size = rings
        self.all_towers = towers

        # 2. Map towers to integer pegs (0, 1, 2)
        if len(towers) == 3:
            for i, tower in enumerate(towers):
                self.peg_map[tower] = i

        # 3. Determine goal peg from the initial 'shiftTower' task
        dest_peg_name = None
        if model.initial_tn and model.initial_tn[0]:
            task_name = model.initial_tn[0].name
            if "shiftTower" in task_name:
                clean_name = task_name.strip()
                if clean_name.startswith('(') and clean_name.endswith(')'):
                    parts = clean_name[1:-1].split()
                    # Expected format: (task_name (shiftTower src aux dest))
                    if len(parts) >= 5 and parts[1] == "shiftTower":
                        dest_peg_name = parts[4]

        if dest_peg_name and dest_peg_name in self.peg_map:
            dest_peg_id = self.peg_map[dest_peg_name]
            for ring in self.rings_by_size:
                self.goal_peg[ring] = dest_peg_id
        else:
            # Heuristic is invalid if goal cannot be determined.
            return super().initialize(model, 0)

        # 4. Pre-compute 'on' fact IDs for fast state lookups
        for ring in self.rings_by_size:
            self.on_facts[ring] = {}
            possible_supports = self.rings_by_size + self.all_towers
            for support in possible_supports:
                if ring != support:
                    fact_name = f"(on {ring} {support})"
                    if fact_name in self.fact_name_to_id:
                        fact_id = self.fact_name_to_id[fact_name]
                        self.on_facts[ring][support] = fact_id
        
        h_value = self._calculate_node_heuristic(initial_node) if hasattr(initial_node, 'state') else 0
        return super().initialize(model, h_value)

    def _get_peg_of(self, obj_name, current_on_state):
        """Helper function to find the base tower for any object."""
        peg_candidate = obj_name
        visited = {obj_name}
        while peg_candidate not in self.peg_map:
            if peg_candidate not in current_on_state:
                return -1 # Error: Ring is not on anything
            peg_candidate = current_on_state[peg_candidate]
            if peg_candidate in visited:
                return -1 # Error: Cycle detected
            visited.add(peg_candidate)
        return self.peg_map[peg_candidate]

    def _calculate_node_heuristic(self, node):
        """Calculates the heuristic value for a given search node."""
        if not self.goal_peg or not hasattr(node, 'state') or len(self.peg_map) != 3:
            return 0

        # 1. Build current 'on' configuration from the node's state bitmask
        current_on_state = {}
        for ring, supports in self.on_facts.items():
            for support, fact_id in supports.items():
                if (node.state >> fact_id) & 1:
                    current_on_state[ring] = support
                    break

        # 2. Determine the current peg for each ring
        current_peg_state = {}
        for ring in self.rings_by_size:
            if ring in current_on_state:
                peg_id = self._get_peg_of(ring, current_on_state)
                if peg_id == -1:
                    return float('inf') # Invalid state
                current_peg_state[ring] = peg_id
            else:
                return float('inf') # Invalid state, ring isn't on anything

        # 3. Calculate heuristic value using the standard formula
        h_val = 0
        num_rings = len(self.rings_by_size)
        temp_dest_peg = self.goal_peg.copy()

        for i in range(num_rings, 0, -1):
            ring_name = self.rings_by_size[i - 1]
            if current_peg_state.get(ring_name) != temp_dest_peg.get(ring_name):
                h_val += 2**(i - 1)
                other_peg = 3 - current_peg_state[ring_name] - temp_dest_peg[ring_name]
                for j in range(1, i):
                    smaller_ring_name = self.rings_by_size[j - 1]
                    temp_dest_peg[smaller_ring_name] = other_peg
        
        return h_val

    def __call__(self, parent_node, node):
        """Callable method for the planner to get the heuristic value."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value