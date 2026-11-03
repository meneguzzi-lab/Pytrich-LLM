import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model
import re

class TowersGemini25ProHeuristic5(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic implements the Korn-Dorfman algorithm, which provides an exact
    (admissible and consistent) calculation of the minimum number of moves remaining
    to solve the Towers of Hanoi puzzle from any given state. It works by iterating
    through the rings (disks) from largest to smallest, determining if each ring is
    on its correct target peg for the current subproblem. If a ring is misplaced, the
    heuristic adds the cost of moving all smaller rings (2^k) to the total and
    updates the target for the next smaller set of rings.

    Design Rationale:
    1.  Domain Analysis: The `towers` domain is a hierarchical representation of the
        classic Towers of Hanoi problem. The core challenge is the exponential search
        space (2^N-1 optimal moves for N rings). A simple goal-counting heuristic would be
        uninformative.
    2.  State-Dependent Heuristic: The optimal action depends entirely on the current
        configuration of rings. Therefore, a state-based heuristic is essential.
        This heuristic directly inspects the state of the board at each node.
    3.  Leveraging Domain Knowledge: We leverage the well-known optimal recursive
        solution to Towers of Hanoi. The cost to move a stack of `k` rings is 2^k - 1.
        The Korn-Dorfman algorithm efficiently uses this knowledge to calculate the
        remaining cost without solving subproblems.
    4.  Admissibility and Accuracy: This heuristic is not just a lower bound but the
        *exact* cost-to-go, making it perfectly admissible (h(n) = h*(n)). This will
        guide an A* search directly along an optimal path, drastically reducing the
        number of expanded nodes.
    5.  Efficiency: The heuristic requires parsing the state to find the location of each
        ring. This is the main computational cost. The calculation itself is a single
        loop over the rings. For efficiency, powers of 2 are pre-calculated during
        initialization.
    6.  Scalability: The heuristic's complexity is linear with the number of rings for
        each state evaluation, making it efficient even for larger instances like `pfile_20`
        with 20 rings.
    """
    def __init__(self, towers_gemini25pro_heuristic_5=1.0):
        # The parameter is not used but required by the spec.
        self.weight = towers_gemini25pro_heuristic_5
        super().__init__(name="towers_gemini_25_pro_heuristic_5")
        self.preprocessing_time = 0
        self.model = None
        self.rings = []
        self.towers = []
        self.ring_order = []  # Sorted from smallest to largest
        self.main_source = None
        self.main_dest = None
        self.main_aux = None
        self.powers_of_2 = []

    def initialize(self, model, initial_node):
        start_time = time.time()
        self.model = model

        self._extract_objects()
        self._determine_ring_order()
        self._extract_main_task_info()

        if self.rings:
            self.powers_of_2 = [2**i for i in range(len(self.rings))]

        self.preprocessing_time = time.time() - start_time
        
        # Calculate heuristic for the initial node
        initial_h_value = self._calculate_heuristic_for_state(initial_node.state)
        return super().initialize(model, initial_h_value)

    def _extract_objects(self):
        """Extract rings and towers by parsing all grounded facts in the model."""
        rings = set()
        towers = set()

        for fact in self.model.facts:
            fact_name = fact.name.strip("()").lower()
            parts = fact_name.split()
            predicate = parts[0]
            args = parts[1:]

            if not args:
                continue

            if predicate in ["on", "goal_on"]:
                if args[0].startswith('r'): rings.add(args[0])
                if args[1].startswith('r'): rings.add(args[1])
                elif args[1].startswith('t'): towers.add(args[1])
            elif predicate == "towertop":
                if args[0].startswith('r'): rings.add(args[0])
                elif args[0].startswith('t'): towers.add(args[0])
                if args[1].startswith('t'): towers.add(args[1])
            elif predicate == "smallerthan":
                if args[0].startswith('r'): rings.add(args[0])
                if args[1].startswith('r'): rings.add(args[1])
                elif args[1].startswith('t'): towers.add(args[1])

        self.rings = sorted(list(rings), key=lambda r: int(r[1:]))
        self.towers = sorted(list(towers), key=lambda t: int(t[1:]))

    def _determine_ring_order(self):
        """Determine the size order of rings based on their names (e.g., r1, r2)."""
        self.ring_order = sorted(self.rings, key=lambda r: int(r[1:]))

    def _extract_main_task_info(self):
        """Extract source, destination, and auxiliary towers from the initial task."""
        initial_task_name = self.model.initial_tn[0].name.lower()
        match = re.match(r"^\(shifttower (.+) (.+) (.+)\)$", initial_task_name)
        if match:
            self.main_source, self.main_aux, self.main_dest = match.groups()
        elif len(self.towers) >= 3:
            self.main_source, self.main_aux, self.main_dest = self.towers[0], self.towers[1], self.towers[2]

    def _get_ring_positions(self, state_bitmask):
        """For a given state, determine which tower each ring is on."""
        on_mapping = {}
        state_facts = self.model.state_explicit_repr(state_bitmask)
        for fact_str in state_facts:
            if fact_str.startswith("(on "):
                parts = fact_str.strip("()").lower().split()
                on_mapping[parts[1]] = parts[2]
        
        ring_locations = {}
        for ring in self.rings:
            current_obj = ring
            while current_obj in on_mapping:
                obj_under = on_mapping[current_obj]
                if obj_under in self.towers:
                    ring_locations[ring] = obj_under
                    break
                current_obj = obj_under
        return ring_locations

    def _calculate_heuristic_for_state(self, state_bitmask):
        """
        Calculates the Korn-Dorfman heuristic for the Towers of Hanoi problem.
        """
        if not self.rings or not self.main_dest:
            return 0
            
        h = 0
        ring_positions = self._get_ring_positions(state_bitmask)
        
        target_peg = self.main_dest
        
        # Iterate from the largest ring down to the smallest
        for i in range(len(self.ring_order) - 1, -1, -1):
            ring = self.ring_order[i]
            
            if ring not in ring_positions:
                return 0

            current_peg = ring_positions[ring]
            
            if current_peg != target_peg:
                h += self.powers_of_2[i]
                aux_peg = [t for t in self.towers if t != current_peg and t != target_peg][0]
                target_peg = aux_peg

        return h

    def __call__(self, parent_node, node):
        """
        Calculate the heuristic value for the given search node.
        """
        h_value = self._calculate_heuristic_for_state(node.state)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"TowersGemini25ProHeuristic5(weight={self.weight})"

    def __str__(self):
        return f"TowersGemini25ProHeuristic5(weight={self.weight})"

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro Heuristic (Korn-Dorfman) 5 info:\n"
            f"\tName: {self.name}\n"
            f"\tRings found: {len(self.rings)}\n"
            f"\tTowers found: {len(self.towers)}\n"
            f"\tMain Task: Move from {self.main_source} to {self.main_dest} using {self.main_aux}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
