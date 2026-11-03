import time
import re
from collections import defaultdict
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic25(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic is based on the classical optimal solution for the Towers of Hanoi problem.
    The core idea is to identify the largest ring that is not in its final, "settled" position.
    A ring is considered settled if it is resting on its goal object (another ring or a tower)
    and all rings larger than it are also settled.

    The heuristic value is an estimate of the minimum number of moves required to solve the
    problem from the current state. This is calculated as 2^k - 1, where 'k' is the size of
    the largest unsettled ring. This value represents the cost of solving the subproblem of
    moving a stack of 'k' rings to their correct final destination, which is an admissible
    (never overestimates the true cost) and highly informative estimate for this domain.

    During initialization, the heuristic pre-processes the model to:
    1. Parse the goal configuration from the `(goal_on ...)` predicates.
    2. Determine a total ordering of rings by size, using their names (e.g., 'r1', 'r2') as a
       primary method and a topological sort of `(smallerThan ...)` predicates as a fallback.

    For each state evaluation, the heuristic:
    1. Reconstructs the current configuration of rings on towers from the bitwise state representation.
    2. Iterates from the largest ring downwards to find the first one that is not on its goal object.
    3. Calculates the heuristic value based on the size of this largest unsettled ring.
    """
    def __init__(self, towers_gemini25pro_heuristic_25):
        super().__init__(name="towers_gemini25pro_heuristic_25")
        self.preprocessing_time = 0.0
        self.goal_on_map = {}
        self.rings = []
        self.ring_order = {}
        self.param = towers_gemini25pro_heuristic_25

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by parsing goal state and ordering rings.
        """
        start_time = time.time()
        self.model = model

        self._parse_goal_state()
        self._order_rings()

        self.preprocessing_time = time.time() - start_time
        h_value = self.__call__(None, initial_node)
        return super().initialize(model, h_value)

    def _parse_goal_state(self):
        """
        Parses `goal_on` predicates from the fact list to build the goal configuration.
        """
        goal_on_regex = re.compile(r"^goal_on\s+\(\s*(\S+)\s*,\s*(\S+)\s*\)$")
        for fact in self.model.facts:
            fact_name_no_space = fact.name.replace(" ", "")
            match = goal_on_regex.match(fact_name_no_space)
            if match:
                ring, on_obj = match.groups()
                self.goal_on_map[ring] = on_obj

    def _order_rings(self):
        """
        Establishes a size-based order for all rings.
        Tries to sort numerically by name (r1, r2, ...) first.
        Falls back to a topological sort if numerical sort fails.
        """
        all_rings = set(self.goal_on_map.keys())
        if not all_rings:
            on_regex = re.compile(r"^on\s+\(\s*(r\d+)\s*,\s*(\S+)\s*\)$")
            for fact in self.model.facts:
                match = on_regex.match(fact.name.replace(" ", ""))
                if match:
                    all_rings.add(match.group(1))

        try:
            self.rings = sorted(list(all_rings), key=lambda r: int(r[1:]))
        except (ValueError, IndexError):
            adj = defaultdict(list)
            in_degree = defaultdict(int)
            smaller_than_regex = re.compile(r"^smallerthan\s+\(\s*(\S+)\s*,\s*(\S+)\s*\)$")
            for fact in self.model.facts:
                match = smaller_than_regex.match(fact.name.replace(" ", ""))
                if match:
                    smaller, larger = match.groups()
                    if smaller in all_rings and larger in all_rings:
                        adj[smaller].append(larger)
                        in_degree[larger] += 1

            queue = sorted([r for r in all_rings if in_degree[r] == 0])
            sorted_rings = []
            while queue:
                u = queue.pop(0)
                sorted_rings.append(u)
                for v in sorted(adj[u]):
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
                        queue.sort()
            self.rings = sorted_rings

        for i, ring in enumerate(self.rings):
            self.ring_order[ring] = i

    def _get_current_config(self, state_bitwise):
        """
        Reconstructs the current `on` relationships from the bitwise state.
        """
        on_map = {}
        on_regex = re.compile(r"^on\s+\(\s*(\S+)\s*,\s*(\S+)\s*\)$")
        current_facts = self.model.state_explicit_repr(state_bitwise)

        for fact_str in current_facts:
            match = on_regex.match(fact_str.replace(" ", ""))
            if match:
                top_obj, bottom_obj = match.groups()
                if top_obj in self.rings:
                    on_map[top_obj] = bottom_obj
        return on_map

    def __call__(self, parent_node, node):
        """
        Calculates the heuristic value for a given search node.
        """
        if node.state is None:
            return 0

        current_on_map = self._get_current_config(node.state)
        h_val = 0

        for i in range(len(self.rings) - 1, -1, -1):
            ring = self.rings[i]
            goal_base = self.goal_on_map.get(ring)
            current_base = current_on_map.get(ring)

            if goal_base != current_base:
                # This is the largest unsettled ring. The size of the subproblem
                # is determined by this ring's size (i+1).
                # The minimum cost to solve this subproblem is 2^(i+1) - 1.
                k = i + 1
                h_val = (2 ** k) - 1
                break
        
        super().update_info(h_val)
        return h_val

    def __repr__(self):
        return f"TowersGemini25ProHeuristic25(rings={len(self.rings)})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 25 Info:\n"
            f"\tName: {self.name}\n"
            f"\tRings processed: {len(self.rings)}\n"
            f"\tGoal configuration loaded for {len(self.goal_on_map)} rings.\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s"
        )
