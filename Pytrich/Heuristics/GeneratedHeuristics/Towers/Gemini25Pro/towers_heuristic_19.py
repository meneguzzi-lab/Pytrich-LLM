import time
from Pytrich.Heuristics.heuristic import Heuristic

class TowersGemini25ProHeuristic19(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi domain.
    
    This heuristic estimates the remaining cost to solve a subproblem based on the
    size of the stack of rings being manipulated. The core idea is that the complexity
    of any Towers of Hanoi subproblem is determined by the number of rings that need
    to be moved.

    Design Rationale:
    1.  **Preprocessing for Ring Sizes:** The heuristic first analyzes the initial state
        to determine a size ranking for all rings. It counts how many other rings are
        smaller than a given ring, establishing a 0-indexed size rank. This is done
        once during initialization for efficiency.
    
    2.  **Task-Based Estimation:** The heuristic value for a search node is derived from
        the task associated with that node. It identifies the rings involved in the
        task's parameters.

    3.  **Admissible Cost Estimation:** The cost is estimated as the number of rings in the
        stack being manipulated. For a task involving a top ring with size rank `k`,
        the stack size is `k+1`. The heuristic value is `k+1`. This is admissible because
        each of the `k+1` rings in that stack must be moved at least once to solve
        the subproblem.

    4.  **Hierarchical Awareness:** By associating cost with the tasks (`shiftTower`,
        `rotateTower`, `exchange`), the heuristic naturally respects the hierarchical
        problem structure. Larger, more complex tasks (like the initial `shiftTower`)
        will involve larger rings or more rings in total, resulting in higher
        heuristic values, guiding the search appropriately.

    5.  **Efficiency:** The core calculation is a simple lookup of precomputed ring sizes
        and a max operation, making the heuristic very fast to compute at each node.
    """

    def __init__(self, towers_gemini25pro_heuristic_19):
        # The parameter name is per instruction. It's treated as a config dict.
        config = towers_gemini25pro_heuristic_19 if isinstance(towers_gemini25pro_heuristic_19, dict) else {}
        self.move_cost = config.get('move_cost', 1)
        super().__init__(name="towers_gemini25pro_heuristic_19")
        self.ring_sizes = {}
        self.total_rings = 0
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing ring sizes from the model.
        """
        start_time = time.time()
        self.model = model
        self._precompute_ring_sizes(model)
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_ring_sizes(self, model):
        """
        Parses the initial state's 'smallerThan' facts to create a size ranking
        for every ring. The size of a ring is the number of other rings smaller than it.
        """
        initial_facts = model.state_explicit_repr(model.initial_state)
        
        smaller_than_facts = [f for f in initial_facts if f.startswith('(smallerthan ')]

        rings = set()
        for fact_str in smaller_than_facts:
            try:
                parts = fact_str.strip('()').split()
                if len(parts) == 3 and parts[0].lower() == 'smallerthan':
                    rings.add(parts[1])
                    rings.add(parts[2])
            except:
                continue

        # For each ring, its size is its 0-indexed rank (number of smaller rings).
        for r_major in rings:
            count = 0
            for r_minor in rings:
                if r_major == r_minor:
                    continue
                if f'(smallerthan {r_minor} {r_major})' in initial_facts:
                    count += 1
            self.ring_sizes[r_major] = count
        
        self.total_rings = len(rings)

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node based on its task.
        The heuristic value is the number of rings in the subproblem, which is an
        admissible estimate of the minimum number of moves required.
        """
        if not hasattr(node, 'task') or not node.task:
            return 0
        
        task_tuple = node.task
        task_name = task_tuple[0].lower()

        if 'move_abstract' in task_name:
            return self.move_cost

        rings_in_task = [p for p in task_tuple[1:] if p in self.ring_sizes]
        
        if not rings_in_task:
            # Handle tasks without ring parameters, like the initial 'shiftTower'.
            # The cost is the total number of rings to move.
            if 'shifttower' in task_name:
                return self.total_rings
            return 0

        # Find the size rank of the largest ring involved in the task.
        max_size_rank = -1
        for r in rings_in_task:
            if self.ring_sizes.get(r, -1) > max_size_rank:
                max_size_rank = self.ring_sizes[r]
        
        if max_size_rank == -1:
            return 0

        # The stack size is the rank + 1. This is our heuristic value.
        # This is admissible because each ring in the stack must move at least once.
        stack_size = max_size_rank + 1
        
        return stack_size
        
    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic19(move_cost={self.move_cost})"

    def __str__(self):
        return f"TowersGemini25ProHeuristic19(total_rings={self.total_rings})"

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 19 Info:\n"
            f"\tName: {self.name}\n"
            f"\tTotal Rings Detected: {self.total_rings}\n"
            f"\tMove Cost: {self.move_cost}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
            f"\tRing Sizes Precomputed: {len(self.ring_sizes)} rings"
        )
