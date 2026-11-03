import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model


class TowersGemini25ProHeuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.

    This heuristic leverages the well-known optimal solution cost for the
    Towers of Hanoi problem, which is 2^k - 1 moves for k disks.
    It functions as a perfect heuristic (h=h*) for the subproblems defined
    by the HTN tasks.

    - Preprocessing: During initialization, it analyzes the `smallerThan` facts
      from the model to determine the size ordering of all rings.
    - Heuristic Calculation: For any given planning node, it inspects the
      abstract task being decomposed.
      - If the task is `(selectDirection ?r ...)` or a related higher-level task,
        it identifies the size 'k' of the ring `?r`.
      - It then calculates the heuristic value as the optimal number of moves
        required to move a stack of k rings, which is 2^k - 1.
    This provides a highly accurate, admissible estimate of the remaining cost,
    guiding the planner directly towards the optimal recursive solution and
    minimizing search effort.
    """

    def __init__(self, towers_gemini25pro_heuristic_15=None):
        super().__init__(name="towers_gemini25pro_heuristic_15")
        self.rings_by_size = []
        self.preprocessing_time = 0
        self.model = None

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic by preprocessing the model to learn the domain structure.
        """
        start_time = time.time()
        self.model = model

        # Preprocessing: Determine the size order of rings from `smallerThan` facts.
        rings = set()
        smaller_than_relations = []

        for fact in self.model.facts:
            parts = fact.name.strip('()').split()
            if len(parts) > 0:
                # Find all potential ring objects from relevant predicates
                if parts[0].lower() in ['smallerthan', 'on', 'goal_on', 'towertop']:
                    for p in parts[1:]:
                        if p.startswith('r'): # Assume rings are named 'r...'
                            rings.add(p)
                # Store relationships for sorting
                if parts[0].lower() == 'smallerthan' and parts[1].startswith('r'):
                    smaller_than_relations.append((parts[1], parts[2]))

        # A ring's size is inversely proportional to how many other rings it is smaller than.
        # The smallest ring is smaller than N-1 rings; the largest is smaller than 0.
        smaller_than_counts = {r: 0 for r in rings}
        for r_small, r_large in smaller_than_relations:
            if r_small in rings and r_large in rings:
                smaller_than_counts[r_small] += 1

        sorted_rings = sorted(smaller_than_counts.items(), key=lambda item: item[1], reverse=True)
        self.rings_by_size = [ring for ring, count in sorted_rings]

        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _get_ring_size_k(self, ring_name):
        """Helper function to get the 1-based size index of a ring."""
        try:
            return self.rings_by_size.index(ring_name) + 1
        except (ValueError, IndexError):
            # Fallback for single-ring problems or if a ring isn't in the list
            return 1

    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a node based on its task.
        The value is the exact number of primitive actions (moves) required
        to solve the subproblem represented by the task.
        """
        if not hasattr(node, 'task') or not node.task:
            return 0

        task_str = str(node.task).strip('()')
        parts = task_str.split()
        if not parts:
            return 0
            
        task_name = parts[0]
        params = parts[1:]

        h_value = 0
        if task_name == 'shiftTower':
            # Main task to move all N rings.
            k = len(self.rings_by_size)
            h_value = (2**k) - 1
        elif task_name == 'selectDirection':
            # Subtask to move a specific ring 'r' and the stack above it.
            ring_name = params[0]
            k = self._get_ring_size_k(ring_name)
            h_value = (2**k) - 1
        elif task_name == 'rotateTower':
            # This task moves a stack of size k-1 after moving disk k.
            # We need the parent task context to find 'k'.
            if node.parent and hasattr(node.parent, 'task') and node.parent.task:
                parent_task_str = str(node.parent.task).strip('()')
                parent_parts = parent_task_str.split()
                if parent_parts and parent_parts[0] == 'selectDirection':
                    ring_name = parent_parts[1]
                    k = self._get_ring_size_k(ring_name)
                    # Cost is moving disk k (1) + moving k-1 stack (2**(k-1) - 1)
                    h_value = 2**(k-1)
                else:
                    h_value = 1  # Admissible fallback
            else:
                h_value = 1 # Admissible fallback
        elif task_name in ['exchange', 'move_abstract']:
            # These are lower-level tasks. A simple cost of 1 is an admissible
            # and reasonable estimate for the minimum effort remaining.
            h_value = 1

        return h_value

    def __call__(self, parent_node, node):
        """
        Calculate the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic(rings={len(self.rings_by_size)})"

    def __str__(self):
        return f"TowersGemini25ProHeuristic(rings={len(self.rings_by_size)})"

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 15 Info:\n"
            f"\tName: {self.name}\n"
            f"\tNumber of rings identified: {len(self.rings_by_size)}\n"
            f"\tRing order (smallest to largest): {self.rings_by_size}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
