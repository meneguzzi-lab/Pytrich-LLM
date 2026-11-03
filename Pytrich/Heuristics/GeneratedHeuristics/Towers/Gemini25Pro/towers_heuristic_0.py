import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic0(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic evaluates the promise of a decomposition method based on its role
    in the classic recursive solution to the Towers of Hanoi problem. It assigns a base cost
    to each method type, prioritizing methods that terminate recursion (base cases) or
    lead directly to primitive actions. A lower heuristic value indicates a more promising path.

    The heuristic is designed to be admissible by keeping the cost values small and representative
    of planning effort rather than a direct estimate of the number of moves.

    Design Rationale:
    1.  **Hierarchical Role Analysis**: The domain's methods follow a strict recursive pattern.
        The heuristic categorizes methods into roles: top-level task, compound recursive step,
        base case, and action wrapper. Methods closer to the primitive 'move' action receive
        lower scores.
    2.  **Object-based Penalty**: The core challenge in Towers of Hanoi is moving the largest rings.
        Any method that manipulates a larger ring is part of a more complex subproblem. The
        heuristic adds a small, proportional penalty based on the size of the largest ring in the
        method's parameters. This helps guide the search to solve for smaller subproblems first,
        which is the correct strategy (e.g., move small disks out of the way to access a large one).
    3.  **Admissibility**: The heuristic values are not an estimate of the true cost (which would be 2^n-1)
        but rather small, consistent costs that guide the search. By keeping the values low (typically < 1.0),
        it helps A* prefer better paths without becoming inadmissible and failing to find optimal solutions.
    4.  **Efficiency**: The heuristic pre-calculates the base score for each method type during an
        initialization phase, making the per-node calculation during search very fast.
    """
    def __init__(self, towers_gemini25pro_heuristic_0=0.5, ring_penalty=0.01):
        """
        Initializes the heuristic.

        :param towers_gemini25pro_heuristic_0: A factor to scale the base cost of methods.
        :param ring_penalty: A factor to scale the penalty associated with ring size.
        """
        super().__init__(name="TowersGemini25ProHeuristic0")
        self.base_cost_factor = towers_gemini25pro_heuristic_0
        self.ring_penalty = ring_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing information about the domain methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _get_ring_size(self, ring_name: str) -> int:
        """Extracts the numerical size from a ring's name (e.g., 'r10' -> 10)."""
        if ring_name.startswith('r'):
            try:
                return int(ring_name[1:])
            except (ValueError, IndexError):
                pass
        return 0  # Default for non-ring objects or parse failure

    def _precompute_method_info(self):
        """Analyzes all methods in the model and assigns a base score based on their role."""
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name_key = method.name.split()[0]
            if method_name_key in self.method_scores:
                continue

            score = 1.0  # Default cost

            # Lower scores are better. Prioritize methods closer to actions or base cases.
            if 'newMethod21' in method_name_key:  # Directly wraps the 'move' action
                score = 0.1
            elif 'exchangeClear' in method_name_key:  # Base case for 'exchange' recursion
                score = 0.2
            elif method_name_key == 'selectedDirection':  # Base case for 'selectDirection' recursion
                score = 0.3
            elif 'rotateTower' in method_name_key:  # Intermediate compound task
                score = 0.5
            elif 'selectDirection' in method_name_key:  # Recursive step for selection
                score = 0.6
            elif 'exchange' in method_name_key:  # Recursive step for exchange
                score = 0.7
            elif 'shiftTower' in method_name_key:  # Top-level, most abstract task
                score = 0.8

            self.method_scores[method_name_key] = {'base_score': score}

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """Calculates the heuristic value for a given search node."""
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        method_name_key = method_name.split()[0]

        score_info = self.method_scores.get(method_name_key, {'base_score': 1.0})
        base_score = score_info['base_score'] * self.base_cost_factor

        # Add a penalty based on the size of the largest ring being manipulated.
        # This encourages solving for smaller disks first.
        max_ring_size = 0
        try:
            params = method_name.split()[1:]
            for param in params:
                size = self._get_ring_size(param)
                if size > max_ring_size:
                    max_ring_size = size
        except IndexError:
            pass  # Method has no parameters

        ring_cost = max_ring_size * self.ring_penalty
        
        return base_score + ring_cost

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """Makes the heuristic callable by the planner's search algorithm."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic0(base_factor={self.base_cost_factor}, ring_penalty={self.ring_penalty})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic0 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tBase Cost Factor: {self.base_cost_factor}\n"
            f"\tRing Penalty: {self.ring_penalty}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
