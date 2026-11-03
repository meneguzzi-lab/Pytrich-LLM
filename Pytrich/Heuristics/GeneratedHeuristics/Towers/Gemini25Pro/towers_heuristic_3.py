import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Towers of Hanoi domain.

    Rationale:
    The Towers of Hanoi problem has a well-defined, optimal recursive structure.
    The total number of moves to solve a problem with N disks is 2^N - 1. The key
    challenge for the planner is to discover and follow this recursive solution path
    efficiently.

    This heuristic does not evaluate the state itself, as the provided framework seems
    to favor heuristics that guide the selection of decomposition methods. Instead, it
    assigns a cost to each method based on its role in the recursive solution.
    The heuristic value estimates the number of primitive 'move' actions that are
    generated as a direct consequence of choosing a particular decomposition method.

    Design Principles:
    1.  **Action-based Cost**: The fundamental unit of cost is a single 'move' action.
        Methods that directly lead to a 'move' are assigned a heuristic cost of 1.0.
    2.  **Base Case Recognition**: Recursive base cases represent the termination of a
        sub-problem. The method 'exchangeClear' is the primary base case, where a
        tower is clear and no moves are needed. This method is assigned a cost of 0.0,
        making it highly desirable when its preconditions are met.
    3.  **Recursive Step Cost**: The main recursive methods ('m-rotateTower', 'exchangeLR',
        'exchangeRL') each contain one subtask ('move_abstract') that decomposes into a
        single 'move' action, plus another recursive subtask. The immediate cost of
        choosing such a method is one action, so they are assigned a cost of 1.0.
    4.  **Overhead Penalty**: Some methods ('m-shiftTower', 'selectedDirection') exist to
        navigate the task hierarchy and set up the main recursive calls. They do not
        produce 'move' actions themselves. They are given a small cost (0.1) to represent
        planning overhead. This ensures that, all else being equal, the planner prefers
        methods that directly contribute to the solution (i.e., move rings).

    This approach is computationally very efficient as it relies on a pre-computed lookup
    table based on method names. It directly leverages the domain structure encoded in
    the HTN methods to guide the search towards the known optimal decomposition strategy.
    """
    def __init__(self, towers_gemini25pro_heuristic_3=None):
        super().__init__(name="towers_gemini25pro_heuristic_3")
        self.method_scores = {}
        self.preprocessing_time = 0
        self.params = towers_gemini25pro_heuristic_3

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing method scores.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_scores()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_scores(self):
        """
        Assigns a pre-calculated cost to each method based on its function
        within the Towers of Hanoi recursive algorithm.
        """
        self.method_scores = {
            # Base case: A tower is clear, no moves needed. Most preferred.
            'exchangeClear': 0.0,

            # Overhead methods: Navigate the hierarchy but don't perform moves.
            # Small cost to prefer action-generating methods.
            'm-shiftTower': 0.1,
            'selectedDirection': 0.1,
            'm-selectDirection': 0.1,

            # Core recursive methods: Each generates one 'move' and one recursive call.
            # The immediate cost is one action.
            'm-rotateTower': 1.0,
            'exchangeLR': 1.0,
            'exchangeRL': 1.0,

            # Action bridge: Directly decomposes into the primitive 'move' action.
            'newMethod21': 1.0
        }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node by looking up the score
        of the method used for its decomposition.
        """
        if not hasattr(node, 'method') or not node.method:
            # Root node or un-decomposed task node has no associated method yet.
            return 0.0

        method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)

        # Default to 1.0 for any unknown methods, a conservative estimate of one action.
        return self.method_scores.get(method_name, 1.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic3()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro Heuristic 3 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Scored: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
