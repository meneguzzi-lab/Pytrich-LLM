import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RoverGemini25ProHeuristic17(Heuristic):
    """
    A domain-dependent heuristic for the HDDL ROVER domain.

    This heuristic follows the pattern of providing a small, admissible cost estimate for
    each decomposition method available in the domain. The planner uses these scores to
    guide its search, prioritizing methods that are simpler or lead more directly to a solution.
    The scores are pre-calculated during initialization based on a manual analysis of the
    domain's hierarchical structure.

    Design Rationale:
    The primary challenge in the ROVER domain is managing the sequence of navigation, data
    collection (sometimes requiring calibration), and communication. This often involves
    long chains of actions and recursive pathfinding.

    The heuristic assigns scores to methods based on their complexity:
    - Score 0.0 (Best): Base-case methods that terminate a decomposition branch (e.g., a task is already achieved).
    - Score 0.1 (Good): Methods that decompose into a single primitive action (e.g., 'navigate', 'drop'). These directly advance the ground-level plan.
    - Score 0.2 (Moderate): Methods involving a mix of a primitive action and one abstract task (e.g., navigate then communicate).
    - Score 0.3 (Complex): The recursive pathfinding method, which can significantly expand the search tree.
    - Score 0.5-0.6 (Most Complex): High-level methods that decompose into a long sequence of abstract tasks, representing a full sub-goal (e.g., 'get_image_data' which requires calibration, navigation, imaging, and sending).

    By preferring lower-cost methods, the planner is guided to resolve tasks with fewer steps and less recursion where possible, leading to a more efficient search.
    """
    def __init__(self, rover_gemini25pro_heuristic_17=None):
        # The parameter is unused but included to match the required signature.
        super().__init__(name="rover_gemini25pro_heuristic_17")
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic, pre-calculating scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes each decomposition method from the domain and assigns a static complexity score.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            score = 0.2  # Default score for unclassified methods

            # Tier 1: Base cases (cost = 0.0)
            if method_name in ['m0_do_navigate1', 'm2_do_navigate2', 'm5_empty_store']:
                score = 0.0
            # Tier 2: Single primitive actions (cost = 0.1)
            elif method_name in ['m3_do_navigate2', 'm6_empty_store', 'm9_send_soil_data', 'm12_send_rock_data']:
                score = 0.1
            # Tier 3: Simple abstract decompositions (cost = 0.2)
            elif method_name in ['m1_do_navigate1', 'm8_send_soil_data', 'm11_send_rock_data', 
                                 'm14_send_image_data', 'm15_do_calibrate']:
                score = 0.2
            # Tier 4: Recursive navigation (cost = 0.3)
            elif method_name == 'm4_do_navigate2':
                score = 0.3
            # Tier 5: Complex goal achievement (cost = 0.5)
            elif method_name in ['m7_get_soil_data', 'm10_get_rock_data']:
                score = 0.5
            # Tier 6: Most complex goal (image data) (cost = 0.6)
            elif method_name == 'm13_get_image_data':
                score = 0.6
            
            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Retrieves the pre-calculated heuristic value for the method that generated this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        return self.method_scores.get(method_name, 0.5) # Default score for safety

    def __call__(self, parent_node, node):
        """
        The main heuristic function called by the planner.
        Returns a small, admissible cost associated with the decomposition choice.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
        
    def __repr__(self):
        return f"RoverGemini25ProHeuristic17()"

    def __str__(self):
        return f"RoverGemini25ProHeuristic17 - A heuristic based on pre-computed method complexity scores."

    def __output__(self):
        return (
            f"Rover Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
        )
