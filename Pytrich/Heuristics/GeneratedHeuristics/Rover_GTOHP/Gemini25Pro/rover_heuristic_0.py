import time
from Pytrich.Heuristics.heuristic import Heuristic

class RoverGemini25ProHeuristic0(Heuristic):
    """
    A domain-dependent heuristic for the HDDL ROVER domain.

    This heuristic evaluates the promise of a decomposition method by assigning a cost
    based on the types of subtasks it generates. The core idea is that navigation is
    the most expensive operation, followed by data collection and other rover actions.
    By penalizing methods that lead to more complex and lengthy action sequences,
    the heuristic guides the planner towards more efficient solutions.

    The heuristic pre-computes a score for each method in the domain during
    initialization. This score is a weighted sum of the costs associated with the
    method's subtasks. This makes the heuristic evaluation during the search very fast.
    """
    def __init__(self, rover_gemini25pro_heuristic_0="rover_gemini25pro_heuristic_0"):
        super().__init__(name="rover_gemini25pro_heuristic_0")
        
        # Penalties are kept small to act as admissible tie-breakers and guides
        self.penalties = {
            'navigation': 0.5,        # For do_navigate*, navigate
            'data_collection': 0.3,   # For sample_*, take_image
            'calibration': 0.2,       # For do_calibrate, calibrate
            'communication': 0.15,    # For send_*, communicate_*
            'resource_management': 0.1, # For empty_store, drop
            'nop': 0.0                # For base cases, no cost
        }
        
        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain and assigns a heuristic score
        to each based on its subtasks. This is done once at the beginning of the search.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            method_name = method.name
            score = 0.0

            if not hasattr(method, 'task_network'):
                self.method_scores[method_name] = score
                continue

            for subtask in method.task_network:
                subtask_name = subtask.name.lower()

                if any(nav_term in subtask_name for nav_term in ['navigate', 'do_navigate1', 'do_navigate2']):
                    score += self.penalties['navigation']
                elif any(dc_term in subtask_name for dc_term in ['sample_soil', 'sample_rock', 'take_image']):
                    score += self.penalties['data_collection']
                elif any(cal_term in subtask_name for cal_term in ['calibrate', 'do_calibrate']):
                    score += self.penalties['calibration']
                elif any(comm_term in subtask_name for comm_term in ['send_data', 'communicate']):
                    score += self.penalties['communication']
                elif any(res_term in subtask_name for res_term in ['empty_store', 'drop']):
                    score += self.penalties['resource_management']
                elif 'nop' in subtask_name:
                    score += self.penalties['nop']
            
            self.method_scores[method_name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node.
        This is a fast lookup of the pre-computed score for the method that
        was used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        # Return the pre-computed score, with a default of 1.0 if not found
        return self.method_scores.get(node.method.name, 1.0)

    def __call__(self, parent_node, node):
        """
        The main entry point for the heuristic evaluation during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RoverGemini25ProHeuristic0(penalties={self.penalties})"

    def __str__(self):
        return self.name

    def __output__(self):
        return (
            f"Heuristic '{self.name}' Info:\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f}s\n"
            f"\tPenalties: {self.penalties}"
        )
