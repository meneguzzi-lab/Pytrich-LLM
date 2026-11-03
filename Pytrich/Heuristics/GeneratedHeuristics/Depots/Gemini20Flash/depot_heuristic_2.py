import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for the Depot domain.
class DepotGemini20FlashHeuristic2(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_2=1, name="DepotGemini20FlashHeuristic2"):
        super().__init__(name=name)
        self.name = name
        self.task_penalty = 1  # Initial penalty for each open task
        self.distance_weight = 1 #Weight to calculate remaining distance
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute data.
        """
        start_time = time.time()
        self.model = model
        #self._precompute_distances()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on open tasks.
        """
        if node.is_primitive():
            return 0
        open_tasks = node.open_tasks
        #if not hasattr(node, 'open_tasks') or not node.open_tasks:
        #    return 0

        #h_value = len(node.open_tasks) * self.task_penalty

        #h_value = self._estimate_remaining_distance(node) * self.distance_weight

        h_value = len(open_tasks) * self.task_penalty + self._estimate_remaining_distance(node) * self.distance_weight
        return h_value

    def _estimate_remaining_distance(self, node):
        """
        Estimate the remaining distance to the goal based on open tasks.
        #This is a place holder function, in the future a state evalutor should be implemented.
        """
        #Placeholder function
        return 1


    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node during search.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic2(task_penalty={self.task_penalty})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic2(task_penalty={self.task_penalty})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 2 info:\n"
            f"\tName: {self.name}\n"
            f"\tTask penalty: {self.task_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
