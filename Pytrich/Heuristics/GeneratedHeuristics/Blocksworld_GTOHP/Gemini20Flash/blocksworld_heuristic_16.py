import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for Blocksworld domain
class BlocksworldGemini20FlashHeuristic16(Heuristic):
    def __init__(self, blocksworld_gemini20flash_heuristic_16=1, name="blocksworld_gemini20flash_heuristic_16"):
        super().__init__(name=name)
        self.name = name
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        # self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        This heuristic estimates the number of 'put_on' tasks remaining in the plan,
        based on the current task network.
        """
        if not hasattr(node, 'task_network') or not node.task_network:
            return 0

        # Count the number of 'do_put_on' tasks in the task network.
        put_on_tasks_count = 0
        for task in node.task_network:
            if hasattr(task, 'name') and 'do_put_on' in task.name:
                put_on_tasks_count += 1
            elif hasattr(task, 'task') and hasattr(task.task, 'name') and 'do_put_on' in task.task.name:
                put_on_tasks_count += 1

        return put_on_tasks_count

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BlocksworldGemini20FlashHeuristic16(name={self.name})"

    def __str__(self):
        return f"BlocksworldGemini20FlashHeuristic16(name={self.name})"

    def __output__(self):
        return (
            f"Blocksworld Gemini 2.0 Flash Heuristic info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
