import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain dependant heuristic for towers domain
class TowersGemini20FlashHeuristic2(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_2="TowersGemini20FlashHeuristic2", misplaced_penalty=1, blocked_penalty=0.5, method_penalty=0.1):
        super().__init__(name=towers_gemini20flash_heuristic_2)
        self.misplaced_penalty = misplaced_penalty
        self.blocked_penalty = blocked_penalty
        self.method_penalty = method_penalty
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and precompute method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precompute information about methods for faster scoring.
        """
        if not self.model:
            return

        # Try different possible attributes for methods
        methods = []

        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
            # Get methods from abstract tasks
            for task in self.model.abstract_tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count the number of subtasks
            subtask_count = 0
            if hasattr(method, 'subtasks'):
                subtask_count = len(method.subtasks)
            elif hasattr(method, 'task_network'):
                subtask_count = len(method.task_network)

            self.method_scores[method_name] = {
                'subtask_count': subtask_count,
                'is_move_method': 'move' in method_name.lower(),
                'is_exchange_method': 'exchange' in method_name.lower(),
                'is_rotate_method': 'rotate' in method_name.lower(),
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a node based on its decomposition method.
        """
        state = node.state
        goal_state = self.model.goals  # Assuming Model.goals is a set of goal facts

        # Count misplaced rings: rings that are not on the goal tower.
        misplaced_rings = 0
        for fact in self.model.facts:
            if fact.name.startswith('on') and fact in state:
                ring, obj = fact.name.split()[1], fact.name.split()[2]
                if ring != 't3' and ring != 't2' and ring != 't1' : # only checking the ring
                    if f'(goal_on {ring} t3)' in [g.name for g in goal_state]:
                        misplaced_rings += 1


        # Count blocked rings: rings that have a larger ring on top of them
        blocked_rings = 0
        # (Implementation for counting blocked rings would require more complex
        # state analysis, potentially involving parsing the 'on' predicates and
        # 'smallerThan' predicates.  This is left as a placeholder.)
        # Iterate through the rings to assess if any ring is blocked by other rings

        # Penalize methods with a cost based on subtask count.
        method_penalty = 0
        if hasattr(node, 'method') and node.method:
            method_name = node.method.name if hasattr(node.method, 'name') else str(node.method)
            if method_name in self.method_scores:
                method_penalty = self.method_scores[method_name]['subtask_count'] * self.method_penalty

        heuristic_value = (
            misplaced_rings * self.misplaced_penalty +
            blocked_rings * self.blocked_penalty +
            method_penalty
        )

        return heuristic_value

    def __call__(self, parent_node, node):
        """
        Calculate heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic2(misplaced={self.misplaced_penalty}, blocked={self.blocked_penalty}, method={self.method_penalty})"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic2(misplaced={self.misplaced_penalty}, blocked={self.blocked_penalty}, method={self.method_penalty})"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 2 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tMisplaced penalty: {self.misplaced_penalty}\n"
            f"\tBlocked penalty: {self.blocked_penalty}\n"
            f"\tMethod penalty: {self.method_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
