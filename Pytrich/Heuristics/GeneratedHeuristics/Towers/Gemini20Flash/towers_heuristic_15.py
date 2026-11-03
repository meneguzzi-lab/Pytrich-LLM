import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Towers of Hanoi domain.
# This heuristic estimates the cost of moving rings to their goal positions, 
# considering the hierarchical nature of the problem.
class TowersGemini20FlashHeuristic15(Heuristic):
    def __init__(self, towers_gemini20flash_heuristic_15=1, name="towers_gemini20flash_heuristic_15"):
        super().__init__(name=name)
        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the problem model.
        Precomputes information about methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Precomputes information about methods for efficient heuristic calculation.
        This involves analyzing the methods to identify their relevance to 
        different aspects of the problem, such as moving rings or selecting directions.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'abstract_tasks'):
             for abstract_task in self.model.abstract_tasks:
                if hasattr(abstract_task, 'decompositions'):
                    methods.extend(abstract_task.decompositions)


        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)
            move_abstract_count = 0
            exchange_count = 0
            rotate_count = 0

            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'move_abstract' in subtask_lower:
                        move_abstract_count += 1
                    if 'exchange' in subtask_lower:
                        exchange_count += 1
                    if 'rotate' in subtask_lower:
                        rotate_count += 1


            self.method_scores[method_name] = {
                'move_abstract_count': move_abstract_count,
                'exchange_count': exchange_count,
                'rotate_count': rotate_count,
                'is_shift_tower': self._is_shift_tower_method(method_name),
                'is_select_direction': self._is_select_direction_method(method_name),
                'is_exchange': self._is_exchange_method(method_name)
            }


    def _is_shift_tower_method(self, method_name):
        return 'shiftTower' in method_name

    def _is_select_direction_method(self, method_name):
        return 'selectDirection' in method_name
    
    def _is_exchange_method(self, method_name):
        return 'exchange' in method_name

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node based on its decomposition method.
        This heuristic favors nodes that decompose into simpler tasks related to moving
        and exchanging rings.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Scores a method based on its relevance to the problem goals.
        Lower scores are assigned to methods that directly contribute to moving rings
        or setting up the conditions for those moves.
        """
        score = 0
        method_name = method.name if hasattr(method, 'name') else str(method)
        
        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            score += info['move_abstract_count'] * 0.5
            score += info['exchange_count'] * 0.3
            score += info['rotate_count'] * 0.2

            if info['is_shift_tower']:
                score += 1  # Encourage shift tower decomposition
            if info['is_select_direction']:
                score += 0.7  # Slightly encourage select direction
            if info['is_exchange']:
                score += 0.5  # Encourage exchange operations

        return score

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini20FlashHeuristic15()"

    def __str__(self):
        return f"TowersGemini20FlashHeuristic15()"

    def __output__(self):
        return (
            f"Towers Gemini 2.0 Flash Heuristic 15 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
        )
