import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Depot domain
class DepotGemini20FlashHeuristic32(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_32, name="DepotGemini20FlashHeuristic32"):
        super().__init__(name=name)
        self.method_scores = {}
        self.preprocessing_time = 0
        self.num_trucks = 0  # Number of trucks in the problem
        self.num_hoists = 0  # Number of hoists in the problem
        self.num_crates = 0  # Number of crates
        self.num_areas = 0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the model and precomputes method characteristics.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self._analyze_problem_structure()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _analyze_problem_structure(self):
        """
        Analyzes the problem to count the number of trucks, hoists and crates.
        This helps in estimating the complexity of the problem and weighting the
        heuristic accordingly.
        """
        self.num_trucks = 0
        self.num_hoists = 0
        self.num_crates = 0
        self.num_areas = 0

        for fact in self.model.facts:
            fact_name = fact.name.lower()
            if 'truck' in fact_name:
                self.num_trucks += 1
            elif 'hoist' in fact_name:
                self.num_hoists += 1
            elif 'crate' in fact_name:
                self.num_crates += 1
            elif 'area' in fact_name:
                self.num_areas += 1

    def _precompute_method_info(self):
        """
        Precomputes information about methods for faster scoring.  Counts key words in
        the method and subtask names to get some domain awareness.
        """
        if not self.model:
            return

        methods = []
        if hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, 'name') else str(method)

            # Count different types of subtasks
            move_count = 0
            load_count = 0
            unload_count = 0

            subtask_count = 0

            # Get subtasks from method
            subtasks = []
            if hasattr(method, 'subtasks'):
                subtasks = method.subtasks
            elif hasattr(method, 'task_network'):
                subtasks = method.task_network
            elif hasattr(method, 'decomposition'):
                subtasks = method.decomposition

            if subtasks:
                subtask_count = len(subtasks)
                for subtask in subtasks:
                    subtask_name = ""
                    if hasattr(subtask, 'name'):
                        subtask_name = subtask.name
                    elif hasattr(subtask, 'task'):
                        subtask_name = str(subtask.task)
                    else:
                        subtask_name = str(subtask)

                    subtask_lower = subtask_name.lower()

                    if 'move' in subtask_lower:
                        move_count += 1
                    if 'load' in subtask_lower:
                        load_count += 1
                    if 'unload' in subtask_lower:
                        unload_count += 1

            self.method_scores[method_name] = {
                'move_count': move_count,
                'load_count': load_count,
                'unload_count': unload_count,
                'subtask_count': subtask_count,
                'is_move_method': 'move' in method_name.lower(),
                'is_load_method': 'load' in method_name.lower(),
                'is_unload_method': 'unload' in method_name.lower(),
                'is_base_case': 'base' in method_name.lower(),
                'complexity_score': move_count + load_count + unload_count
            }

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        return self._score_method(node.method, node)

    def _score_method(self, method, node):
        """
        Scores a method based on precomputed information and problem structure.
        """
        score = 0

        method_name = method.name if hasattr(method, 'name') else str(method)
        method_lower = method_name.lower()

        if method_name in self.method_scores:
            info = self.method_scores[method_name]

            # Adjust score based on operation counts
            score += info['move_count'] * 0.2
            score += info['load_count'] * 0.3
            score += info['unload_count'] * 0.3

            # Give a small bonus for base cases
            if info['is_base_case']:
                score -= 0.1
        else:
            # if method not in precomputed scores penalize slightly
            score += 0.2

        # Adjust score based on whether the method contains
        # certain keywords
        if info['is_move_method']:
            score += 0.2

        return max(score, 0)

    def __call__(self, parent_node, node):
        """
        Calculates and returns the heuristic value for a given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic32()"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic32()"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 32 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tNumber of Trucks: {self.num_trucks}\n"
            f"\tNumber of Hoists: {self.num_hoists}\n"
            f"\tNumber of Crates: {self.num_crates}\n"
            f"\tNumber of Areas: {self.num_areas}\n"
        )
