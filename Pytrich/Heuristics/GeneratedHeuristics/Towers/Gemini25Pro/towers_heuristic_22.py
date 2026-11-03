import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class TowersGemini25ProHeuristic22(Heuristic):
    """
    A domain-dependent heuristic for the Towers of Hanoi (towers) domain.

    This heuristic guides the HTN search by assigning costs to different decomposition
    methods based on their role in the recursive solution of the Towers of Hanoi problem.
    The core idea is to penalize methods that lead to deeper recursion, thereby encouraging
    the planner to pursue decompositions that represent base cases or lead to the completion
    of subproblems.

    The heuristic operates by statically analyzing the domain's methods during an
    initialization phase. It classifies each method into categories based on its function:
    - Base Case: Methods that terminate a recursive chain (e.g., 'exchangeClear').
    - Primitive Wrapper: Methods that directly call a primitive action (e.g., moving a disk).
    - Terminal Recursion: Methods that are the last step of one recursive process but initiate
      another complex one.
    - Simple Recursion: Methods that make a straightforward recursive call.
    - Complex Recursion: Methods that form the core of the exponential complexity, typically
      involving multiple recursive sub-problems.

    Higher penalties are assigned to more complex recursive methods, guiding the search
    towards simpler and terminal decompositions first.
    """

    def __init__(self, towers_gemini25pro_heuristic_22=None, name="towers_gemini25pro_heuristic_22"):
        """
        Initializes the Towers of Hanoi heuristic.

        Args:
            towers_gemini25pro_heuristic_22 (dict, optional): A dictionary of penalties for different
                                                          method categories. If None, default
                                                          penalties are used.
            name (str, optional): The name of the heuristic instance.
        """
        super().__init__(name=name)
        if towers_gemini25pro_heuristic_22 is None:
            self.penalties = {
                "base_case": 0.0,
                "primitive_wrapper": 0.1,
                "terminal_recursion": 0.3,
                "simple_recursion": 0.6,
                "complex_recursion": 1.0,
            }
        else:
            self.penalties = towers_gemini25pro_heuristic_22

        self.method_info = {}
        self.preprocessing_time = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-analyzing the domain methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _get_methods_from_model(self):
        """Robustly retrieves the list of decomposition methods from the model."""
        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'tasks'):
            for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)
        return methods

    def _precompute_method_info(self):
        """
        Statically analyzes all methods in the domain and classifies them based on their
        role in the Towers of Hanoi algorithm.
        """
        if not hasattr(self, 'model'):
            return

        methods = self._get_methods_from_model()
        for method in methods:
            method_name = method.name
            category = self._classify_method(method_name)
            self.method_info[method_name] = {'category': category}

    def _classify_method(self, method_name: str) -> str:
        """
        Classifies a method into a predefined category based on its name.

        Args:
            method_name (str): The name of the method to classify.

        Returns:
            str: The category of the method.
        """
        name_lower = method_name.lower()

        if 'exchangeclear' in name_lower:
            return "base_case"
        if 'newmethod21' in name_lower:  # Wraps the primitive 'move' action
            return "primitive_wrapper"
        if 'selecteddirection' in name_lower and 'm-selectdirection' not in name_lower:
            # This is the base case for the selectDirection recursion, but it leads to rotateTower
            return "terminal_recursion"
        if 'm-selectdirection' in name_lower:
            # This is a simple recursive step (peeling off one disk)
            return "simple_recursion"
        if 'm-rotatetower' in name_lower or 'exchange' in name_lower or 'shifttower' in name_lower:
            # These are the core, complex recursive methods
            return "complex_recursion"

        # Default fallback for any unrecognized method
        return "complex_recursion"

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """Calculates the heuristic value for a given search node."""
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        return self._score_method(node.method)

    def _score_method(self, method) -> float:
        """
        Scores a method based on its pre-calculated classification.
        Methods that represent recursion are penalized more heavily.
        """
        method_name = method.name
        info = self.method_info.get(method_name, {'category': 'complex_recursion'})
        category = info['category']
        return self.penalties.get(category, 1.0)  # Default to max penalty

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Computes the heuristic value for the current node transition.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"TowersGemini25ProHeuristic22(name={self.name})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Towers Gemini 2.5 Pro of Hanoi Heuristic 22 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_info)}\n"
            f"\tPenalties: {self.penalties}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
        )
