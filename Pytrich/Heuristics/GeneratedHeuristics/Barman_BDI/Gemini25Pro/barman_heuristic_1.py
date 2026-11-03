import time
from Pytrich.Heuristics.heuristic import Heuristic

class BarmanGemini25ProHeuristic1(Heuristic):
    """
    A domain-dependent heuristic for the Barman HDDL domain.

    This heuristic estimates the cost to reach a goal state by analyzing the
    decomposition methods chosen by the planner. It pre-computes a "penalty"
    score for each method in the domain based on the complexity and resource
    contention of the subtasks it generates.

    Rationale:
    1.  **Domain Analysis**: The key challenges in the Barman domain are resource
        contention for the single shaker and the two hands, and the procedural
        overhead of cleaning containers (shots and the shaker) between uses.
    2.  **Heuristic Strategy**: The heuristic assigns a small cost (penalty) to
        methods based on the high-impact actions their subtasks will eventually
        require. Operations like cleaning the shaker or shaking a cocktail are
        more costly than simple manipulations like grasping an object.
    3.  **Admissibility**: By assigning very small, non-negative costs, the heuristic
        aims to be admissible (i.e., it never overestimates the true cost).
        The cost for methods that achieve an already-satisfied goal ('Null' methods)
        is correctly set to zero.
    4.  **Efficiency**: The scores for all domain methods are pre-calculated during
        the `initialize` phase. During search, the heuristic value for a node is
        a fast dictionary lookup, imposing minimal overhead.
    5.  **Guidance**: The heuristic guides the search to prefer simpler decompositions
        and to recognize already-completed sub-goals, which helps prune the search space.
        For example, it penalizes methods that require cleaning the shaker more heavily
        than those that only require cleaning a shot glass, reflecting the shaker's
        role as a central, shared resource.
    """
    def __init__(self, barman_gemini25pro_heuristic_1=None):
        """
        Initializes the heuristic with configurable penalties for key actions.

        :param barman_gemini25pro_heuristic_1: A dictionary with custom penalty values.
        """
        super().__init__(name="barman_gemini25pro_heuristic_1")
        config = barman_gemini25pro_heuristic_1 if barman_gemini25pro_heuristic_1 is not None else {}

        # Define penalties for key operations. These values are chosen to reflect
        # the relative cost and bottleneck potential of each action.
        self.penalties = {
            'clean_shaker': config.get('clean_shaker', 0.4),
            'shake': config.get('shake', 0.3),
            'add_ingredient': config.get('add_ingredient', 0.2),
            'clean_shot': config.get('clean_shot', 0.1),
            'pour': config.get('pour', 0.1),
            'manipulation': config.get('manipulation', 0.05)  # e.g., grasp, drop
        }

        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-computing scores for all methods.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all decomposition methods in the domain and calculates a
        heuristic score for each, storing it for fast lookup during search.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions
        for method in methods:
            method_name = str(method.name)
            score = self._score_method(method)
            self.method_scores[method_name] = score

    def _score_method(self, method):
        """
        Scores a single method based on the complexity of its subtasks.
        A lower score indicates a more preferred (less costly) method.
        """
        method_name_lower = str(method.name).lower()

        # 'Null' methods correspond to already-achieved goals, so their cost is zero.
        if 'null' in method_name_lower:
            return 0.0

        score = 0.0

        subtasks = []
        if hasattr(method, 'task_network'):
            subtasks = method.task_network

        if not subtasks:
            # A non-null method that decomposes to no tasks is also zero-cost.
            return 0.0

        for subtask in subtasks:
            # Extract subtask name robustly
            subtask_name = ""
            if hasattr(subtask, 'name'):
                subtask_name = str(subtask.name)
            elif hasattr(subtask, 'task'):
                subtask_name = str(subtask.task)
            else:
                subtask_name = str(subtask)

            subtask_lower = subtask_name.lower().replace("-", "")

            # Add penalties based on keywords in subtask names
            if 'achievecleanshaker' in subtask_lower or 'cleanshaker' in subtask_lower:
                score += self.penalties['clean_shaker']
            elif 'shake' in subtask_lower:
                score += self.penalties['shake']
            elif 'achievecontainsshakeringredient' in subtask_lower or 'fillshot' in subtask_lower:
                score += self.penalties['add_ingredient']
            elif 'achievecleanshot' in subtask_lower or 'cleanshot' in subtask_lower:
                score += self.penalties['clean_shot']
            elif 'pour' in subtask_lower:
                score += self.penalties['pour']
            elif any(m in subtask_lower for m in ['achieveholding', 'grasp', 'drop', 'achievehandempty']):
                score += self.penalties['manipulation']

        return score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a given search node by looking up the
        pre-computed score of the method used for decomposition.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = str(node.method.name)
        return self.method_scores.get(method_name, 0.0)

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic1(penalties={self.penalties})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 1 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
            f"\tPenalties: {self.penalties}\n"
        )
