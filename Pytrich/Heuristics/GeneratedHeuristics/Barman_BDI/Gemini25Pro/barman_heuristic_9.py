import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Decomposition

class BarmanGemini25ProHeuristic9(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.

    This heuristic statically analyzes HTN methods (decompositions) to estimate the cost
    of applying them. The core idea is that some methods are more "productive" than
    others in achieving the primary goal of creating a cocktail. The heuristic assigns
    a score to each method during an initialization phase, making the per-node evaluation
    during search very fast.

    The scoring logic is as follows:
    - Base Score: The number of subtasks in a method's decomposition. This forms an
      admissible baseline, assuming each subtask costs at least 1 action to solve.
    - Null Methods: Methods that represent an already-achieved goal have a score of 0.
    - Penalties (Higher Cost):
        - Cleaning: Methods involving cleaning are penalized, as they are necessary
          overhead. Cleaning a full container is penalized more than an empty one.
        - Manipulation: Simple actions like picking up or dropping items are given a
          small penalty as they are supporting actions, not primary goal-achieving steps.
    - Bonuses (Lower Cost):
        - Productive Steps: Methods that directly contribute to the cocktail (adding
          ingredients, shaking, pouring the final product) are given a bonus to
          encourage the planner to prioritize them.

    This approach guides the search towards decompositions that make direct progress on
    the cocktail goals, while being mindful of the overhead costs associated with
    preparation and cleanup. By pre-computing scores, it remains computationally efficient.
    """
    def __init__(self, clean_full_penalty=2.0, clean_empty_penalty=1.0, ingredient_bonus=1.0, manipulation_penalty=0.5, name="barman_gemini25pro_heuristic_9"):
        super().__init__(name=name)
        self.clean_full_penalty = clean_full_penalty
        self.clean_empty_penalty = clean_empty_penalty
        self.ingredient_bonus = ingredient_bonus
        self.manipulation_penalty = manipulation_penalty
        
        self.method_scores = {}
        self.preprocessing_time = 0.0
        self.model = None

    def initialize(self, model: Model, initial_node: HTNNode):
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
        Iterates through all decompositions in the model and calculates a heuristic score for each,
        storing it in a dictionary for fast lookups during search.
        """
        if not self.model:
            return
            
        methods = []
        if hasattr(self.model, 'decompositions'):
            methods = self.model.decompositions
        elif hasattr(self.model, 'methods'):
            methods = self.model.methods
        elif hasattr(self.model, 'tasks'):
             for task in self.model.tasks:
                if hasattr(task, 'decompositions'):
                    methods.extend(task.decompositions)
        
        if not methods:
            return

        for method in methods:
            score = self._score_method(method)
            self.method_scores[method.name] = score

    def _score_method(self, method: Decomposition) -> float:
        """
        Calculates a heuristic score for a single method based on its name and subtasks.
        """
        method_name = method.name.lower()
        subtask_count = len(method.task_network)

        # Null methods represent an already satisfied goal, so their cost is 0.
        if "null" in method_name:
            return 0.0
        
        # The base score is the number of subtasks, an admissible estimate.
        score = float(subtask_count)
        
        # Penalize cleaning overhead. Cleaning a full container is more costly.
        if "cleanfull" in method_name:
            score += self.clean_full_penalty
        elif "cleanempty" in method_name:
            score += self.clean_empty_penalty
        
        # Reward productive steps that directly advance cocktail making.
        if "addingredientto" in method_name:
            score -= self.ingredient_bonus
        
        # Reward other core cocktail creation and serving steps.
        if "makecocktail" in method_name and "null" not in method_name:
             score -= self.ingredient_bonus
        
        if "dopourshakertoshot" in method_name:
             score -= self.ingredient_bonus
        
        # Penalize simple manipulation overhead.
        if "pickup" in method_name or "emptyhand" in method_name or "putdown" in method_name:
             score += self.manipulation_penalty

        # Heuristic cost must be non-negative.
        return max(score, 0.0)

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Retrieves the pre-calculated heuristic value for the method used to generate this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        # Default to subtask count if method not found, but it should have been pre-scored.
        return self.method_scores.get(method_name, float(len(node.method.task_network)))

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        Calculates the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value
    
    def __repr__(self) -> str:
        return f"BarmanGemini25ProHeuristic9(clean_full={self.clean_full_penalty}, clean_empty={self.clean_empty_penalty}, ingredient_bonus={self.ingredient_bonus})"

    def __str__(self) -> str:
        return self.__repr__()

    def __output__(self) -> str:
        """
        Returns a formatted string with information about the heuristic's configuration and performance.
        """
        return (
            f"Barman Gemini 2.5 Pro Heuristic 9 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tClean Full Penalty: {self.clean_full_penalty}\n"
            f"\tClean Empty Penalty: {self.clean_empty_penalty}\n"
            f"\tIngredient Bonus: {self.ingredient_bonus}\n"
            f"\tManipulation Penalty: {self.manipulation_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
