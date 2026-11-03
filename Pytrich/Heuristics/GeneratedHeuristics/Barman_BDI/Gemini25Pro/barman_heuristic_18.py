import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class BarmanGemini25ProHeuristic18(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Barman domain.
    This heuristic estimates the cost of achieving goals by assigning penalties
    to methods based on the complexity and resource-intensiveness of the
    operations they represent in the cocktail-making process.

    The heuristic values are precomputed for each method in the domain during
    initialization to ensure efficient evaluation during search. The penalties
    are designed to guide the planner towards more efficient sequences of actions,
    for instance by correctly weighing the cost of cleaning the shaker, which is a
    significant bottleneck in problems with many cocktails and few shakers.
    """
    def __init__(self, barman_gemini25pro_heuristic_18=None):
        """
        Initializes the heuristic with customizable weights for different operations.
        :param barman_gemini25pro_heuristic_18: A dictionary of parameters to tune the heuristic's behavior.
        """
        super().__init__(name="barman_gemini25pro_heuristic_18")
        if barman_gemini25pro_heuristic_18 is None:
            barman_gemini25pro_heuristic_18 = {}

        # Heuristic weights representing the estimated cost of key operations.
        # These values are kept small to help maintain admissibility.
        self.penalties = {
            "make_cocktail": barman_gemini25pro_heuristic_18.get("make_cocktail", 1.0),
            "clean_container": barman_gemini25pro_heuristic_18.get("clean_container", 0.6),
            "add_ingredient": barman_gemini25pro_heuristic_18.get("add_ingredient", 0.4),
            "final_pour": barman_gemini25pro_heuristic_18.get("final_pour", 0.2),
            "manipulation": barman_gemini25pro_heuristic_18.get("manipulation", 0.1),
            "default": barman_gemini25pro_heuristic_18.get("default", 0.1) # Default penalty for unrecognized methods.
        }

        self.method_scores = {}
        self.preprocessing_time = 0.0

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic by pre-calculating costs for all domain methods.
        :param model: The planning model containing domain information.
        :param initial_node: The root node of the search.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        # The initial node has no method applied to it, so its heuristic cost is 0.
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain model and assigns a heuristic cost to each.
        This cost is based on a domain-specific understanding of the task's complexity,
        e.g., cleaning is a costly overhead, making a cocktail is the main work.
        """
        if not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            method_name_lower = method.name.lower()
            score = self.penalties["default"]

            # 'Null' methods are for goals already satisfied; they have zero cost.
            if "null" in method_name_lower:
                score = 0.0
            # 'MakeCocktail' is the core, complex task of mixing ingredients.
            elif "makecocktail" in method_name_lower and "makeandpour" not in method_name_lower:
                score = self.penalties["make_cocktail"]
            # Cleaning is a frequent and necessary step, especially for the shaker.
            elif "clean" in method_name_lower:
                score = self.penalties["clean_container"]
            # Adding ingredients is a multi-step sub-process.
            elif "addingredient" in method_name_lower:
                score = self.penalties["add_ingredient"]
            # The final step of serving the prepared cocktail.
            elif "dopourshakertoshot" in method_name_lower:
                score = self.penalties["final_pour"]
            # Basic robot-arm-like actions (grasping, releasing).
            elif "achieveholding" in method_name_lower or "achievehandempty" in method_name_lower:
                score = self.penalties["manipulation"]
            # The top-level method that includes both making and serving.
            elif "makeandpourcocktail" in method_name_lower:
                score = self.penalties["make_cocktail"] + self.penalties["final_pour"]
            
            self.method_scores[method.name] = score

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a search node.
        The value is the precomputed cost of the decomposition method applied at this node.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0

        method_name = node.method.name
        return self.method_scores.get(method_name, self.penalties["default"])

    def __call__(self, parent_node, node):
        """
        The main heuristic function called by the planner for each new node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic18(penalties={self.penalties})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Barman Gemini 2.5 Pro Heuristic 18 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods Analyzed: {len(self.method_scores)}\n"
            f"\tPenalties: {self.penalties}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s"
        )
