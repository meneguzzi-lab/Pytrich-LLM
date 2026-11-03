import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class RobotGemini25ProHeuristic6(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Robot domain.

    This heuristic guides the search by assigning a cost to each method decomposition.
    The core idea is to prioritize actions that make direct progress towards the goal
    of delivering packages. It establishes a clear preference order:

    1.  `putdown`: Completing a delivery is the highest priority.
    2.  `pickup`: Starting a delivery is the next priority.
    3.  `open`: Opening a door is a necessary setup action but less direct than handling a package.
    4.  `move`: Moving is the most common and least goal-directed action on its own.

    By assigning lower costs (heuristics values) to more desirable methods, the planner
    is encouraged to explore more promising paths first. This heuristic is static,
    meaning it scores methods based on their structure rather than the current world
    state. This makes it computationally very efficient and scalable, as the cost for
    each method is pre-computed during initialization.

    The heuristic costs are designed to be small floats to act as effective tie-breakers
    and guides without drastically altering the search behavior for problems where all
    paths are very short.
    """
    def __init__(self, robot_gemini25pro_heuristic_6=None):
        """
        Initializes the RobotGemini25ProHeuristic.

        Args:
            robot_gemini25pro_heuristic_6 (dict, optional): A dictionary of weights to customize the heuristic's behavior.
                                                Keys can be 'putdown_cost', 'pickup_cost', 'open_cost', 'move_cost'.
                                                Defaults are provided for a sensible heuristic.
        """
        super().__init__(name="robot_gemini25pro_heuristic_6")

        # Set default costs/priorities. Lower is better.
        config = {
            'putdown_cost': 0.1,  # Highest priority
            'pickup_cost': 0.2,
            'open_cost': 0.3,
            'move_cost': 0.4      # Lowest priority
        }
        if isinstance(robot_gemini25pro_heuristic_6, dict):
            config.update(robot_gemini25pro_heuristic_6)

        self.putdown_cost = config['putdown_cost']
        self.pickup_cost = config['pickup_cost']
        self.open_cost = config['open_cost']
        self.move_cost = config['move_cost']

        self.method_scores = {}
        self.preprocessing_time = 0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by pre-computing scores for all methods in the domain model.

        Args:
            model (Model): The planning domain and problem model.
            initial_node (HTNNode): The root node of the search.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start_time
        
        # The initial heuristic value can be 0 as costs are applied per decomposition.
        return super().initialize(model, 0)

    def _precompute_method_info(self):
        """
        Analyzes all methods in the domain and assigns a heuristic score based on the
        type of subtasks they contain. This score is stored for fast lookup during search.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        methods = self.model.decompositions

        for method in methods:
            # Default score is the highest cost (for 'move')
            score = self.move_cost
            
            # Get the names of all subtasks in the method's decomposition
            subtask_names = []
            if hasattr(method, 'task_network'):
                for subtask in method.task_network:
                    if hasattr(subtask, 'name'):
                        subtask_names.append(subtask.name.lower())

            # Assign score based on a priority of subtasks
            if not subtask_names:
                # This is a terminal method, like 'finished'. Give it the best score.
                score = 0.0
            elif any('putdown' in name for name in subtask_names):
                score = self.putdown_cost
            elif any('pickup' in name for name in subtask_names):
                score = self.pickup_cost
            elif any('open' in name for name in subtask_names):
                score = self.open_cost
            
            # If none of the above, it defaults to the move_cost.
            self.method_scores[method.name] = score

    def __call__(self, parent_node: HTNNode, node: HTNNode):
        """
        Calculates the heuristic value for a search node.

        This is done by looking up the pre-computed score for the method that
        was used to generate this node. A small cost is added to the parent's cost.

        Args:
            parent_node (HTNNode): The parent node in the search tree.
            node (HTNNode): The current node to be evaluated.

        Returns:
            float: The heuristic value for the node.
        """
        method_cost = 0.0
        if hasattr(node, 'method') and node.method:
            method_name = node.method.name
            # Look up the score, default to the highest cost if not found
            method_cost = self.method_scores.get(method_name, self.move_cost)
        
        # The heuristic value is incremental. 
        # Here we just return the cost of the single step (method application).
        # The planner's search algorithm (e.g., A*) will sum this up.
        h_value = method_cost
        
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"RobotGemini25ProHeuristic6(costs={{'putdown': {self.putdown_cost}, 'pickup': {self.pickup_cost}, 'open': {self.open_cost}, 'move': {self.move_cost}}})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Robot Gemini 2.5 Pro Heuristic 6 Info:\n"
            f"\tName: {self.name}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tCosts:\n"
            f"\t  - Putdown: {self.putdown_cost}\n"
            f"\t  - Pickup: {self.pickup_cost}\n"
            f"\t  - Open: {self.open_cost}\n"
            f"\t  - Move: {self.move_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
