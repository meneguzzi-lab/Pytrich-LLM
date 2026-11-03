import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model, Operator, AbstractTask

class BarmanGemini25ProHeuristic4(Heuristic):
    """
    A domain-dependent heuristic for the Barman HDDL domain.

    Design Rationale:
    This heuristic is based on the concept of a relaxed plan graph cost adapted for
    Hierarchical Task Networks (HTNs). It estimates the minimum number of primitive
    actions required to complete any abstract task, ignoring all preconditions.
    This provides an admissible (never overestimates the true cost) and highly
    informative cost-to-go estimate.

    Domain Insights:
    The Barman domain involves a structured, multi-step process for creating cocktails:
    cleaning containers, adding multiple ingredients, shaking, and pouring. The main
    source of plan length is the repetition of these sub-processes for each ingredient
    and each final cocktail. A simple action-counting heuristic captures this complexity
    well.

    By pre-calculating the minimum action count for each abstract task (e.g.,
    'AchieveContainsShakerIngredient'), the heuristic can quickly evaluate the cost
    of a decomposition method by summing the costs of its subtasks. This guides the
    planner towards methods that decompose into a smaller number of primitive actions,
    effectively pruning less efficient branches of the search space.

    Implementation:
    1.  `initialize()`: Triggers a one-time pre-computation.
    2.  `_precompute_task_costs()`: Iteratively calculates the cost for each abstract task.
        It starts with a cost of 1 for all primitive actions and infinity for abstract
        tasks. It then propagates costs up the hierarchy based on the domain's
        decomposition methods until the costs stabilize. This is a fixed-point
        calculation that finds the shortest path in the domain's implicit AND/OR graph.
    3.  `__call__()`: During search, for any given node, it retrieves the method being
        applied and calculates the heuristic value by summing the pre-computed costs of
        the method's subtasks. This is computationally very cheap.
    """
    def __init__(self, barman_gemini25pro_heuristic_4):
        """
        Initializes the heuristic.

        :param barman_gemini25pro_heuristic_4: A parameter as required by the task specification.
        """
        super().__init__(name="barman_gemini25pro_heuristic_4")
        self.preprocessing_time = 0.0
        self.task_costs = {}
        self.model = None

    def initialize(self, model, initial_node):
        """
        Initializes the heuristic with the domain model and pre-computes task costs.

        :param model: The planning model.
        :param initial_node: The initial node of the search.
        :return: The result of the parent class's initialize method.
        """
        start_time = time.time()
        self.model = model
        self._precompute_task_costs()
        self.preprocessing_time = time.time() - start_time

        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _get_task_key(self, task_name):
        """
        Extracts a canonical key from a grounded task name.
        Example: '(clean-shaker shaker1 left right)' -> 'clean-shaker'
        """
        return task_name.split()[0].strip('()')

    def _precompute_task_costs(self):
        """
        Iteratively calculates the relaxed plan cost (minimum action count) for each
        abstract task in the domain. The costs are stored in `self.task_costs`.
        """
        # 1. Initialize costs: 1 for primitives, infinity for abstract tasks.
        for op in self.model.operators:
            key = self._get_task_key(op.name)
            # Set cost to 1 for all primitive action types
            self.task_costs[key] = 1

        for at in self.model.abstract_tasks:
            key = self._get_task_key(at.name)
            self.task_costs.setdefault(key, float('inf'))

        # 2. Iteratively update costs until they stabilize (fixed-point calculation).
        changed = True
        while changed:
            changed = False
            for at in self.model.abstract_tasks:
                at_key = self._get_task_key(at.name)

                if not hasattr(at, 'decompositions') or not at.decompositions:
                    continue

                min_method_cost = float('inf')
                for method in at.decompositions:
                    current_method_cost = 0
                    if not hasattr(method, 'task_network'):
                        continue

                    # Sum the costs of all subtasks in the method's decomposition
                    for subtask in method.task_network:
                        subtask_key = self._get_task_key(subtask.name)
                        cost = self.task_costs.get(subtask_key, 1)  # Default to 1 for primitives
                        current_method_cost += cost
                    
                    min_method_cost = min(min_method_cost, current_method_cost)

                # If we found a cheaper way to achieve this task, update its cost.
                if min_method_cost < self.task_costs.get(at_key, float('inf')):
                    if self.task_costs[at_key] != min_method_cost:
                        self.task_costs[at_key] = min_method_cost
                        changed = True

    def _calculate_node_heuristic(self, node):
        """
        Calculates the heuristic value for a search node.
        The value is the sum of the pre-computed costs of the subtasks of the node's method.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0

        method = node.method
        if not hasattr(method, 'task_network'):
            return 0
        
        h_value = 0
        for subtask in method.task_network:
            subtask_key = self._get_task_key(subtask.name)
            cost = self.task_costs.get(subtask_key, 1) # Default cost of 1
            h_value += cost
        
        return h_value

    def __call__(self, parent_node, node):
        """
        The main heuristic evaluation function called by the planner.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"BarmanGemini25ProHeuristic4()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        """Returns a string with summary information about the heuristic."""
        return (
            f"Barman Gemini 2.5 Pro Heuristic 4 Info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f}s\n"
            f"\tNumber of pre-computed task costs: {len(self.task_costs)}\n"
        )
