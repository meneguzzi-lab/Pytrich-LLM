import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

class DepotGemini25ProHeuristic1(Heuristic):
    """
    A domain-dependent heuristic for the HDDL Depot domain.

    This heuristic estimates the cost of a decomposition method by analyzing its subtasks.
    It assigns penalties to methods based on the complexity and resource-intensiveness of the
    operations they entail, such as driving trucks, loading/unloading crates, and clearing
    surfaces. The core idea is to guide the planner towards solutions that minimize costly
    transportation and handling actions.

    The heuristic values are precomputed for all methods during initialization to ensure
    fast evaluation during the search. Penalties are chosen to reflect the relative costs:
    - Driving between locations is the most expensive operation.
    - Loading/unloading requires coordinating a hoist and a truck, making it moderately expensive.
    - Clearing a surface is penalized as it may require moving other crates.
    - Lifting/dropping is the cheapest basic handling operation.
    - Getting a truck is penalized as it may involve a drive action.
    """

    def __init__(self, depot_heuristic_1=None):
        """
        Initializes the heuristic with predefined penalties for domain operations.
        """
        super().__init__(name="depot_gemini25pro_heuristic_1")
        # Penalties for various operations, designed to guide the search effectively.
        # Values are kept small and relative to each other to act as a good guide
        # while aiming for admissibility.
        self.penalties = {
            "DRIVE": 0.5,          # Highest cost for inter-place travel
            "LOAD_UNLOAD": 0.3,    # Cost for using a hoist to move crate to/from truck
            "CLEAR": 0.2,          # Cost for clearing a surface, may involve moving crates
            "LIFT_DROP": 0.1,      # Basic cost for picking up or putting down a crate
            "GET_TRUCK": 0.2       # Cost for ensuring a truck is present, may involve driving
        }
        self.method_costs = {}
        self.preprocessing_time = 0.0

    def initialize(self, model: Model, initial_node: HTNNode):
        """
        Initializes the heuristic by precomputing the costs for all methods in the domain.
        """
        start_time = time.time()
        self.model = model
        self._precompute_method_costs()
        self.preprocessing_time = time.time() - start_time
        h_value = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h_value)

    def _precompute_method_costs(self):
        """
        Analyzes all decomposition methods in the domain model and assigns a heuristic cost
        based on the primitive actions and abstract tasks in their subtask networks. This
        cost reflects the operational complexity of choosing a particular method.
        """
        if not self.model or not hasattr(self.model, 'decompositions'):
            return

        for method in self.model.decompositions:
            cost = 0.0
            subtasks = []
            if hasattr(method, 'task_network'):
                subtasks = method.task_network

            for subtask in subtasks:
                subtask_name = subtask.name.lower()

                # Penalize based on primitive actions
                if 'drive' in subtask_name:
                    cost += self.penalties["DRIVE"]
                elif 'load' in subtask_name:
                    cost += self.penalties["LOAD_UNLOAD"]
                elif 'unload' in subtask_name:
                    cost += self.penalties["LOAD_UNLOAD"]
                elif 'lift' in subtask_name:
                    cost += self.penalties["LIFT_DROP"]
                elif 'drop' in subtask_name:
                    cost += self.penalties["LIFT_DROP"]
                # Penalize based on abstract tasks which imply future work
                elif 'do_clear' in subtask_name:
                    cost += self.penalties["CLEAR"]
                elif 'do_get_truck' in subtask_name:
                    cost += self.penalties["GET_TRUCK"]
                elif 'do_lift_crate' in subtask_name:
                    cost += self.penalties["LIFT_DROP"]
                elif 'do_load_truck' in subtask_name:
                    cost += self.penalties["GET_TRUCK"] + self.penalties["CLEAR"] + self.penalties["LOAD_UNLOAD"]
                elif 'do_unload_truck' in subtask_name:
                    cost += self.penalties["CLEAR"] + self.penalties["LOAD_UNLOAD"] + self.penalties["LIFT_DROP"]
            
            self.method_costs[method.name] = cost

    def _calculate_node_heuristic(self, node: HTNNode) -> float:
        """
        Calculates the heuristic value for a given search node by looking up the
        precomputed cost of the method that produced it.
        """
        if not hasattr(node, 'method') or not node.method:
            return 0.0
        
        method_name = node.method.name
        # Return a default cost of 1.0 if a method was not found during precomputation.
        return self.method_costs.get(method_name, 1.0)

    def __call__(self, parent_node: HTNNode, node: HTNNode) -> float:
        """
        The main entry point for the heuristic evaluation.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __str__(self) -> str:
        return f"{self.__class__.__name__} with precomputed costs for {len(self.method_costs)} methods."

    def __output__(self) -> str:
        """
        Returns a formatted string with details about the heuristic's configuration and precomputed costs.
        """
        cost_items = sorted(self.method_costs.items())
        cost_str = "\n\t".join([f"{name}: {cost:.2f}" for name, cost in cost_items])
        return (
            f"Depot Heuristic Info:\n"
            f"\tName: {self.name}\n"
            f"\tPreprocessing Time: {self.preprocessing_time:.4f} s\n"
            f"\tPrecomputed Method Costs ({len(self.method_costs)} methods):\n\t{cost_str}"
        )
