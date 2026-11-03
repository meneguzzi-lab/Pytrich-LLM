import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.Search.htn_node import HTNNode
from Pytrich.model import Model

# Domain-dependent heuristic for the Depot domain
class DepotGemini20FlashHeuristic37(Heuristic):
    def __init__(self, depot_gemini20flash_heuristic_37=1.0, name="DepotGemini20FlashHeuristic37"):
        super().__init__(name=name)
        self.truck_move_cost = depot_gemini20flash_heuristic_37  # Cost for moving a truck
        self.hoist_cost = 0.5       # Cost for hoisting operations
        self.deliver_cost = 1.0     # Cost for delivering a crate
        self.preprocessing_time = 0

        self.location_map = {}

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model and perform preprocessing if needed.
        """
        start_time = time.time()
        self.model = model
        self._precompute_location_info()
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._calculate_node_heuristic(initial_node)

        return super().initialize(model, h_value)

    def _precompute_location_info(self):
        """
        Precompute location information to optimize heuristic calculations.
        For example, create a mapping of locations to their types (depot, distributor).
        """
        # Example (replace with actual logic based on domain file if available)
        self.location_map = {}
        for fact in self.model.facts:
            fact_name = fact.name.lower()
            if 'depot' in fact_name:
                self.location_map[fact.name] = 'depot'
            elif 'distributor' in fact_name:
                self.location_map[fact.name] = 'distributor'
            elif 'truck' in fact_name:
              self.trucks = fact.name # save the trucks to calculate the heuristic


    def _calculate_node_heuristic(self, node):
        """
        Calculate the heuristic value for a given node.
        """
        h_value = 0

        # Estimate cost of remaining tasks based on the node's task network
        if hasattr(node, 'task_network') and node.task_network:
            for task in node.task_network:
                task_name = task.name.lower() if hasattr(task, 'name') else str(task).lower()

                if 'deliver' in task_name or 'move' in task_name:
                    h_value += self.deliver_cost  # Assume a delivery task
                elif 'hoist' in task_name:
                    h_value += self.hoist_cost  # Assume a hoisting task

                # Add cost for truck movement
                if 'move-truck' in task_name or 'drive-truck' in task_name:
                    h_value += self.truck_move_cost
        elif hasattr(node, 'method') and node.method:
            method_name = node.method.name.lower()
            if 'deliver' in method_name:
                h_value += self.deliver_cost #Assume a delivery task
            elif 'hoist' in method_name:
                h_value += self.hoist_cost #Assume a hoisting task
            elif 'move-truck' in method_name or 'drive-truck' in method_name:
                h_value += self.truck_move_cost

        # Add heuristic based on state (very basic, can be significantly improved)
        if hasattr(node, 'state'):
            state = node.state
            # Count unmet goals. Requires to look into model
            unmet_goals = 0
            for goal in self.model.goals:
                if not (state & (1 << goal)):
                    unmet_goals += 1
            h_value += unmet_goals * self.deliver_cost

        return h_value

    def __call__(self, parent_node, node):
        """
        Calculate and return the heuristic value for the given node.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotGemini20FlashHeuristic37(truck_move_cost={self.truck_move_cost}, hoist_cost={self.hoist_cost}, deliver_cost={self.deliver_cost})"

    def __str__(self):
        return f"DepotGemini20FlashHeuristic37(truck_move_cost={self.truck_move_cost}, hoist_cost={self.hoist_cost}, deliver_cost={self.deliver_cost})"

    def __output__(self):
        return (
            f"Depot Gemini 2.0 Flash Heuristic 37 info:\n"
            f"\tName: {self.name}\n"
            f"\tTruck Move Cost: {self.truck_move_cost}\n"
            f"\tHoist Cost: {self.hoist_cost}\n"
            f"\tDeliver Cost: {self.deliver_cost}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.2f} s\n"
            f"\tLocation Mapping: {self.location_map}\n"
        )
