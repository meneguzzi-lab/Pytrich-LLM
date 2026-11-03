"""
Domain-driven heuristic for Depots domain.

This heuristic is based on careful analysis of the Depots domain structure:
- 12 methods with varying complexity
- Transport operations (m3_do_put_on) are most expensive
- Base cases (m0, m4, m6) have zero cost

Key insight: The existing LLM heuristics failed because they looked for
keywords like 'move-truck', 'hoist', 'deliver' which don't exist in the
actual domain. This heuristic uses the actual method names.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic


class DepotClaudeHeuristicV1(Heuristic):
    """
    Domain-driven heuristic for Depots based on method complexity analysis.

    Key principles:
    1. Assign accurate costs to each method based on decomposition depth
    2. Heavily penalize transport methods (m3_do_put_on)
    3. Reward base-case methods (m0, m4, m6)
    4. Consider remaining task network size
    """

    # Method costs based on domain analysis
    # These reflect the actual decomposition complexity
    METHOD_COSTS = {
        # Base cases (already satisfied) - goal is reached, no work needed
        'm0_do_put_on': 0.0,      # crate already on target → nop
        'm4_do_clear': 0.0,       # surface already clear → nop
        'm6_do_get_truck': 0.0,   # truck already at location → nop

        # Simple single-action operations
        'm7_do_get_truck': 1.0,   # Drive truck to location
        'm9_do_lift_crate': 1.0,  # Lift crate from surface

        # Local compound operations (crate at same location as target)
        'm1_do_put_on': 3.5,      # clear + clear + lift_crate + Drop
        'm8_do_lift_crate': 2.5,  # get_truck + Unload (crate in truck)
        'm11_do_unload_truck': 3.5, # clear + Unload + Drop

        # Medium complexity (involves truck coordination)
        'm2_do_put_on': 4.5,      # get_truck + clear + Unload + Drop
        'm10_do_load_truck': 5.0, # get_truck + clear + Lift + Load

        # High complexity (clearing requires putting crate in truck)
        'm5_do_clear': 6.0,       # clear + Lift + get_truck + Load

        # TRANSPORT (most expensive - involves driving between locations)
        'm3_do_put_on': 8.0,      # load_truck + Drive + unload_truck
    }

    # Abstract task base costs (for task network estimation)
    TASK_COSTS = {
        'do_put_on': 3.0,         # Average cost of put_on methods
        'do_clear': 2.0,          # Average cost of clear methods
        'do_get_truck': 0.5,      # Usually cheap (truck often nearby)
        'do_lift_crate': 1.5,     # Average lift cost
        'do_load_truck': 5.0,     # Complex operation
        'do_unload_truck': 3.5,   # Medium complexity
    }

    def __init__(self, name="DepotClaudeHeuristicV1"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.method_hits = 0
        self.method_misses = 0

    def initialize(self, model, initial_node):
        """
        Initialize the heuristic with the model.
        """
        start_time = time.time()
        self.model = model
        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value based on task network
        h_value = self._estimate_task_network(initial_node.task_network)

        return super().initialize(model, h_value)

    def _estimate_task_network(self, task_network):
        """
        Estimate the cost of completing the remaining task network.
        """
        if not task_network:
            return 0.0

        total_cost = 0.0
        for task in task_network:
            task_name = self._get_task_name(task)

            # Look up base cost for this task type
            for base_name, cost in self.TASK_COSTS.items():
                if base_name in task_name:
                    total_cost += cost
                    break
            else:
                # Unknown task - use default cost
                total_cost += 2.0

        return total_cost

    def _get_task_name(self, task):
        """Extract task name from various possible formats."""
        if hasattr(task, 'name'):
            return task.name.lower()
        return str(task).lower()

    def _get_method_name(self, method):
        """Extract method name from various possible formats."""
        if hasattr(method, 'name'):
            return method.name
        return str(method)

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a search node.

        The value is based on:
        1. The method used to reach this node (if applicable)
        2. The remaining task network
        """
        h_value = 0.0

        # Score based on the decomposition/method used
        if hasattr(node, 'decomposition') and node.decomposition:
            method_name = self._get_method_name(node.decomposition)
            if method_name in self.METHOD_COSTS:
                h_value += self.METHOD_COSTS[method_name]
                self.method_hits += 1
            else:
                # Unknown method - use moderate cost
                self.method_misses += 1
                h_value += 2.0

        # Add estimated cost of remaining task network
        if hasattr(node, 'task_network') and node.task_network:
            h_value += self._estimate_task_network(node.task_network)

        return h_value

    def __call__(self, parent_node, node):
        """
        Main heuristic evaluation function.
        """
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotClaudeHeuristicV1()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Claude Heuristic V1 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethod hits: {self.method_hits}\n"
            f"\tMethod misses: {self.method_misses}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
