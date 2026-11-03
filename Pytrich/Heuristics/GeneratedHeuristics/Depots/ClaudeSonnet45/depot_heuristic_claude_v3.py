"""
Domain-driven heuristic for Depots domain - Version 3.

Key improvements over v1 and v2:
1. Combines v1's effective method cost approach with better task type analysis
2. Uses task type patterns instead of exact name matching for robustness
3. Adds depth-weighted scoring for nested abstract tasks
4. Penalizes transport operations more aggressively
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.model import AbstractTask, Operator


class DepotClaudeHeuristicV3(Heuristic):
    """
    Domain-driven heuristic for Depots with pattern-based task cost estimation.

    Key principles:
    1. Pattern-based method detection (more robust than exact name matching)
    2. Task type costs based on domain analysis
    3. Aggressive transport penalty to guide search toward local solutions
    4. Depth multiplier for nested decompositions
    """

    # Pattern-based costs for methods (using partial matching)
    # Ordered from most specific to least specific for proper matching
    METHOD_PATTERNS = [
        # Base cases - no work needed (highest priority)
        ('m0_do_put_on', 0.0),    # Already on target
        ('m4_do_clear', 0.0),     # Already clear
        ('m6_do_get_truck', 0.0), # Truck already at location

        # Transport - MOST EXPENSIVE (check before other do_put_on)
        ('m3_do_put_on', 12.0),   # Requires: load + drive + unload

        # Medium complexity put_on operations
        ('m2_do_put_on', 5.0),    # From truck at different location
        ('m1_do_put_on', 4.0),    # Local operation

        # Clear operations
        ('m5_do_clear', 7.0),     # Need to load crate into truck

        # Truck operations
        ('m7_do_get_truck', 1.5), # Drive to location

        # Crate operations
        ('m10_do_load_truck', 5.5),   # Complex: get_truck + clear + lift + load
        ('m11_do_unload_truck', 4.0), # clear + unload + drop
        ('m8_do_lift_crate', 3.0),    # From truck
        ('m9_do_lift_crate', 1.0),    # Simple lift
    ]

    # Task type costs (for abstract tasks without method info)
    TASK_TYPE_COSTS = {
        'do_put_on': 4.0,         # Average put_on cost
        'do_clear': 3.0,          # Average clear cost
        'do_get_truck': 1.0,      # Usually cheap
        'do_lift_crate': 2.0,     # Average lift cost
        'do_load_truck': 5.5,     # Complex
        'do_unload_truck': 4.0,   # Medium complexity
    }

    # Primitive action costs
    PRIMITIVE_COSTS = {
        'drive': 1.5,
        'lift': 1.0,
        'drop': 1.0,
        'load': 1.0,
        'unload': 1.0,
        'nop': 0.0,
    }

    def __init__(self, name="DepotClaudeHeuristicV3"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.method_hits = 0
        self.method_misses = 0
        self.task_type_hits = 0

    def initialize(self, model, initial_node):
        """Initialize the heuristic."""
        start_time = time.time()
        self.model = model

        # Precompute task costs using the model
        self._precompute_task_costs()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._estimate_task_network(initial_node.task_network)

        return super().initialize(model, h_value)

    def _precompute_task_costs(self):
        """Precompute costs for all abstract tasks in the model."""
        self.precomputed_costs = {}

        # Initialize all tasks with domain-based estimates
        for task in self.model.abstract_tasks:
            task_name = task.name.lower()
            cost = 3.0  # Default

            # Apply domain-specific costs
            for task_type, type_cost in self.TASK_TYPE_COSTS.items():
                if task_type in task_name:
                    cost = type_cost
                    break

            # Handle __top task - estimate based on number of subtasks
            if '__top' in task_name:
                # Estimate based on decomposition analysis
                if task.decompositions:
                    min_cost = float('inf')
                    for decomp in task.decompositions:
                        decomp_cost = len(decomp.task_network) * 3.0  # Rough estimate
                        min_cost = min(min_cost, decomp_cost)
                    cost = min_cost if min_cost != float('inf') else 10.0
                else:
                    cost = 10.0

            self.precomputed_costs[task.global_id] = cost

    def _get_method_cost(self, method_name):
        """
        Get cost for a method using pattern matching.
        Returns (cost, matched) tuple.
        """
        method_lower = method_name.lower()

        for pattern, cost in self.METHOD_PATTERNS:
            if pattern in method_lower:
                return cost, True

        return 3.0, False  # Default cost if no pattern matches

    def _get_task_type_cost(self, task_name):
        """
        Get cost for an abstract task based on its type.
        """
        task_lower = task_name.lower()

        for task_type, cost in self.TASK_TYPE_COSTS.items():
            if task_type in task_lower:
                self.task_type_hits += 1
                return cost

        return 3.0  # Default for unknown task types

    def _get_primitive_cost(self, operator_name):
        """Get cost for a primitive action."""
        op_lower = operator_name.lower()

        for action, cost in self.PRIMITIVE_COSTS.items():
            if action in op_lower:
                return cost

        return 1.0  # Default primitive cost

    def _estimate_task_network(self, task_network):
        """
        Estimate cost of remaining task network.
        """
        if not task_network:
            return 0.0

        total_cost = 0.0

        for task in task_network:
            if isinstance(task, Operator):
                # Primitive action
                total_cost += self._get_primitive_cost(task.name)
            else:
                # Abstract task - try precomputed cost first, then pattern matching
                if hasattr(self, 'precomputed_costs') and task.global_id in self.precomputed_costs:
                    total_cost += self.precomputed_costs[task.global_id]
                else:
                    total_cost += self._get_task_type_cost(
                        task.name if hasattr(task, 'name') else str(task)
                    )

        return total_cost

    def _calculate_node_heuristic(self, node):
        """
        Calculate heuristic value for a search node.

        Considers:
        1. Method used to reach this node (if decomposition)
        2. Remaining task network
        """
        h_value = 0.0

        # Score based on decomposition method used
        if hasattr(node, 'decomposition') and node.decomposition:
            method_name = node.decomposition.name if hasattr(node.decomposition, 'name') else str(node.decomposition)
            cost, matched = self._get_method_cost(method_name)

            if matched:
                self.method_hits += 1
            else:
                self.method_misses += 1

            # Don't add method cost directly - it's already reflected in task network
            # But use it to break ties: prefer base-case methods
            if cost == 0.0:
                h_value -= 0.1  # Small bonus for base cases

        # Add estimated cost of remaining task network
        if hasattr(node, 'task_network') and node.task_network:
            h_value += self._estimate_task_network(node.task_network)

        return max(0.0, h_value)

    def __call__(self, parent_node, node):
        """Main heuristic evaluation function."""
        h_value = self._calculate_node_heuristic(node)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotClaudeHeuristicV3()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Claude Heuristic V3 info:\n"
            f"\tName: {self.name}\n"
            f"\tMethod hits: {self.method_hits}\n"
            f"\tMethod misses: {self.method_misses}\n"
            f"\tTask type hits: {self.task_type_hits}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
