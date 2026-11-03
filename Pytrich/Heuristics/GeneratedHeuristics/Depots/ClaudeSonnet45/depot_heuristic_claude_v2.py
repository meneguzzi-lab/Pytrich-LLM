"""
Domain-driven heuristic for Depots domain - Version 2.

Improvements over v1:
1. Precomputes minimum costs for each abstract task based on decomposition analysis
2. Considers the actual subtasks in each decomposition
3. Uses recursive cost estimation for better accuracy
4. Counts primitive actions to estimate actual work remaining
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.model import AbstractTask, Operator, Decomposition


class DepotClaudeHeuristicV2(Heuristic):
    """
    Domain-driven heuristic for Depots with precomputed task costs.

    Key improvements:
    1. Precomputes minimum achievable cost for each abstract task
    2. Analyzes decomposition structure to assign accurate costs
    3. Sums costs over remaining task network
    """

    # Base costs for primitive actions
    PRIMITIVE_COSTS = {
        'drive': 1.5,      # Moving truck between locations is expensive
        'lift': 1.0,       # Lifting a crate
        'drop': 1.0,       # Dropping a crate
        'load': 1.0,       # Loading crate into truck
        'unload': 1.0,     # Unloading crate from truck
        'nop': 0.0,        # No operation
    }

    # Method-specific costs (used as hints during precomputation)
    METHOD_HINTS = {
        # Base cases - no work needed
        'm0_do_put_on': 0.0,      # Already on target
        'm4_do_clear': 0.0,       # Already clear
        'm6_do_get_truck': 0.0,   # Truck already there

        # Transport method - most expensive, should be avoided
        'm3_do_put_on': 10.0,     # Full transport cycle
    }

    def __init__(self, name="DepotClaudeHeuristicV2"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.task_costs = {}      # Precomputed costs for abstract tasks
        self.decomp_costs = {}    # Precomputed costs for decompositions
        self.operator_set = set() # Set of operator names for quick lookup

    def initialize(self, model, initial_node):
        """Initialize with precomputed task costs."""
        start_time = time.time()
        self.model = model

        # Build operator set for quick primitive action detection
        self.operator_set = {op.name.lower() for op in model.operators}

        # Precompute costs for all decompositions and abstract tasks
        self._precompute_costs()

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = self._estimate_task_network(initial_node.task_network)

        return super().initialize(model, h_value)

    def _precompute_costs(self):
        """
        Precompute costs for all decompositions and abstract tasks.
        Uses iterative refinement similar to TDG but with domain-specific costs.
        """
        # Initialize decomposition costs based on subtask analysis
        for decomp in self.model.decompositions:
            cost = self._analyze_decomposition(decomp)
            self.decomp_costs[decomp.global_id] = cost
            self.decomp_costs[decomp.name] = cost

        # Initialize abstract task costs to infinity
        for task in self.model.abstract_tasks:
            self.task_costs[task.global_id] = float('inf')
            self.task_costs[task.name] = float('inf')

        # Iteratively refine costs until convergence
        changed = True
        iterations = 0
        max_iterations = 50

        while changed and iterations < max_iterations:
            changed = False
            iterations += 1

            # Update abstract task costs based on minimum decomposition cost
            for task in self.model.abstract_tasks:
                min_cost = float('inf')
                for decomp in task.decompositions:
                    decomp_cost = self._get_decomp_cost(decomp)
                    if decomp_cost < min_cost:
                        min_cost = decomp_cost

                if min_cost < self.task_costs[task.global_id]:
                    self.task_costs[task.global_id] = min_cost
                    self.task_costs[task.name] = min_cost
                    changed = True

            # Update decomposition costs based on subtask costs
            for decomp in self.model.decompositions:
                new_cost = self._compute_decomp_cost(decomp)
                if abs(new_cost - self.decomp_costs[decomp.global_id]) > 0.01:
                    self.decomp_costs[decomp.global_id] = new_cost
                    self.decomp_costs[decomp.name] = new_cost
                    changed = True

    def _analyze_decomposition(self, decomp):
        """
        Analyze a decomposition to estimate its cost.
        """
        # Check for method hints first
        for hint_name, hint_cost in self.METHOD_HINTS.items():
            if hint_name in decomp.name.lower():
                return hint_cost

        # Count primitive actions and abstract tasks
        primitive_cost = 0.0
        abstract_count = 0

        for subtask in decomp.task_network:
            if isinstance(subtask, Operator):
                # Get cost based on action type
                subtask_name = subtask.name.lower()
                for action, cost in self.PRIMITIVE_COSTS.items():
                    if action in subtask_name:
                        primitive_cost += cost
                        break
                else:
                    primitive_cost += 1.0  # Default primitive cost
            else:
                # Abstract task - will be refined later
                abstract_count += 1

        # Initial estimate: primitive cost + estimate for abstract tasks
        return primitive_cost + abstract_count * 2.0

    def _get_decomp_cost(self, decomp):
        """Get the current cost estimate for a decomposition."""
        if decomp.global_id in self.decomp_costs:
            return self.decomp_costs[decomp.global_id]
        return self._analyze_decomposition(decomp)

    def _compute_decomp_cost(self, decomp):
        """
        Compute decomposition cost based on current task cost estimates.
        """
        # Check for method hints
        for hint_name, hint_cost in self.METHOD_HINTS.items():
            if hint_name in decomp.name.lower():
                # For base cases, return the hint directly
                if hint_cost == 0.0:
                    return 0.0

        total_cost = 0.0

        for subtask in decomp.task_network:
            if isinstance(subtask, Operator):
                # Primitive action cost
                subtask_name = subtask.name.lower()
                action_cost = 1.0
                for action, cost in self.PRIMITIVE_COSTS.items():
                    if action in subtask_name:
                        action_cost = cost
                        break
                total_cost += action_cost
            else:
                # Abstract task - use precomputed cost
                task_cost = self.task_costs.get(subtask.global_id,
                            self.task_costs.get(subtask.name, 3.0))
                if task_cost == float('inf'):
                    task_cost = 3.0  # Default estimate
                total_cost += task_cost

        # Apply method hint as a multiplier for expensive methods
        for hint_name, hint_cost in self.METHOD_HINTS.items():
            if hint_name in decomp.name.lower() and hint_cost > 0:
                # Use hint as minimum cost
                total_cost = max(total_cost, hint_cost)
                break

        return total_cost

    def _estimate_task_network(self, task_network):
        """
        Estimate the cost of completing the remaining task network.
        """
        if not task_network:
            return 0.0

        total_cost = 0.0

        for task in task_network:
            if isinstance(task, Operator):
                # Primitive action
                task_name = task.name.lower()
                action_cost = 1.0
                for action, cost in self.PRIMITIVE_COSTS.items():
                    if action in task_name:
                        action_cost = cost
                        break
                total_cost += action_cost
            else:
                # Abstract task - use precomputed cost
                task_cost = self.task_costs.get(task.global_id,
                            self.task_costs.get(task.name, 3.0))
                if task_cost == float('inf'):
                    task_cost = 3.0
                total_cost += task_cost

        return total_cost

    def __call__(self, parent_node, node):
        """Main heuristic evaluation function."""
        h_value = self._estimate_task_network(node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotClaudeHeuristicV2()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Claude Heuristic V2 info:\n"
            f"\tName: {self.name}\n"
            f"\tAbstract tasks: {len([k for k in self.task_costs.keys() if isinstance(k, int)])}\n"
            f"\tDecompositions: {len([k for k in self.decomp_costs.keys() if isinstance(k, int)])}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
