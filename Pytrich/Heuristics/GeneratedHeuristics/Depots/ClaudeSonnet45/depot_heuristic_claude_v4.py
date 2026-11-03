"""
Domain-driven heuristic for Depots domain - Version 4.

Key approach:
1. Uses TDG-style iterative cost refinement via AND/OR graph analysis
2. Applies domain-specific primitive action costs
3. Adds method preference penalties to guide decomposition choices
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.ProblemRepresentation.and_or_graph import AndOrGraph, ContentType, NodeType
from Pytrich.model import Operator


class DepotClaudeHeuristicV4(Heuristic):
    """
    Domain-driven heuristic using TDG-style computation with domain-specific costs.
    """

    # Primitive action costs based on domain analysis
    PRIMITIVE_COSTS = {
        'drive': 2.0,      # Moving truck is expensive
        'lift': 1.0,       # Standard action
        'drop': 1.0,       # Standard action
        'load': 1.0,       # Standard action
        'unload': 1.0,     # Standard action
        'nop': 0.0,        # No-op is free
    }

    # Method preference penalties (added to base cost)
    # Higher values = less preferred
    METHOD_PENALTIES = {
        # Transport methods - penalize to prefer local solutions
        'm3_do_put_on': 3.0,     # Full transport - most expensive
        'm5_do_clear': 2.0,      # Clear by loading into truck

        # Medium complexity
        'm2_do_put_on': 1.0,     # From truck at location
        'm10_do_load_truck': 1.0,

        # Base cases - reward (negative penalty)
        'm0_do_put_on': -0.5,    # Already done
        'm4_do_clear': -0.5,     # Already clear
        'm6_do_get_truck': -0.5, # Truck already there
    }

    def __init__(self, name="DepotClaudeHeuristicV4"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.task_values = {}
        self.iterations = 0
        self.and_or_graph = None

    def initialize(self, model, initial_node):
        """Initialize with AND/OR graph computation."""
        start_time = time.time()
        self.model = model

        # Build AND/OR graph
        self.and_or_graph = AndOrGraph(model, graph_type=3)

        # Compute values with domain-specific costs
        self._compute_values()

        # Extract values for lookup
        for node in self.and_or_graph.nodes:
            if node and node.content_type in {ContentType.OPERATOR, ContentType.ABSTRACT_TASK}:
                self.task_values[node.ID] = node.value

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value
        h_value = sum(self.task_values.get(task.global_id, float('inf'))
                      for task in initial_node.task_network)

        return super().initialize(model, h_value)

    def _get_primitive_cost(self, operator_name):
        """Get domain-specific cost for a primitive action."""
        op_lower = operator_name.lower()
        for action, cost in self.PRIMITIVE_COSTS.items():
            if action in op_lower:
                return cost
        return 1.0  # Default

    def _get_method_penalty(self, method_name):
        """Get penalty for a decomposition method."""
        method_lower = method_name.lower()
        for pattern, penalty in self.METHOD_PENALTIES.items():
            if pattern in method_lower:
                return penalty
        return 0.0  # No penalty for unknown methods

    def _compute_values(self):
        """
        Compute AND/OR graph values using domain-specific costs.
        Similar to TDG but with weighted primitive actions and method preferences.
        """
        # Initialize node values
        for node in self.and_or_graph.nodes:
            if not node:
                continue

            if node.content_type == ContentType.OPERATOR:
                # Use domain-specific primitive cost
                node.value = self._get_primitive_cost(node.name if hasattr(node, 'name') else '')
            else:
                node.value = float('inf')

        # Iteratively refine values
        changed = True
        while changed:
            self.iterations += 1
            changed = False

            for node in self.and_or_graph.nodes:
                if not node:
                    continue

                if node.type == NodeType.OR:
                    # OR node: minimum of predecessors + weight
                    if node.predecessors:
                        # Add method penalty based on node name
                        penalty = self._get_method_penalty(
                            node.name if hasattr(node, 'name') else ''
                        )
                        new_value = node.weight + penalty + min(n.value for n in node.predecessors)
                        if new_value != node.value:
                            changed = True
                            node.value = new_value

                elif node.type == NodeType.AND:
                    # AND node: sum of predecessors + weight
                    if node.predecessors:
                        pred_sum = sum(n.value for n in node.predecessors)
                        if pred_sum != float('inf'):
                            new_value = node.weight + pred_sum
                            if new_value != node.value:
                                changed = True
                                node.value = new_value

    def __call__(self, parent_node, node):
        """Main heuristic evaluation function."""
        h_value = sum(self.task_values.get(task.global_id, float('inf'))
                      for task in node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotClaudeHeuristicV4()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Claude Heuristic V4 info:\n"
            f"\tName: {self.name}\n"
            f"\tGraph size: {len(self.task_values)}\n"
            f"\tIterations: {self.iterations}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
