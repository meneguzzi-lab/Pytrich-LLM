"""
Domain-driven heuristic for Depots domain - Version 5.

Approach: TDG-style computation with domain-specific primitive costs only.
No method penalties - just weighted primitive actions.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.ProblemRepresentation.and_or_graph import AndOrGraph, ContentType, NodeType


class DepotClaudeHeuristicV5(Heuristic):
    """
    TDG-style heuristic with domain-specific primitive action costs.

    Key insight: Drive operations are more expensive than other actions,
    so methods requiring drives should naturally cost more.
    """

    # Primitive action costs - drive is penalized
    PRIMITIVE_COSTS = {
        'drive': 3.0,      # Driving between locations is expensive
        'lift': 1.0,
        'drop': 1.0,
        'load': 1.0,
        'unload': 1.0,
        'nop': 0.0,
    }

    def __init__(self, name="DepotClaudeHeuristicV5"):
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

        # Compute values
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
        op_lower = operator_name.lower() if operator_name else ''
        for action, cost in self.PRIMITIVE_COSTS.items():
            if action in op_lower:
                return cost
        return 1.0

    def _compute_values(self):
        """Compute AND/OR graph values with weighted primitives."""
        # Initialize
        for node in self.and_or_graph.nodes:
            if not node:
                continue
            if node.content_type == ContentType.OPERATOR:
                node.value = self._get_primitive_cost(
                    node.name if hasattr(node, 'name') else ''
                )
            else:
                node.value = float('inf')

        # Iterate until convergence
        changed = True
        while changed:
            self.iterations += 1
            changed = False

            for node in self.and_or_graph.nodes:
                if not node:
                    continue

                new_value = node.weight
                if node.type == NodeType.OR:
                    if node.predecessors:
                        new_value += min(n.value for n in node.predecessors)
                elif node.type == NodeType.AND:
                    if node.predecessors:
                        new_value += sum(n.value for n in node.predecessors)

                if new_value != node.value and new_value != float('inf'):
                    changed = True
                    node.value = new_value

    def __call__(self, parent_node, node):
        """Main heuristic evaluation."""
        h_value = sum(self.task_values.get(task.global_id, float('inf'))
                      for task in node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return "DepotClaudeHeuristicV5()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Claude Heuristic V5 info:\n"
            f"\tName: {self.name}\n"
            f"\tGraph size: {len(self.task_values)}\n"
            f"\tIterations: {self.iterations}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
