"""
Domain-driven heuristic for Depots domain - Version 6.

Hybrid approach: TDG base values + domain-specific task scoring for tie-breaking.
"""

import time
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.ProblemRepresentation.and_or_graph import AndOrGraph, ContentType, NodeType
from Pytrich.model import Operator


class DepotClaudeHeuristicV6(Heuristic):
    """
    Hybrid heuristic combining TDG computation with domain task scoring.

    Uses TDG values as base, adds small domain-specific adjustments
    to improve tie-breaking in search.
    """

    # Task type adjustments (added to TDG values)
    # Small values to only affect tie-breaking, not overall ordering
    TASK_ADJUSTMENTS = {
        'do_put_on': 0.3,        # Slightly prefer fewer put_on tasks
        'do_clear': 0.2,         # Clear is relatively cheap
        'do_get_truck': 0.1,     # Usually cheap
        'do_lift_crate': 0.15,
        'do_load_truck': 0.4,    # Loading is complex
        'do_unload_truck': 0.35,
    }

    def __init__(self, name="DepotClaudeHeuristicV6"):
        super().__init__(name=name)
        self.preprocessing_time = 0
        self.tdg_values = {}
        self.iterations = 0
        self.and_or_graph = None

    def initialize(self, model, initial_node):
        """Initialize with TDG computation."""
        start_time = time.time()
        self.model = model

        # Build AND/OR graph (standard TDG)
        self.and_or_graph = AndOrGraph(model, graph_type=3)
        self._compute_tdg()

        # Extract TDG values
        for node in self.and_or_graph.nodes:
            if node and node.content_type in {ContentType.OPERATOR, ContentType.ABSTRACT_TASK}:
                self.tdg_values[node.ID] = node.value

        self.preprocessing_time = time.time() - start_time

        # Calculate initial heuristic value with adjustments
        h_value = self._estimate_task_network(initial_node.task_network)

        return super().initialize(model, h_value)

    def _compute_tdg(self):
        """Standard TDG computation."""
        for node in self.and_or_graph.nodes:
            if not node:
                continue
            if node.content_type == ContentType.OPERATOR:
                node.value = 0
            else:
                node.value = float('inf')

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

    def _get_task_adjustment(self, task_name):
        """Get domain-specific adjustment for a task type."""
        task_lower = task_name.lower()
        for task_type, adjustment in self.TASK_ADJUSTMENTS.items():
            if task_type in task_lower:
                return adjustment
        return 0.2  # Default small adjustment

    def _estimate_task_network(self, task_network):
        """Estimate cost using TDG values + domain adjustments."""
        if not task_network:
            return 0.0

        total_cost = 0.0

        for task in task_network:
            # Get TDG base value
            tdg_val = self.tdg_values.get(task.global_id, 1.0)

            if isinstance(task, Operator):
                total_cost += tdg_val
            else:
                # Add small domain-specific adjustment for tie-breaking
                task_name = task.name if hasattr(task, 'name') else str(task)
                adjustment = self._get_task_adjustment(task_name)
                total_cost += tdg_val + adjustment

        return total_cost

    def __call__(self, parent_node, node):
        """Main heuristic evaluation."""
        h_value = self._estimate_task_network(node.task_network)
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return "DepotClaudeHeuristicV6()"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Claude Heuristic V6 info:\n"
            f"\tName: {self.name}\n"
            f"\tGraph size: {len(self.tdg_values)}\n"
            f"\tIterations: {self.iterations}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.4f} s\n"
        )
