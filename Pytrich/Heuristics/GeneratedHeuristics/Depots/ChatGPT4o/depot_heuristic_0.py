"""
DepotChatGPTHeuristic0
Creative heuristic based on dependency-graph depth in stacking goals.
"""

from typing import Any, Dict, List, Set, Tuple
from Pytrich.Heuristics.heuristic import Heuristic


class DepotChatGPTHeuristic0(Heuristic):
    def __init__(self, depot_gpt_heuristic_0: str = "depot_gpt_heuristic_0"):
        super().__init__(name=depot_gpt_heuristic_0)
        self.depot_gpt_heuristic_0 = depot_gpt_heuristic_0
        self.model = None
        self.goal_edges: Dict[str, str] = {}  # crate -> target
        self.goal_nodes: Set[str] = set()

    def initialize(self, model: Any, initial_node: Any):
        """Extract goals and build dependency graph."""
        self.model = model
        self.goal_edges, self.goal_nodes = self._extract_goal_graph(model, initial_node)
        h0 = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, float(h0))

    def _extract_goal_graph(self, model: Any, node: Any) -> Tuple[Dict[str, str], Set[str]]:
        """Parse goals into crate -> surface edges (dependency graph)."""
        edges = {}
        nodes = set()

        # try to get goals from model
        goals = getattr(model, "goals", None)
        if isinstance(goals, (list, set)):
            for g in goals:
                if isinstance(g, tuple) and len(g) == 3 and g[0] == "on":
                    c, t = str(g[1]), str(g[2])
                    edges[c] = t
                    nodes.update([c, t])
        # fallback: check node.task_network
        if not edges and hasattr(node, "task_network"):
            for t in getattr(node.task_network, "subtasks", []):
                if getattr(t, "name", "").endswith("do_put_on"):
                    params = getattr(t, "parameters", [])
                    if len(params) >= 2:
                        c, t = str(params[0]), str(params[1])
                        edges[c] = t
                        nodes.update([c, t])
        return edges, nodes

    def _longest_chain(self, edges: Dict[str, str]) -> int:
        """Return the depth of the longest crate→surface chain in the goal graph."""
        memo = {}

        def depth(node: str) -> int:
            if node in memo:
                return memo[node]
            if node not in edges:  # pallet or base
                memo[node] = 0
                return 0
            parent = edges[node]
            d = 1 + depth(parent)
            memo[node] = d
            return d

        return max((depth(n) for n in edges.keys()), default=0)

    def _calculate_node_heuristic(self, node: Any) -> float:
        """Heuristic = depth of longest unresolved chain + penalties for misplaced crates."""
        # current state facts
        facts = set()
        if hasattr(node, "state") and isinstance(node.state, (set, frozenset)):
            facts = {tuple(map(str, f)) for f in node.state}

        # depth of dependency graph
        depth_cost = self._longest_chain(self.goal_edges)

        # penalty: crates not yet in place
        unsatisfied = 0
        for c, t in self.goal_edges.items():
            if ("on", c, t) not in facts:
                unsatisfied += 1

        # heuristic = depth cost + unsatisfied count (scaled)
        h_val = float(depth_cost + 0.5 * unsatisfied)
        return h_val

    def __call__(self, parent_node: Any, node: Any):
        h_value = float(self._calculate_node_heuristic(node))
        super().update_info(h_value)
        return h_value

    def __output__(self):
        return (
            f"DepotChatGPTHeuristic0(name={self.depot_gpt_heuristic_0}, "
            f"goals={len(self.goal_edges)}, depth={self._longest_chain(self.goal_edges)})"
        )
