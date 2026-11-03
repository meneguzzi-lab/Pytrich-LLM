"""
DepotChatGPTHeuristic3: Forest Distance Heuristic

Idea:
 - Represent goals as a forest of stacks (trees).
 - Compare current stacks to goal stacks.
 - Cost = edit distance between current and goal forest, with penalties for moves/drives.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from Pytrich.Heuristics.heuristic import Heuristic


class DepotChatGPTHeuristic3(Heuristic):
    def __init__(self, depot_gpt_heuristic_3: str = "depot_gpt_heuristic_3",
                 local_misplace: float = 1.0,
                 wrong_place: float = 3.0,
                 subtree_penalty: float = 0.6,
                 debug: bool = False):
        super().__init__(name=depot_gpt_heuristic_3)
        self.depot_gpt_heuristic_3 = depot_gpt_heuristic_3

        # weights
        self.local_misplace = local_misplace
        self.wrong_place = wrong_place
        self.subtree_penalty = subtree_penalty

        self.model = None
        self.goals: List[Tuple[str, str]] = []
        self.goal_parents: Dict[str, str] = {}   # crate -> target support
        self.goal_children: Dict[str, List[str]] = {}
        self.debug = debug

    # ------------------------------
    def initialize(self, model: Any, initial_node: Any):
        self.model = model
        self.goals = self._extract_goals(model)
        self._build_goal_graph()
        h0 = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, float(h0))

    def _extract_goals(self, model: Any) -> List[Tuple[str, str]]:
        goals = []
        mg = getattr(model, "goals", None)
        if isinstance(mg, (list, set, tuple)):
            for g in mg:
                if isinstance(g, tuple) and len(g) >= 3 and g[0] == "on":
                    goals.append((str(g[1]), str(g[2])))
                elif isinstance(g, str) and g.strip().startswith("(on "):
                    toks = g.strip()[1:-1].split()
                    if len(toks) >= 3:
                        goals.append((toks[1], toks[2]))
        return goals

    def _build_goal_graph(self):
        self.goal_parents.clear()
        self.goal_children.clear()
        for (c, s) in self.goals:
            self.goal_parents[c] = s
            self.goal_children.setdefault(s, []).append(c)

    # ------------------------------
    def _state_to_facts(self, node: Any) -> Set[Tuple[str, ...]]:
        facts = set()
        st = getattr(node, "state", None) or getattr(node, "state_set", None) or getattr(node, "facts", None)
        if isinstance(st, (set, frozenset)):
            for f in st:
                if isinstance(f, tuple):
                    facts.add(tuple(map(str, f)))
                elif isinstance(f, str) and f.startswith("("):
                    toks = f[1:-1].split()
                    facts.add(tuple(toks))
        elif isinstance(st, int) and self.model and hasattr(self.model, "state_explicit_repr"):
            for f in self.model.state_explicit_repr(st):
                if isinstance(f, tuple):
                    facts.add(tuple(map(str, f)))
        return facts

    def _build_current_graph(self, facts: Set[Tuple[str, ...]]):
        parents = {}
        at_map = {}
        for f in facts:
            if not f:
                continue
            if f[0] == "on" and len(f) >= 3:
                parents[f[1]] = f[2]
            elif f[0] == "at" and len(f) >= 3:
                at_map[f[1]] = f[2]
        return parents, at_map

    # ------------------------------
    def _calculate_node_heuristic(self, node: Any) -> float:
        facts = self._state_to_facts(node)
        cur_on, at_map = self._build_current_graph(facts)

        total_cost = 0.0
        seen = set()

        for (crate, target) in self.goals:
            if crate in seen:
                continue
            seen.add(crate)

            # already in place?
            if ("on", crate, target) in facts:
                continue

            # check current support
            cur_support = cur_on.get(crate)
            if cur_support is None:
                # floating or in truck -> wrong place
                total_cost += self.wrong_place
            elif cur_support == target:
                # correct parent but maybe wrong location
                src_loc = at_map.get(crate)
                tgt_loc = at_map.get(target)
                if src_loc and tgt_loc and src_loc != tgt_loc:
                    total_cost += self.wrong_place
            else:
                # wrong parent
                total_cost += self.local_misplace

            # add subtree penalty if crate has descendants in goal
            if crate in self.goal_children:
                total_cost += len(self.goal_children[crate]) * self.subtree_penalty

        return max(0.0, float(total_cost))

    def __call__(self, parent_node: Any, node: Any):
        h_value = float(self._calculate_node_heuristic(node))
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotChatGPTHeuristic3(name={self.depot_gpt_heuristic_3})"
