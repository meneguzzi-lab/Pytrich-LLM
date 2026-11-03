"""
Depot domain heuristic for Pytrich.

Class name: DepotChatGPTHeuristic1
__init__ parameter (required name): depot_gpt_heuristic_1

This heuristic:
 - precomputes simple statistics for domain decompositions (load/unload/drive/lift/drop counts)
 - scores methods with small penalties (to keep heuristic admissible-ish)
 - estimates unsatisfied goals when possible (bitmask or set)
 - always returns a numeric float from __call__
"""

import time
from typing import Optional, Any

from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.model import Model


class DepotChatGPTHeuristic1(Heuristic):
    def __init__(
        self,
        depot_gpt_heuristic_1: str = "depot_gpt_heuristic_1",
        transport_penalty: float = 0.12,
        drive_penalty: float = 0.08,
        lift_penalty: float = 0.04,
    ):
        # required parameter naming (document-specified)
        super().__init__(name=depot_gpt_heuristic_1)
        self.depot_gpt_heuristic_1 = depot_gpt_heuristic_1

        # tunable small penalties (kept small for admissibility)
        self.transport_penalty = float(transport_penalty)
        self.drive_penalty = float(drive_penalty)
        self.lift_penalty = float(lift_penalty)

        # internal bookkeeping
        self.method_scores = {}
        self.preprocessing_time = 0.0
        self.model: Optional[Model] = None

    def initialize(self, model: Model, initial_node: Any):
        """
        Called by planner to initialize heuristic with the model and initial node.
        Precompute method features and compute an initial heuristic value.
        """
        start = time.time()
        self.model = model
        self._precompute_method_info()
        self.preprocessing_time = time.time() - start

        # compute initial heuristic (safe numeric)
        h0 = self._calculate_node_heuristic(initial_node)
        return super().initialize(model, h0)

    def _precompute_method_info(self):
        """
        Walk model decompositions / abstract tasks and collect cheap statistics
        (counts of load/unload/drive/lift/drop and subtask count).
        This speeds scoring during search.
        """
        self.method_scores.clear()
        if not self.model:
            return

        methods = []
        # try several model attributes (the model can expose different names)
        if hasattr(self.model, "methods"):
            methods = self.model.methods
        elif hasattr(self.model, "decompositions"):
            methods = self.model.decompositions
        elif hasattr(self.model, "abstract_tasks"):
            # collect decompositions from abstract tasks
            for at in getattr(self.model, "abstract_tasks", []):
                if hasattr(at, "decompositions"):
                    methods.extend(at.decompositions)

        if not methods:
            return

        for method in methods:
            method_name = method.name if hasattr(method, "name") else str(method)
            load_count = unload_count = drive_count = lift_count = drop_count = 0
            subtask_count = 0

            subtasks = []
            if hasattr(method, "subtasks"):
                subtasks = getattr(method, "subtasks") or []
            elif hasattr(method, "task_network"):
                subtasks = getattr(method, "task_network") or []
            elif hasattr(method, "decomposition"):
                subtasks = getattr(method, "decomposition") or []

            subtask_count = len(subtasks) if subtasks is not None else 0

            # try to infer operations from subtask names (robust to representation differences)
            for st in subtasks or []:
                sname = ""
                if hasattr(st, "name"):
                    sname = st.name
                elif hasattr(st, "task"):
                    sname = str(st.task)
                else:
                    sname = str(st)
                s = sname.lower()
                if "load" in s and "truck" in s:
                    load_count += 1
                if "unload" in s and "truck" in s:
                    unload_count += 1
                if "drive" in s or "move" in s:
                    drive_count += 1
                if "lift" in s or "lift-crate" in s:
                    lift_count += 1
                if "drop" in s:
                    drop_count += 1

            self.method_scores[method_name] = {
                "load_count": load_count,
                "unload_count": unload_count,
                "drive_count": drive_count,
                "lift_count": lift_count,
                "drop_count": drop_count,
                "subtask_count": subtask_count,
                # heuristics about method: goal-achieving if name references put_on / load / unload
                "is_goal_achieving": any(tok in method_name.lower() for tok in ["do_put_on", "do_load_truck", "do_unload_truck", "drop", "load", "unload"]),
                "raw_name": method_name,
            }

    def _score_method(self, method, node) -> float:
        """
        Produce a small numeric cost for a method.
        Lower = more promising (we return small numbers for admissibility).
        """
        method_name = method.name if hasattr(method, "name") else str(method)
        info = self.method_scores.get(method_name, None)

        score = 0.0
        if info:
            score += info["load_count"] * 0.03
            score += info["unload_count"] * 0.03
            score += info["drive_count"] * self.drive_penalty
            score += info["lift_count"] * self.lift_penalty
            score += info["drop_count"] * 0.02
            # small bonus (negative) for methods that directly achieve goals
            if info["is_goal_achieving"]:
                score -= 0.04
            # complexity penalty for many subtasks
            score += info["subtask_count"] * 0.01

        # name-pattern tweaks (robust fallback if precompute didn't find method)
        mlow = method_name.lower()
        if "do_put_on" in mlow:
            score += 0.02
        if "do_load_truck" in mlow or "load" in mlow:
            score += 0.02
        if "do_unload_truck" in mlow or "unload" in mlow:
            score += 0.02

        return max(score, 0.0)

    def _count_unsatisfied_goals(self, node) -> int:
        """
        Best-effort count of unsatisfied goals.
        Handles several node representations (bitmask int, set of facts, or fallbacks).
        """
        if not self.model:
            return 0

        # Attempt to read state bitmask or state set from node
        state = None
        if hasattr(node, "state"):
            state = getattr(node, "state")
        elif hasattr(node, "state_bitwise"):
            state = getattr(node, "state_bitwise")
        elif hasattr(node, "state_set"):
            state = getattr(node, "state_set")
        elif hasattr(node, "s"):  # some nodes use .s
            state = getattr(node, "s")

        # If we have integer bitmask representation (common in Pytrich examples)
        try:
            if isinstance(state, int):
                # model.goals might be int mask or a set
                goals = getattr(self.model, "goals", None)
                if isinstance(goals, int):
                    rem = goals & ~state
                    return bin(rem).count("1")
                else:
                    # convert state's explicit names to a set and compare with goal names (fallback)
                    try:
                        current_names = set(self.model.state_explicit_repr(state))
                        if isinstance(goals, (set, list)):
                            return len(set(goals) - current_names)
                    except Exception:
                        pass
        except Exception:
            pass

        # If the node state is a set of facts
        if isinstance(state, (set, frozenset)):
            goals = getattr(self.model, "goals", None)
            if isinstance(goals, (set, frozenset)):
                return len(set(goals) - set(state))

        # If we can't inspect the state reliably, attempt to infer from node.task(s)
        # Fallback: give a conservative non-zero estimate if there's remaining TN or method
        remaining_guess = 0
        if hasattr(node, "task_network") and getattr(node, "task_network"):
            remaining_guess = len(getattr(node, "task_network"))
        elif hasattr(node, "method") and getattr(node, "method"):
            remaining_guess = 1
        else:
            # final fallback: assume moderate remaining work if model declares goals
            try:
                goals = getattr(self.model, "goals", None)
                if isinstance(goals, (set, list, frozenset)):
                    remaining_guess = max(0, len(goals) // 2)
                elif isinstance(goals, int):
                    remaining_guess = max(0, bin(goals).count("1") // 2)
            except Exception:
                remaining_guess = 0

        return int(remaining_guess)

    def _calculate_node_heuristic(self, node) -> float:
        """
        Combines a method-based small cost and a conservative unsatisfied-goals estimate.
        Returns a numeric float (kept small for admissibility).
        """
        # base method score
        method_score = 0.0
        if hasattr(node, "method") and getattr(node, "method"):
            method_score = float(self._score_method(node.method, node))

        # unsatisfied goals (best-effort)
        unsat = self._count_unsatisfied_goals(node)

        # combine: unsatisfied goals get a slightly larger influence,
        # but scaled down by a normalizer so final numbers remain small.
        # normalizer: maximum possible goals (best-effort)
        max_goals = 1
        try:
            goals = getattr(self.model, "goals", None)
            if isinstance(goals, (set, list, frozenset)):
                max_goals = max(1, len(goals))
            elif isinstance(goals, int):
                max_goals = max(1, bin(goals).count("1"))
        except Exception:
            max_goals = max(1, unsat)

        # small per-goal penalty, scaled to keep h <= ~1
        per_goal_penalty = 0.25 / max_goals  # tuned to give reasonable guidance but remain small
        h_value = method_score + unsat * per_goal_penalty

        # final safe clamp
        if not isinstance(h_value, float):
            try:
                h_value = float(h_value)
            except Exception:
                h_value = 0.0

        return max(0.0, min(h_value, 1.0))  # keep in [0,1]

    def __call__(self, parent_node, node):
        """
        Called by the planner for node expansion. Must return numeric float.
        """
        h_value = self._calculate_node_heuristic(node)
        # keep values small for admissibility; examples cap at 1.0
        h_value = min(float(h_value), 1.0)
        # update heuristic internal statistics / logging in the base class
        super().update_info(h_value)
        return h_value

    def __repr__(self):
        return f"DepotChatGPTHeuristic1(name={self.depot_gpt_heuristic_1})"

    def __str__(self):
        return self.__repr__()

    def __output__(self):
        return (
            f"Depot Heuristic info:\n"
            f"\tName: {self.depot_gpt_heuristic_1}\n"
            f"\tMethods analyzed: {len(self.method_scores)}\n"
            f"\tTransport penalty: {self.transport_penalty}\n"
            f"\tDrive penalty: {self.drive_penalty}\n"
            f"\tLift penalty: {self.lift_penalty}\n"
            f"\tPreprocessing time: {self.preprocessing_time:.3f}s\n"
        )
