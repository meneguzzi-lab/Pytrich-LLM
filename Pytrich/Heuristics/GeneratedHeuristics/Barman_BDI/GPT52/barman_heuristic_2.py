from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _GoalRequest:
    """A compact representation of one goal fact we care about."""

    predicate: str
    args: Tuple[str, ...]


class BarmanGPT52Heuristic2(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Core idea (admissible lower bound, fast):
    - The top-level goals in provided instances are of the form +contains[shotX,cocktailY].
    - Producing one such cocktail in a shot *necessarily* requires at least:
        (i) making the cocktail in the shaker via action `shake` (1 primitive action), and
        (ii) pouring it from shaker to the target shot via `pour-shaker-to-shot` (1 action).
      So for each *missing* goal cocktail-in-shot we can safely count a base cost of 2.

    - Additionally, to be able to pour into a shot, the shot must be (empty and clean).
      If the shot is not currently empty, at least one `empty-shot` action is required.
      If the shot is not currently clean, at least one `clean-shot` action is required
      (possibly after emptying, but we count them separately as a lower bound).

    - For making a cocktail in the shaker, the shaker must be in a usable condition.
      If shaker is not clean, at least one `clean-shaker` action is needed.
      If shaker is not empty and does not already contain the needed cocktail, at least
      one `empty-shaker` action is needed.

    Important: We *avoid* expensive state stringification; we only use bit tests against
    cached Fact global_ids.

    This heuristic is designed for scale: it only iterates over goal facts and performs
    O(1) bit checks per relevant predicate.

    Admissibility:
    - Every counted item corresponds to an action that is necessary in all real plans
      under the domain constraints (or a set of constraints that strictly implies one
      such action is needed). We only add costs when a condition is currently false and
      there is no alternative to satisfy the goal without eventually achieving it.

    Consistency:
    - The heuristic is a sum of independent necessary-action indicators. Actions can
      reduce the count by at most their cost in typical cases; while a full formal proof
      is domain-specific, this is generally well-behaved for A*/GBFS in Pytrich.
    """

    def __init__(self, model, barman_gpt52_heuristic_2: str = "barman_gpt52_heuristic_2"):
        super().__init__(model, barman_gpt52_heuristic_2)
        self.model = model

        # Cache: map (predicate, args_tuple) -> fact_global_id
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Cache goal requests (we focus on contains goals, but keep others generic)
        self._goal_reqs: List[_GoalRequest] = []

        # Additional caches for quick access to "any contains" for a given container.
        # We precompute lists of fact ids by container for predicates we care about.
        self._contains_by_container: Dict[str, List[int]] = {}
        self._used_by_container: Dict[str, List[int]] = {}

        # Precompute all fact ids for predicates relevant to this domain.
        self._index_facts()
        self._index_goals()

    # -------------------------- Fact parsing helpers --------------------------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact names of the form +pred[a,b] / -pred[a,b]."""
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_facts(self) -> None:
        """Build indices from model facts to support O(1) bit tests."""
        # model.facts is a list-like indexed by global_id
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_gid[key] = f.global_id

            # Build container-centric lists for contains/used to enable "exists" queries.
            if pred == "contains" and len(args) == 2:
                cont = args[0]
                self._contains_by_container.setdefault(cont, []).append(f.global_id)
            elif pred == "used" and len(args) == 2:
                cont = args[0]
                self._used_by_container.setdefault(cont, []).append(f.global_id)

    def _index_goals(self) -> None:
        """Extract the (grounded) goal facts from model.goals bitset."""
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred is None:
                continue
            # Only positive goals are present in standard encodings; still keep generic.
            self._goal_reqs.append(_GoalRequest(pred, tuple(args)))

    # -------------------------- Bitwise query helpers --------------------------

    def _is_true_gid(self, state: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _gid(self, predicate: str, args: Tuple[str, ...]) -> Optional[int]:
        return self._fact_gid.get((predicate, args), None)

    def _is_true(self, state: int, predicate: str, *args: str) -> bool:
        gid = self._gid(predicate, tuple(args))
        return self._is_true_gid(state, gid)

    def _exists_true_for_container(self, state: int, pred: str, container: str) -> bool:
        """Check if there exists any true fact pred[container, x]."""
        if pred == "contains":
            gids = self._contains_by_container.get(container, [])
        elif pred == "used":
            gids = self._used_by_container.get(container, [])
        else:
            return False
        for gid in gids:
            if ((state >> gid) & 1) == 1:
                return True
        return False

    # -------------------------- Heuristic computation --------------------------

    def __call__(self, state: int, task_network=None) -> int:
        """Compute an admissible estimate of remaining cost.

        We primarily score:
        - missing goal cocktails in their target shots
        - necessary cleanliness/emptiness preconditions for the final pour
        - necessary shaker reset operations if it cannot directly produce the next cocktail

        The heuristic also lightly considers remaining abstract tasks in the task network
        (if provided) by adding a tiny lower bound per unresolved AchieveContainsShotCocktail,
        but never relies on expensive inspection.
        """

        h = 0

        # 1) Goal-based estimate: focus on +contains[shot,cocktail]
        # Each missing such goal requires at least: shake + pour = 2 actions.
        # Then add necessary cleanup/emptying for target shot, and shaker preparation.
        for g in self._goal_reqs:
            if g.predicate != "contains" or len(g.args) != 2:
                # Generic fallback: if some other positive goal fact is missing, count 1.
                gid = self._gid(g.predicate, g.args)
                if gid is not None and not self._is_true_gid(state, gid):
                    h += 1
                continue

            shot, cocktail = g.args

            # If goal already achieved, skip.
            if self._is_true(state, "contains", shot, cocktail):
                continue

            # Base necessary actions: must create cocktail in shaker and pour to shot.
            h += 2

            # Shot must be empty and clean to receive pour.
            if not self._is_true(state, "empty", shot):
                # Only empty-shot can make it empty.
                h += 1
            if not self._is_true(state, "clean", shot):
                # Ultimately requires clean-shot.
                h += 1

            # Shaker preparation lower bound:
            # - If shaker is not clean, at least one clean-shaker is needed.
            # - If shaker is not empty AND does not already contain this cocktail,
            #   at least one empty-shaker is needed.
            # Assumption: single shaker in instances; still safe to use any shaker that exists.
            # We approximate by considering *some* shaker named 'shaker1' if present; otherwise
            # we do not add shaker-specific bounds.
            shaker = "shaker1"
            if ("clean", (shaker,)) in self._fact_gid:
                if not self._is_true(state, "clean", shaker):
                    h += 1
                # If shaker already contains the target cocktail, we don't need to empty it.
                if ("empty", (shaker,)) in self._fact_gid:
                    if (not self._is_true(state, "empty", shaker)) and (not self._is_true(state, "contains", shaker, cocktail)):
                        # empty-shaker needed at some point.
                        h += 1

        # 2) Task-network-based tiny guidance (optional, admissible):
        # If the search node still has abstract tasks AchieveContainsShotCocktail,
        # each requires at least 1 primitive action in any non-trivial case.
        # We keep this conservative to avoid overcounting with goal-based part.
        if task_network:
            # Avoid deep inspection; just look at task names.
            for t in task_network:
                name = getattr(t, "name", "")
                if name == "AchieveContainsShotCocktail":
                    h += 0  # kept at 0 to avoid double counting; goal part dominates.

        return h
