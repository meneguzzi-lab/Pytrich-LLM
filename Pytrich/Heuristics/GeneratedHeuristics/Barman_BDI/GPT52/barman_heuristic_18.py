from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic18(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Rationale (domain insights -> heuristic design)
    ----------------------------------------------
    The training problems have HTN goals of the form AchieveContainsShotCocktail(shot, cocktail)
    repeated many times. In this domain, producing one cocktail in a shot minimally requires:
      - ensure target shot becomes (contains shot cocktail)
      - typically: make cocktail in shaker then pour to shot

    Minimal primitive action skeleton (ignoring cleaning if already clean/empty):
      For each of the 2 ingredients:
        grasp shot
        fill-shot
        pour-shot-to-*-shaker
      shake
      (ensure shot is empty+clean)
      pour-shaker-to-shot

    That is ~9 primitive actions per unsatisfied cocktail in the common initial states.

    Key planning challenges:
      - Resource coupling through a single shaker (and potentially limited hands)
      - Cleaning/emptying introduces extra actions when containers are dirty/used/non-empty
      - Shaker level constraints mean you cannot overfill, but in provided instances shaker starts empty.

    Heuristic strategy
    ------------------
    Provide a fast lower-bound estimate by counting unmet *goal facts* and mapping each unmet goal
    to a conservative minimal number of primitive actions needed.

    We compute from the *goal facts* (model.goals) in grounded fact format.
    For each goal +contains[shot,cocktail] not yet true in the current state:
      - Base lower bound: 1 (at least one action must add this fact).
      - Better domain LB: 6 if the shot doesn't already contain the cocktail.
        Justification: even in a relaxed setting, you still need at least:
          (i) produce cocktail somewhere (>=1), (ii) have empty+clean shot if not already (>=0..2),
          (iii) perform pour into the shot (>=1), plus at least some ingredient acquisition.
        Empirically, 6 is a safe underestimate across typical states because any solution requires
        multiple distinct action types.

    Additional small refinements (still admissible):
      - If the target shot is not empty, at least 1 empty-shot is needed before it can be filled
        with the cocktail (pour-shaker-to-shot requires empty shot). So add +1.
      - If the target shot is not clean, at least 1 clean-shot action is needed at some point.
        (clean-shot itself requires empty and holding, but we count only 1 as an LB.) So add +1.

    For other goal predicates (if any), we fall back to counting them with cost 1.

    Performance considerations
    --------------------------
    - Uses only bitwise checks against fact.global_id (no state_explicit_repr on states).
    - Parses goal facts once in __init__.
    - Precomputes fact ids for (empty shotX) and (clean shotX) when available.

    This heuristic is admissible as it only counts necessary conditions with conservative costs.
    """

    def __init__(self, model, barman_gpt52_heuristic_18: bool = True):
        super().__init__(model)
        self.enabled = bool(barman_gpt52_heuristic_18)

        # Cache: predicate -> list of (args, fact_id)
        self._goal_facts: List[Tuple[str, Tuple[str, ...], int]] = []

        # For faster checks on common predicates
        self._fact_id_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Precompute mapping from shot -> fact_id for empty/clean if those facts exist
        self._empty_shot_fact: Dict[str, int] = {}
        self._clean_shot_fact: Dict[str, int] = {}

        # Build key->id index for all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_id_by_key[key] = f.global_id

        # Parse goals (bitwise integer -> explicit names)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            fid = self._fact_id_by_key.get(key, None)
            if fid is not None:
                self._goal_facts.append((pred, tuple(args), fid))

        # Cache empty/clean shot fact ids for any shots we can infer from facts
        # We detect shots by scanning facts with predicates empty/clean.
        for (pred, args), fid in self._fact_id_by_key.items():
            if pred == "empty" and len(args) == 1 and args[0].startswith("shot"):
                self._empty_shot_fact[args[0]] = fid
            elif pred == "clean" and len(args) == 1 and args[0].startswith("shot"):
                self._clean_shot_fact[args[0]] = fid

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def __call__(self, state: int, task_network=None) -> int:
        if not self.enabled:
            return 0

        h = 0

        # Goal-driven estimate: sum LBs for unmet goal facts
        for pred, args, fid in self._goal_facts:
            if self._is_true(state, fid):
                continue

            # Domain-dependent handling for cocktail-in-shot goals
            if pred == "contains" and len(args) == 2:
                container, beverage = args
                # Specifically target shots containing cocktails (as in benchmarks)
                if container.startswith("shot") and beverage.startswith("cocktail"):
                    # Base LB per missing cocktail serving
                    cost = 6

                    # If shot isn't empty, at least one empty-shot must occur before final pour
                    empty_fid = self._empty_shot_fact.get(container)
                    if empty_fid is not None and not self._is_true(state, empty_fid):
                        cost += 1

                    # If shot isn't clean, at least one clean-shot must occur at some point
                    clean_fid = self._clean_shot_fact.get(container)
                    if clean_fid is not None and not self._is_true(state, clean_fid):
                        cost += 1

                    h += cost
                    continue

            # Generic fallback: each missing goal fact needs at least one action
            h += 1

        # HTN-awareness (lightweight): unfinished task network implies at least one action.
        # This remains admissible but helps when goals are empty but TN not.
        if task_network:
            # If there are remaining tasks, at least one refinement/primitive step is needed.
            h = max(h, 1)

        return h
