from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Pytrich import (the framework provides Heuristic base class)
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback for alternative package layouts
    from Pytrich.Heuristics.Heuristic import Heuristic  # type: ignore


@dataclass
class _GoalCocktail:
    """Internal representation of a top-level goal of the form contains(shot, cocktail)."""

    shot: str
    cocktail: str


class BarmanGPT52Heuristic8(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Core domain insight
    -------------------
    In this benchmark, goals are almost always of the form:
        AchieveContainsShotCocktail(shotX, cocktailY)
    and the unique constructive way (when the shot doesn't already contain the cocktail)
    is:
        - make cocktail in shaker: requires 2 ingredients and a shake
        - ensure target shot is empty & clean
        - hold shaker
        - pour shaker -> shot

    A useful *lower bound* on remaining effort is therefore:
        For each missing goal cocktail-in-shot:
            at least 1 pour-shaker-to-shot action
            plus (if cocktail not already available in shaker) at least 1 shake action
            plus (for each missing ingredient in shaker) at least 1 fill-shot action
              (because every ingredient must be dispensed into some shot first)

    We intentionally ignore hand logistics, grasp/drop, and cleaning/emptying steps
    as they are domain-dependent overhead that can be avoided in some states and
    would break admissibility if over-counted. Counting only *necessary* constructive
    actions keeps the heuristic admissible and cheap.

    Additional state-dependent refinement:
    - If some cocktail already exists in the shaker (contains(shaker1, cocktailY)),
      then we do not charge for shaking/ingredients for that goal.
    - If the shaker already contains required ingredients, we do not charge for them.

    Computational design
    --------------------
    - Uses ONLY bitwise checks on the state (fast).
    - Precomputes fact global_ids for patterns needed by the heuristic.
    - Parses goal facts from model.goals via model.state_explicit_repr (allowed).

    This heuristic is not a full delete-relaxation; it is a domain-specific relaxed
    "recipe" lower bound. It scales well to large instances because each evaluation
    is O(#goal_cocktails + small constant) with minimal dictionary lookups.
    """

    def __init__(self, model, barman_gpt52_heuristic_8: str = "barman_gpt52_heuristic_8"):
        super().__init__(model, barman_gpt52_heuristic_8)
        self.model = model

        # --- Precompute: parse goals into (shot, cocktail) pairs
        self.goal_cocktails: List[_GoalCocktail] = []
        for fname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(fname)
            if pred == "contains" and len(args) == 2:
                shot, bev = args
                # We only treat cocktail goals (domain instances do that).
                self.goal_cocktails.append(_GoalCocktail(shot=shot, cocktail=bev))

        # --- Precompute: cocktail -> (ing1, ing2)
        self.cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # We'll extract from all facts (static in the domain instances)
        # Find cocktailPart1/2 facts: +cocktailPart1[cocktail,ingredient]
        part1: Dict[str, str] = {}
        part2: Dict[str, str] = {}
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "cocktailPart1" and len(args) == 2:
                c, i = args
                part1[c] = i
            elif pred == "cocktailPart2" and len(args) == 2:
                c, i = args
                part2[c] = i
        for c in set(list(part1.keys()) + list(part2.keys())):
            self.cocktail_parts[c] = (part1.get(c), part2.get(c))

        # --- Identify the (unique) shaker object(s) we may have
        # Heuristic assumes typical instances with a single shaker, but supports many.
        self.shakers: List[str] = []
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            # There is no explicit type predicate; infer from object name pattern in facts:
            # shaker appears in predicates clean/empty/contains/shakerLevel etc.
            if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and args:
                s = args[0]
                if s not in self.shakers:
                    self.shakers.append(s)
        # Fallback: common name
        if not self.shakers:
            self.shakers = ["shaker1"]

        # --- Precompute global_id for key contains facts used in bitwise state checks
        # Maps:
        #   contains(container, beverage) -> global_id
        self.contains_gid: Dict[Tuple[str, str], int] = {}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "contains" and len(args) == 2 and fact.name.startswith("+"):
                self.contains_gid[(args[0], args[1])] = fact.global_id

        # Convenience: for each cocktail, store gids for shaker contains cocktail and ingredients
        self.shaker_contains_cocktail_gid: Dict[Tuple[str, str], Optional[int]] = {}
        self.shaker_contains_ing_gid: Dict[Tuple[str, str], Optional[int]] = {}
        for s in self.shakers:
            for c in self.cocktail_parts.keys():
                self.shaker_contains_cocktail_gid[(s, c)] = self.contains_gid.get((s, c))
            # ingredients might be many; we compute lazily via contains_gid

    # ------------ Required helper ------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    # ------------ Heuristic interface ------------
    def __call__(self, state: int, task_network=None) -> int:
        """Return an admissible estimate of remaining primitive actions."""
        # If no relevant goal facts were detected, fall back to 0 (safe).
        if not self.goal_cocktails:
            return 0

        # Pick a shaker that best supports already-made cocktails (min-cost optimistic choice).
        # Since there is usually exactly one shaker, this is essentially constant.
        best_total = None
        for shaker in self.shakers:
            total = 0
            for g in self.goal_cocktails:
                total += self._estimate_for_goal_with_shaker(state, shaker, g)
            if best_total is None or total < best_total:
                best_total = total

        return int(best_total or 0)

    def _estimate_for_goal_with_shaker(self, state: int, shaker: str, goal: _GoalCocktail) -> int:
        """Lower bound for achieving contains(goal.shot, goal.cocktail) using a given shaker.

        Counts only unavoidable constructive actions:
        - if shot already contains cocktail => 0
        - else must perform pour-shaker-to-shot at least once => +1
          and if shaker doesn't already contain cocktail => must produce it:
            - shake at least once => +1
            - and each missing ingredient in shaker requires at least one fill-shot (dispense) => +1 per missing ingredient

        Note: We do NOT count pour-shot-to-shaker, grasp/drop, empty/clean, because those
        may already be satisfied or can sometimes be avoided; counting them risks overestimation.
        """
        # Goal already satisfied?
        gid_goal = self.contains_gid.get((goal.shot, goal.cocktail))
        if self._is_true(state, gid_goal):
            return 0

        # At least one pour from shaker to shot is necessary.
        cost = 1

        # If cocktail already in shaker, we can (optimistically) just pour.
        gid_shaker_has_cocktail = self.contains_gid.get((shaker, goal.cocktail))
        if self._is_true(state, gid_shaker_has_cocktail):
            return cost

        # Otherwise: must create cocktail in shaker.
        cost += 1  # at least one shake

        ing1, ing2 = self.cocktail_parts.get(goal.cocktail, (None, None))
        # If missing recipe info, keep admissible but weak: assume need 2 ingredients.
        if ing1 is None or ing2 is None:
            return cost + 2

        gid_ing1 = self.contains_gid.get((shaker, ing1))
        gid_ing2 = self.contains_gid.get((shaker, ing2))

        # Each ingredient not already in shaker must be dispensed into a shot at least once.
        # (fill-shot is the only way to obtain contains(shot, ingredient) in the domain)
        if not self._is_true(state, gid_ing1):
            cost += 1
        if not self._is_true(state, gid_ing2):
            cost += 1

        return cost
