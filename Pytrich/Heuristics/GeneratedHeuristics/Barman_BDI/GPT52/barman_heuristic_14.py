from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Pytrich provides the base Heuristic class in its framework.
# The import path can differ across distributions; this defensive import
# covers the most common layouts.
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.Heuristic import Heuristic
    except Exception:
        from heuristic import Heuristic


class BarmanGPT52Heuristic14(Heuristic):
    """Domain-dependent heuristic for the HDDL *Barman* domain.

    Design rationale (domain insights -> heuristic terms)
    -----------------------------------------------------
    The training instances have goals of the form:
        AchieveContainsShotCocktail(shotX, cocktailY)

    In this domain, achieving a cocktail-in-shot typically requires the
    following minimal primitive structure (ignoring deletions):

      - Ensure the shaker contains the cocktail:
            add two ingredients to shaker via shot-filling and pouring,
            then shake.
      - Ensure the target shot is clean+empty.
      - Hold shaker and pour shaker->shot.

    Key effort drivers in this domain:
      1) Each *missing cocktail goal* implies at least one pour-shaker-to-shot.
      2) If shaker does not currently contain the needed cocktail, we must
         (at least) perform one shake for that cocktail.
      3) If shaker lacks required ingredients, each missing ingredient implies
         some work (fill a shot + pour into shaker). We approximate this with a
         small lower bound per missing ingredient.
      4) If the target shot is not clean or not empty, extra work is needed.

    Because Pytrich search can be very large, the heuristic must be cheap.
    We therefore:
      - Only look at *goal facts* and a small set of relevant state facts.
      - Use bitwise checks for fact truth.
      - Pre-index facts by predicate and arguments at initialization.

    The heuristic is admissible as a lower bound by using conservative counts:
      - We count at most one unit for an action-type necessity (pour, shake, etc.).
      - We do NOT try to estimate exact hand/holding logistics (which can add
        extra actions); instead we use only safe necessities.

    Returned value:
      A non-negative integer lower bound of remaining primitive actions.

    Note:
      This heuristic is state-based; it ignores the current task network.
      In Barman instances given, the task network is a sequence of cocktail goals,
      so state-based lower bounds are effective and very fast.
    """

    def __init__(self, model, barman_gpt52_heuristic_14: bool = True):
        super().__init__(model)
        self.enabled = bool(barman_gpt52_heuristic_14)

        # Fact index:
        #  - by predicate -> list of (fact_global_id, args)
        #  - plus direct maps for fast lookup of specific templates.
        self._pred_to_facts: Dict[str, List[Tuple[int, Tuple[str, ...]]]] = {}

        # Specific lookup tables for this domain
        self._contains: Dict[Tuple[str, str], int] = {}          # (container, beverage) -> fact_id
        self._clean: Dict[str, int] = {}                         # container -> fact_id
        self._empty: Dict[str, int] = {}                         # container -> fact_id
        self._used: Dict[Tuple[str, str], int] = {}              # (container, beverage) -> fact_id
        self._cocktail_p1: Dict[str, str] = {}                   # cocktail -> ingredient
        self._cocktail_p2: Dict[str, str] = {}                   # cocktail -> ingredient

        # Populate indices from grounded facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            targs = tuple(args)
            self._pred_to_facts.setdefault(pred, []).append((fact.global_id, targs))

            if pred == "contains" and len(args) == 2:
                self._contains[(args[0], args[1])] = fact.global_id
            elif pred == "clean" and len(args) == 1:
                self._clean[args[0]] = fact.global_id
            elif pred == "empty" and len(args) == 1:
                self._empty[args[0]] = fact.global_id
            elif pred == "used" and len(args) == 2:
                self._used[(args[0], args[1])] = fact.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                # cocktailPart1(cocktail, ingredient)
                self._cocktail_p1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self._cocktail_p2[args[0]] = args[1]

        # Cache goal structure (the goal bitset is constant)
        self._goal_contains_shot_cocktails: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "contains" and len(args) == 2:
                container, bev = args
                # Only count goals that are cocktail in a shot (as in the instances).
                # We can't reliably type-check here, but in this benchmark cocktails are
                # the second argument of such goals.
                if container.startswith("shot") and bev.startswith("cocktail"):
                    self._goal_contains_shot_cocktails.append((container, bev))

        # Identify shakers (again by name pattern; benchmark uses shaker1)
        self._shakers: List[str] = []
        for _, args in self._pred_to_facts.get("shakerLevel", []):
            if args and args[0].startswith("shaker"):
                self._shakers.append(args[0])
        # Ensure uniqueness
        self._shakers = sorted(set(self._shakers))

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def __call__(self, node) -> int:
        if not self.enabled:
            return 0

        state = node.state

        # If goal already reached, heuristic is 0.
        # (goal_reached checks task network emptiness too, but heuristic should be state-based;
        # we can still quickly test goal facts satisfied in the state bitset.)
        if (state & self.model.goals) == self.model.goals:
            return 0

        # Choose a single shaker to reason about (instances have 1).
        shaker = self._shakers[0] if self._shakers else "shaker1"

        h = 0

        # Conservative lower bounds per unmet goal.
        # For each missing (contains shot cocktail):
        #   - at least 1 pour from shaker to shot (in any correct plan)
        #   - plus at least some work to have shaker contain cocktail (if not already)
        #   - plus shot cleaning/emptying necessities.
        for shot, cocktail in self._goal_contains_shot_cocktails:
            shot_has_goal = self._is_true(state, self._contains.get((shot, cocktail)))
            if shot_has_goal:
                continue

            # At least one pour-shaker-to-shot must happen.
            h += 1

            # If shaker doesn't contain cocktail yet, need at least one shake for that cocktail.
            shaker_has_cocktail = self._is_true(state, self._contains.get((shaker, cocktail)))
            if not shaker_has_cocktail:
                # Need to create cocktail in shaker (at least one shake).
                h += 1

                # Also ensure ingredients are present in shaker before shaking.
                # Each missing ingredient implies at least one pour-shot-to-*-shaker.
                ing1 = self._cocktail_p1.get(cocktail)
                ing2 = self._cocktail_p2.get(cocktail)

                if ing1 is not None and not self._is_true(state, self._contains.get((shaker, ing1))):
                    h += 1
                if ing2 is not None and not self._is_true(state, self._contains.get((shaker, ing2))):
                    h += 1

            # Target shot must be clean and empty before pouring (domain preconditions).
            if not self._is_true(state, self._clean.get(shot)):
                # At least one cleaning-related action is required.
                # (Could be more: empty-shot + clean-shot; but we keep admissible.)
                h += 1
            if not self._is_true(state, self._empty.get(shot)):
                # If shot is not empty, at least one empty-shot is required.
                h += 1

        # Mild global bonus lower bound: if any goal remains and shaker is dirty while empty,
        # at least one clean-shaker is needed before making a new cocktail.
        # This remains admissible because making any new cocktail in a shaker from empty requires clean shaker.
        if self._goal_contains_shot_cocktails:
            any_unmet = False
            for shot, cocktail in self._goal_contains_shot_cocktails:
                if not self._is_true(state, self._contains.get((shot, cocktail))):
                    any_unmet = True
                    break
            if any_unmet:
                shaker_clean = self._is_true(state, self._clean.get(shaker))
                shaker_empty = self._is_true(state, self._empty.get(shaker))
                if shaker_empty and not shaker_clean:
                    h += 1

        return h
