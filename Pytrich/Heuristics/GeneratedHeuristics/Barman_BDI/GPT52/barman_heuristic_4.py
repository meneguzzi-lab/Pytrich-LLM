from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable, Set

# Pytrich base class (available in the evaluation environment)
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic4(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain.

    Design rationale (domain insights -> heuristic features)
    -------------------------------------------------------
    The training instances have one shaker, many shots, two hands, and goals of the form:
        AchieveContainsShotCocktail(shotX, cocktailY)

    In this domain, producing one cocktail in a shot typically requires (if not already satisfied):
      - ensure shaker contains cocktail (requires: shaker clean, add 2 ingredients via shots, then shake)
      - ensure target shot is clean & empty
      - hold shaker, then pour shaker->shot

    Key challenges / bottlenecks:
      1) Reusing a single shaker across many cocktails creates repeated cycles: fill ingredients, shake, pour,
         and sometimes empty/clean if it became dirty.
      2) Shots are also resources: they become dirty (not clean) after fill/pour and must be cleaned before reuse.
      3) Hands constraints cause extra grasp/drop steps, but for admissibility we only count a *minimal* lower bound.

    Heuristic strategy
    ------------------
    We compute a fast *admissible* lower bound using a relaxed, domain-aware action count:

      - For each unsatisfied goal fact +contains[shot,cocktail], add at least 1 (must perform pour-shaker-to-shot).
      - If the cocktail is not currently in any shaker, add at least 1 (must perform shake at some point).
      - If the cocktail is not currently in the target shot, and its two ingredients are not already both in the shaker,
        add minimal cost to "get ingredients into shaker".
          In this domain, getting an ingredient into the shaker requires first getting it into a shot (fill-shot) and then
          pouring shot->shaker; therefore at least 2 primitive actions per missing ingredient.

    We additionally account for necessary "enablement" facts when they are currently false, but in a way that remains
    a lower bound:
      - If a fill-shot is needed for some ingredient, the shot used must be (clean and empty). If not, at least 1 action
        is needed to empty it (if it contains something), and at least 1 to clean it (if used & empty). We approximate
        with 0/1/2 based on current state of a *best* candidate shot. (Lower bound: optimistic choice of best shot.)
      - Shaker cleanliness: if we will need to pour ingredients into an empty shaker, it must be clean; if it isn't,
        at least 1 clean-shaker action is needed (optimistically assuming it's already empty).

    The heuristic remains admissible because:
      - We only count actions that are logically required in any real plan (pour for goal; shake if cocktail not present;
        two-step ingredient transfer for missing ingredients).
      - For enabling conditions (clean/empty), we take the minimum over available resources (best shot), and we only
        add costs when those conditions are definitely required by counted steps.
      - We ignore hand-grasp/drop and ordering interactions, which can only *increase* true cost.

    Efficiency
    ----------
    - Uses bitwise checks for facts (no state_explicit_repr on current state).
    - Pre-indexes fact global_ids for relevant predicates and goal facts.
    - Parses goal facts once.

    Notes
    -----
    - The heuristic is tailored to the typical benchmark structure (one shaker, many shots, cocktail goals).
    - It should still behave robustly on partial/intermediate states during search.
    """

    def __init__(self, model, barman_gpt52_heuristic_4: str = "barman_gpt52_heuristic_4"):
        super().__init__(model, barman_gpt52_heuristic_4)
        self.model = model

        # --- Fact indexing ---
        # Map: (pred, tuple(args)) -> global_id
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Useful per-predicate indices
        self._contains_gids: Dict[Tuple[str, str], int] = {}   # (container, beverage) -> gid
        self._clean_gid: Dict[str, int] = {}                   # container -> gid
        self._empty_gid: Dict[str, int] = {}                   # container -> gid
        self._used_gids: Dict[Tuple[str, str], int] = {}       # (container, beverage) -> gid

        # Identify objects by type (from fact signatures, best-effort)
        self._shots: Set[str] = set()
        self._shakers: Set[str] = set()
        self._cocktails: Set[str] = set()
        self._ingredients: Set[str] = set()

        # cocktail -> (ing1, ing2)
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Goal facts of form contains(shot,cocktail)
        self._goal_contains: List[Tuple[str, str, int]] = []  # (shot, cocktail, gid)

        # Build indices from all grounded facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_gid[key] = fact.global_id

            if pred == "contains" and len(args) == 2:
                c, b = args
                self._contains_gids[(c, b)] = fact.global_id
                # containers/beverages type hints
                if c.startswith("shot"):
                    self._shots.add(c)
                if c.startswith("shaker"):
                    self._shakers.add(c)
                if b.startswith("cocktail"):
                    self._cocktails.add(b)
                if b.startswith("ingredient"):
                    self._ingredients.add(b)

            elif pred == "clean" and len(args) == 1:
                cont = args[0]
                self._clean_gid[cont] = fact.global_id
                if cont.startswith("shot"):
                    self._shots.add(cont)
                if cont.startswith("shaker"):
                    self._shakers.add(cont)

            elif pred == "empty" and len(args) == 1:
                cont = args[0]
                self._empty_gid[cont] = fact.global_id
                if cont.startswith("shot"):
                    self._shots.add(cont)
                if cont.startswith("shaker"):
                    self._shakers.add(cont)

            elif pred == "used" and len(args) == 2:
                cont, bev = args
                self._used_gids[(cont, bev)] = fact.global_id
                if cont.startswith("shot"):
                    self._shots.add(cont)
                if cont.startswith("shaker"):
                    self._shakers.add(cont)

            elif pred == "cocktailPart1" and len(args) == 2:
                cock, ing = args
                a, b = self._cocktail_parts.get(cock, (None, None))
                self._cocktail_parts[cock] = (ing, b)
                self._cocktails.add(cock)
                self._ingredients.add(ing)

            elif pred == "cocktailPart2" and len(args) == 2:
                cock, ing = args
                a, b = self._cocktail_parts.get(cock, (None, None))
                self._cocktail_parts[cock] = (a, ing)
                self._cocktails.add(cock)
                self._ingredients.add(ing)

        # Parse goal state facts once (explicit repr of goals is allowed/fast enough, done once)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for nm in goal_fact_names:
            pred, args = self._parse_fact_name(nm)
            if pred == "contains" and len(args) == 2:
                shot, cock = args
                if shot.startswith("shot") and cock.startswith("cocktail"):
                    gid = self._contains_gids.get((shot, cock))
                    if gid is not None:
                        self._goal_contains.append((shot, cock, gid))

        # Choose a "main" shaker (common benchmark has one)
        self._main_shaker: Optional[str] = None
        if len(self._shakers) == 1:
            self._main_shaker = next(iter(self._shakers))
        elif "shaker1" in self._shakers:
            self._main_shaker = "shaker1"
        elif len(self._shakers) > 0:
            self._main_shaker = sorted(self._shakers)[0]

    # -------- core heuristic API --------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        # If no recognized goals, be safe
        if not self._goal_contains:
            return 0

        shaker = self._main_shaker
        if shaker is None:
            # No shaker known: can't estimate well; admissible lower bound is 0
            return 0

        h = 0

        # Cache some repeated checks
        def is_true_gid(gid: Optional[int]) -> bool:
            if gid is None:
                return False
            return ((state_bitwise >> gid) & 1) == 1

        # Helper: does shaker contain cocktail right now?
        def shaker_has_cocktail(cocktail: str) -> bool:
            gid = self._contains_gids.get((shaker, cocktail))
            return is_true_gid(gid)

        # Helper: does shaker contain ingredient right now?
        def shaker_has_ing(ing: str) -> bool:
            gid = self._contains_gids.get((shaker, ing))
            return is_true_gid(gid)

        # For enabling cost of using *some* shot for fill-shot
        # Lower bound: pick best shot (min additional actions) given current state.
        def min_prepare_shot_cost() -> int:
            best = 0
            found_any = False
            for s in self._shots:
                found_any = True
                empty_ok = is_true_gid(self._empty_gid.get(s))
                clean_ok = is_true_gid(self._clean_gid.get(s))

                cost = 0
                # If not empty, in real domain we'd need empty-shot (>=1)
                if not empty_ok:
                    cost += 1
                # If not clean, we may need clean-shot (>=1). In domain, clean-shot requires empty and used.
                # As a lower bound, if it's not clean we count 1 (optimistic about used/holding/etc.)
                if not clean_ok:
                    cost += 1

                if not found_any or cost < best:
                    best = cost
                    if best == 0:
                        return 0
            return 0 if not found_any else best

        prepare_shot_lb = min_prepare_shot_cost()

        # Shaker clean requirement only matters when we need to pour into an empty shaker.
        shaker_clean = is_true_gid(self._clean_gid.get(shaker))
        shaker_empty = is_true_gid(self._empty_gid.get(shaker))

        # Evaluate each goal cocktail-in-shot
        for shot, cocktail, goal_gid in self._goal_contains:
            if is_true_gid(goal_gid):
                continue

            # At least one pour-shaker-to-shot needed per unsatisfied goal
            h += 1

            # If cocktail is not in shaker, we will need to make it there => need shake (>=1)
            if not shaker_has_cocktail(cocktail):
                h += 1

                ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
                # Missing cocktail part info: can't infer ingredients; keep admissible
                if ing1 is not None and not shaker_has_ing(ing1):
                    # Need to transfer ingredient to shaker: fill-shot + pour-shot-to-shaker (>=2)
                    h += 2
                    # And to do fill-shot, need a clean&empty shot (lower bound preparation)
                    h += prepare_shot_lb
                if ing2 is not None and not shaker_has_ing(ing2):
                    h += 2
                    h += prepare_shot_lb

                # If shaker is empty and we need to pour first ingredient via "pour-shot-to-clean-shaker",
                # then shaker must be clean. If it's empty and not clean, at least one clean-shaker.
                if shaker_empty and (not shaker_clean):
                    h += 1

        # Small additional bound: if any goal remains and target shots are not (clean & empty),
        # then at least 1 step is needed to prepare them before pouring.
        # We do per-shot minimal prep, but only if goal not already met.
        for shot, cocktail, goal_gid in self._goal_contains:
            if is_true_gid(goal_gid):
                continue
            empty_ok = is_true_gid(self._empty_gid.get(shot))
            clean_ok = is_true_gid(self._clean_gid.get(shot))
            if not empty_ok:
                h += 1
            if not clean_ok:
                h += 1

        # Keep heuristic non-negative int
        return int(h) if h > 0 else 0

    # -------- utilities --------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
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
