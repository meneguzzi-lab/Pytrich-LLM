from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic16(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Rationale (domain insights -> heuristic components)
    -----------------------------------------------
    The top-level training tasks are sequences of (AchieveContainsShotCocktail shotX cocktailY).
    In this domain, the *only* way to put a cocktail into a shot (when it is not already there)
    is via the method MakeAndPourCocktail, which forces the following key structure:

      1) Make cocktail in shaker:
         - Ensure shaker is clean (maybe empty it first if full & shaked)
         - Add two ingredients (each requires filling some shot from a dispenser and pouring to shaker)
         - Shake (requires one hand holding shaker, the other hand empty)
      2) Ensure target shot is clean & empty
      3) Hold shaker
      4) Pour shaker -> shot

    Planning challenges:
      - Cleaning resources: a used/dirty shot must be cleaned; a shaker might need emptying then cleaning.
      - Hand management: shake needs two hands in specific conditions.
      - Many goals in large instances: heuristic must be O(#goals) and use bitwise checks only.

    Heuristic design (admissible lower bound):
      - We count remaining *unsatisfied goal facts* (+contains[shot,cocktail]).
      - For each unsatisfied goal, add a small, conservative lower bound on the number of primitive
        actions still needed in a relaxed sense.

    Specifically, for each goal (+contains shot cocktail) not true in the state:
      - If some shaker already contains that cocktail and is shaked, then at least 1 action remains:
        pour-shaker-to-shot. (We ignore preconditions like clean/empty/holding to stay admissible.)
      - Otherwise, at least 1 action remains to make it available somewhere (via shake) plus 1 to pour.
        So we use 2 as a conservative bound. (Real cost is much higher, but 2 is safely <= optimal cost.)

    Additionally, we add tiny landmark-style lower bounds that are still admissible:
      - If the target shot is not empty, at least 1 action is needed (empty-shot) *before* it can be
        refilled with cocktail. (Again ignoring holding constraints.)
      - If the target shot is not clean, at least 1 action is needed (clean-shot) *at some point*.

    These add-ons improve guidance in states where shots become dirty/non-empty due to earlier actions,
    without risking overestimation.

    Implementation notes:
      - Uses ONLY bitwise fact checks (global_id) for speed.
      - Parses goal facts from model.goals via state_explicit_repr.
      - Precomputes mappings from (predicate,args)->Fact and useful indices over facts.

    This heuristic is not meant to be tight in absolute cost, but is (a) admissible, (b) consistent in
    practice because it is based on monotone counts of missing landmarks/goals, and (c) fast enough for
    large instances with 50+ goals.
    """

    def __init__(self, model, barman_gpt52_heuristic_16: str = "barman_gpt52_heuristic_16"):
        super().__init__(model, barman_gpt52_heuristic_16)
        self.model = model

        # Fact lookup by exact grounded name
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # Index facts by predicate for faster scans
        self._facts_by_pred: Dict[str, List[object]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append(f)

        # Precompute goal contains(shot,cocktail) facts
        self._goal_contains: List[Tuple[str, str, object]] = []  # (shot, cocktail, Fact)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "contains" and len(args) == 2:
                shot, cocktail = args[0], args[1]
                fact = self._fact_by_name.get(n)
                if fact is not None:
                    self._goal_contains.append((shot, cocktail, fact))

        # Precompute helpful shaker-cocktail facts list for quick check: contains(shaker,cocktail)
        # We'll just keep all contains facts and filter by type in a relaxed way (by name prefix is unsafe);
        # instead, we only use the fact structure itself: we look for predicates and arity.
        self._contains_facts: List[Tuple[str, str, object]] = []  # (container, beverage, Fact)
        for f in self._facts_by_pred.get("contains", []):
            pred, args = self._parse_fact_name(f.name)
            if pred == "contains" and len(args) == 2:
                self._contains_facts.append((args[0], args[1], f))

        # Precompute shaked(shaker) facts
        self._shaked_facts: Dict[str, object] = {}  # shaker -> Fact
        for f in self._facts_by_pred.get("shaked", []):
            pred, args = self._parse_fact_name(f.name)
            if pred == "shaked" and len(args) == 1:
                self._shaked_facts[args[0]] = f

        # Precompute empty(container) and clean(container) fact templates by object name
        self._empty_fact_by_obj: Dict[str, object] = {}
        for f in self._facts_by_pred.get("empty", []):
            pred, args = self._parse_fact_name(f.name)
            if pred == "empty" and len(args) == 1:
                self._empty_fact_by_obj[args[0]] = f

        self._clean_fact_by_obj: Dict[str, object] = {}
        for f in self._facts_by_pred.get("clean", []):
            pred, args = self._parse_fact_name(f.name)
            if pred == "clean" and len(args) == 1:
                self._clean_fact_by_obj[args[0]] = f

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

    @staticmethod
    def _is_true(state_bitwise: int, fact) -> bool:
        return ((state_bitwise >> fact.global_id) & 1) == 1

    def _some_shaker_has_shaked_cocktail(self, state: int, cocktail: str) -> bool:
        # Check if exists shaker S such that contains(S,cocktail) and shaked(S)
        # We don't have explicit typing here, but only shakers can be shaked, so we use that.
        for shaker, shaked_fact in self._shaked_facts.items():
            if not self._is_true(state, shaked_fact):
                continue
            # need contains(shaker,cocktail)
            # scan contains facts that match (shaker,cocktail)
            # (small: only one shaker typically; still safe)
            for c_obj, bev, f in self._contains_facts:
                if c_obj == shaker and bev == cocktail and self._is_true(state, f):
                    return True
        return False

    def __call__(self, parent_node, node) -> int:
        state = node.state

        h = 0

        # For each goal contains(shot,cocktail) missing, add a conservative estimate.
        for shot, cocktail, goal_fact in self._goal_contains:
            if self._is_true(state, goal_fact):
                continue

            # Base lower bound: either cocktail already made in shaker (then at least pour), else (make + pour)
            if self._some_shaker_has_shaked_cocktail(state, cocktail):
                h += 1
            else:
                h += 2

            # Additional admissible lower bounds about the target shot readiness.
            empty_fact = self._empty_fact_by_obj.get(shot)
            if empty_fact is not None and not self._is_true(state, empty_fact):
                h += 1

            clean_fact = self._clean_fact_by_obj.get(shot)
            if clean_fact is not None and not self._is_true(state, clean_fact):
                h += 1

        return h
