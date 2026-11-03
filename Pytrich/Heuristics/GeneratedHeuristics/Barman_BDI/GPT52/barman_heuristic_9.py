from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic9(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Design rationale (domain insights -> heuristic signals)
    -----------------------------------------------
    In this Barman HTN domain, the top-level goals are typically instances of:
        (AchieveContainsShotCocktail shotX cocktailY)

    The canonical method MakeAndPourCocktail expands into:
        1) AchieveContainsShakerCocktail(shaker, cocktail)
        2) AchieveCleanShot(shot)
        3) AchieveHolding(hand, shaker)
        4) DoPourShakerToShot(shaker, shot, cocktail)

    And AchieveContainsShakerCocktail usually expands into:
        - AchieveCleanShaker(shaker)
        - AchieveContainsShakerIngredient(shaker, part1)
        - AchieveContainsShakerIngredient(shaker, part2)
        - AchieveHolding(hand, shaker)
        - AchieveHandEmpty(otherhand)
        - shake(...)

    Key planning challenges that repeatedly cause search blow-up:
      * Hand resource: two hands, must be empty/holding appropriately.
      * Clean/empty constraints on shots and shaker.
      * Shaker content lifecycle: once shaken it contains the cocktail; to make
        another cocktail you often must empty/clean the shaker (delete effects).

    Heuristic strategy
    ------------------
    We compute a fast, admissible lower bound for the remaining *primitive* work
    by counting missing "landmarks" for each pending AchieveContainsShotCocktail.

    For each (shot, cocktail) not yet satisfied, we add a conservative estimate:
      - If shot already contains cocktail: 0.
      - Else at least one pour-shaker-to-shot must happen: +1.
      - To pour, shot must be (clean and empty) at that time.
        If currently not empty: +1 (need empty-shot at least once).
        If currently not clean: +1 (need clean-shot at least once).
      - Also, the cocktail must exist in some shaker and be shaked.
        If there exists a shaker that currently (contains shaker cocktail) AND (shaked shaker): 0
        Else we must (at least) do a shake: +1.
        Additionally, for each missing cocktail ingredient in the shaker(s), we add
        a relaxed lower bound on filling those ingredients: +1 per missing ingredient.
        (In reality each ingredient requires fill-shot + pour-shot-to-(clean/used)-shaker,
         but counting +1 keeps it admissible.)

    Task-network awareness
    ----------------------
    We only charge for *pending* abstract tasks in the current task network (tn).
    In the provided benchmarks, the initial TN contains only AchieveContainsShotCocktail,
    and search introduces further subtasks via decomposition. This heuristic remains
    useful throughout search because it re-reads the current TN and focuses on
    cocktail delivery tasks that are still present.

    Efficiency
    ----------
    - Uses bitwise checks for facts (no state_explicit_repr(state) in the hot path).
    - Pre-indexes facts by predicate and arguments at init.
    - Does not run expensive relaxed planning graphs.

    Notes on admissibility
    ----------------------
    All increments correspond to actions that must occur at least once in any valid
    plan under standard semantics (e.g., if shot not empty you must empty it before
    it can be poured into; if cocktail not available shaked in any shaker you must
    perform a shake to create it; if ingredient not present you must introduce it).
    We intentionally undercount multi-step requirements (e.g., ingredients), so the
    heuristic is a safe lower bound.
    """

    def __init__(self, model, barman_gpt52_heuristic_9: str = "barman_gpt52_heuristic_9"):
        super().__init__(model, barman_gpt52_heuristic_9)
        self.model = model

        # --- Pre-index facts for O(1-ish) access ---
        # Map: predicate -> list of (fact_global_id, args_tuple)
        self._pred_facts: Dict[str, List[Tuple[int, Tuple[str, ...]]]] = {}

        # Direct lookup for some frequently used unary/binary predicates
        self._fact_id_contains: Dict[Tuple[str, str], List[int]] = {}  # (container, beverage) -> [fact_id]
        self._fact_id_clean: Dict[str, int] = {}
        self._fact_id_empty: Dict[str, int] = {}
        self._fact_id_shaked: Dict[str, int] = {}

        # cocktail parts (static in all provided instances)
        self._cocktail_part1: Dict[str, str] = {}  # cocktail -> ingredient
        self._cocktail_part2: Dict[str, str] = {}

        # object sets (parsed from facts; robust even if typing info isn't directly exposed)
        self._all_shakers: set[str] = set()
        self._all_shots: set[str] = set()

        self._build_fact_indices()
        self._extract_cocktail_recipes_from_init()

    # ----------------- Parsing utilities -----------------
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

    def _build_fact_indices(self):
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            targs = tuple(args)
            self._pred_facts.setdefault(pred, []).append((fact.global_id, targs))

            # Fast lookup tables
            if pred == "contains" and len(args) == 2:
                self._fact_id_contains.setdefault((args[0], args[1]), []).append(fact.global_id)
            elif pred == "clean" and len(args) == 1:
                self._fact_id_clean[args[0]] = fact.global_id
            elif pred == "empty" and len(args) == 1:
                self._fact_id_empty[args[0]] = fact.global_id
            elif pred == "shaked" and len(args) == 1:
                self._fact_id_shaked[args[0]] = fact.global_id

            # Infer shaker/shot object names from predicate argument positions.
            # - shaker appears as arg0 of shakerLevel/shakerEmptyLevel/shaked/unshaked
            if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and len(args) >= 1:
                self._all_shakers.add(args[0])
            # - shot appears as arg0 of fill-shot/clean-shot preconditions in facts: used/clean/empty/ontable/contains
            #   Harder to infer; we just collect from facts where predicate applies to containers, then later filter.
            if pred in ("used", "clean", "empty", "ontable", "contains") and len(args) >= 1:
                # This includes shaker too; we will keep shots set by excluding known shakers
                pass

        # Infer shots as "containers that are not known shakers" using clean/empty/ontable facts
        candidate_containers = set()
        for p in ("clean", "empty", "ontable"):
            for _, a in self._pred_facts.get(p, []):
                if len(a) == 1:
                    candidate_containers.add(a[0])
        self._all_shots = set(x for x in candidate_containers if x not in self._all_shakers)

    def _extract_cocktail_recipes_from_init(self):
        # Recipes appear as facts in the initial state.
        init = self.model.initial_state
        # cocktailPart1[cocktail, ingredient]
        for fid, args in self._pred_facts.get("cocktailPart1", []):
            if ((init >> fid) & 1) and len(args) == 2:
                self._cocktail_part1[args[0]] = args[1]
        for fid, args in self._pred_facts.get("cocktailPart2", []):
            if ((init >> fid) & 1) and len(args) == 2:
                self._cocktail_part2[args[0]] = args[1]

    # ----------------- Bitwise fact checks -----------------
    def _is_true_fact_id(self, state: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state >> fact_id) & 1) == 1

    def _is_contains(self, state: int, container: str, beverage: str) -> bool:
        for fid in self._fact_id_contains.get((container, beverage), []):
            if ((state >> fid) & 1) == 1:
                return True
        return False

    def _is_clean(self, state: int, container: str) -> bool:
        return self._is_true_fact_id(state, self._fact_id_clean.get(container))

    def _is_empty(self, state: int, container: str) -> bool:
        return self._is_true_fact_id(state, self._fact_id_empty.get(container))

    def _is_shaked(self, state: int, shaker: str) -> bool:
        return self._is_true_fact_id(state, self._fact_id_shaked.get(shaker))

    # ----------------- Task extraction -----------------
    def _parse_task_name(self, task_obj) -> Tuple[str, List[str]]:
        """Best-effort extraction of (task_name, [args]) from Pytrich task/operator objects.

        For abstract tasks, task_obj.name typically looks like:
          'AchieveContainsShotCocktail(shot2,cocktail1)'
        but formats can vary across parsers. We handle common patterns.
        """
        raw = getattr(task_obj, "name", "")
        if not raw:
            return "", []

        # If already like Task(args)
        lpar = raw.find("(")
        rpar = raw.rfind(")")
        if lpar != -1 and rpar != -1 and rpar > lpar:
            tname = raw[:lpar].strip()
            arg_str = raw[lpar + 1 : rpar].strip()
            args = [a.strip() for a in arg_str.split(",")] if arg_str else []
            return tname, args

        # Fallback: no args encoded
        return raw.strip(), []

    def _pending_shot_cocktail_tasks(self, task_network: List[Union[object, object]]):
        res = []
        for t in task_network:
            tname, args = self._parse_task_name(t)
            if tname == "AchieveContainsShotCocktail" and len(args) == 2:
                res.append((args[0], args[1]))
        return res

    # ----------------- Heuristic core -----------------
    def __call__(self, state: int, task_network: List[Union[object, object]] = None) -> int:
        if task_network is None:
            task_network = []

        # If no tasks left, heuristic 0 (search should check goal separately).
        pending = self._pending_shot_cocktail_tasks(task_network)
        if not pending:
            return 0

        h = 0

        for shot, cocktail in pending:
            # Already achieved?
            if self._is_contains(state, shot, cocktail):
                continue

            # Must pour cocktail into shot at least once
            h += 1

            # Shot must be empty and clean (at time of pour). If currently not,
            # at least one action will be needed to fix each.
            if not self._is_empty(state, shot):
                h += 1
            if not self._is_clean(state, shot):
                h += 1

            # Need cocktail available in some shaker in shaked form.
            cocktail_ready = False
            for sh in self._all_shakers:
                if self._is_shaked(state, sh) and self._is_contains(state, sh, cocktail):
                    cocktail_ready = True
                    break

            if cocktail_ready:
                continue

            # Otherwise, at least one shake must occur.
            h += 1

            # Additionally, to be able to shake, the shaker must contain both ingredients.
            # We compute a relaxed lower bound by checking if *some* shaker already contains
            # each ingredient; if not, add +1 per missing ingredient.
            ing1 = self._cocktail_part1.get(cocktail)
            ing2 = self._cocktail_part2.get(cocktail)

            # If recipe unknown, don't overcommit; just keep the +1 for shake.
            if ing1 and ing2:
                ing1_present = False
                ing2_present = False
                for sh in self._all_shakers:
                    if not ing1_present and self._is_contains(state, sh, ing1):
                        ing1_present = True
                    if not ing2_present and self._is_contains(state, sh, ing2):
                        ing2_present = True
                    if ing1_present and ing2_present:
                        break

                if not ing1_present:
                    h += 1
                if not ing2_present:
                    h += 1

        return h
