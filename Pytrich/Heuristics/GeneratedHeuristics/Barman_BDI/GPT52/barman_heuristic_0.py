from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic0(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Rationale (domain insights -> heuristic features)
    -----------------------------------------------
    In this domain, the top-level goal tasks are typically:
        AchieveContainsShotCocktail(shot, cocktail)
    which (when not already satisfied) decomposes into:
        1) AchieveContainsShakerCocktail(shaker, cocktail)
           - requires shaker clean
           - requires adding two ingredients (via filling a shot from a dispenser and pouring into shaker)
           - requires shaking
        2) AchieveCleanShot(shot)
        3) AchieveHolding(hand, shaker)
        4) DoPourShakerToShot(shaker, shot, cocktail)

    The real search difficulty comes from (a) hand resource constraints (two hands),
    (b) cleaning/emptying dependencies (used/clean/empty),
    (c) needing the *right* ingredients available via dispensers, and
    (d) ensuring shaker is shaked and contains the cocktail before pouring.

    Heuristic strategy
    ------------------
    We compute a fast, admissible* lower bound by summing independent "missing-work"
    costs per pending abstract task in the current task network.

    *Admissible in the sense of being a lower bound on remaining primitive actions,
    using optimistic assumptions:
      - ignore negative interactions between different pending tasks (may undercount)
      - assume there exists some suitable hand/shaker/shot to achieve subgoals (optimistic)
      - do not charge for moving between objects (not modeled anyway)

    We also add small, safe "fact-distance" components for key enabling facts:
      - if a task needs holding(x) but nobody holds x, at least one grasp is needed
      - if a container must be ontable but isn't, at least one drop is needed

    Computational efficiency
    ------------------------
    - Uses only bitwise checks on facts (no state_explicit_repr(state) on runtime path).
    - Pre-indexes facts by predicate and key arguments to O(1)/small-set membership tests.
    - Parses the current task network by reading task.name strings.

    Note: This heuristic is designed to reduce node expansions (good guidance) while
    remaining lightweight for large instances.
    """

    def __init__(self, barman_gpt52_heuristic_0):
        super().__init__(barman_gpt52_heuristic_0)
        self.model = barman_gpt52_heuristic_0

        # --- Build fact indices: predicate -> list of (args_tuple, global_id)
        self._fact_index: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
        # Also keep direct lookup: (pred, args_tuple) -> global_id
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            at = tuple(args)
            self._fact_index.setdefault(pred, []).append((at, f.global_id))
            self._fact_gid[(pred, at)] = f.global_id

        # Domain-specific caches
        self._hands: List[str] = self._all_objects_of_type_hint(["left", "right"], fallback_from_facts=True)
        self._shakers: List[str] = self._all_objects_from_predicate_arg("shakerLevel", 0)
        self._shots: List[str] = self._all_objects_from_predicate_arg("clean", 0, type_filter_prefix="shot")

        # Cocktail recipes: cocktail -> (ing1, ing2)
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._init_cocktail_parts()

        # Which dispenser dispenses which ingredient: ingredient -> [dispenser]
        self._dispensers_for_ing: Dict[str, List[str]] = {}
        self._init_dispenses_map()

        # Precompute sets of fact global_ids by predicate for quicker scans
        self._pred_gids: Dict[str, List[int]] = {p: [gid for _, gid in lst] for p, lst in self._fact_index.items()}

    # -------------------- Fact parsing and indexing helpers --------------------

    def _parse_fact_name(self, name: str):
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

    def _bit_true(self, state: int, gid: int) -> bool:
        return ((state >> gid) & 1) == 1

    def _true_fact_exists(self, state: int, pred: str, args: Tuple[str, ...]) -> bool:
        gid = self._fact_gid.get((pred, args))
        if gid is None:
            return False
        return self._bit_true(state, gid)

    def _any_true_with_first_arg(self, state: int, pred: str, a0: str) -> bool:
        for args, gid in self._fact_index.get(pred, []):
            if len(args) >= 1 and args[0] == a0 and self._bit_true(state, gid):
                return True
        return False

    def _any_true_with_args_prefix(self, state: int, pred: str, prefix: Tuple[str, ...]) -> bool:
        plen = len(prefix)
        for args, gid in self._fact_index.get(pred, []):
            if len(args) >= plen and args[:plen] == prefix and self._bit_true(state, gid):
                return True
        return False

    def _all_objects_from_predicate_arg(self, pred: str, arg_index: int, type_filter_prefix: Optional[str] = None) -> List[str]:
        objs = set()
        for args, _gid in self._fact_index.get(pred, []):
            if len(args) > arg_index:
                o = args[arg_index]
                if type_filter_prefix is None or o.startswith(type_filter_prefix):
                    objs.add(o)
        return sorted(objs)

    def _all_objects_of_type_hint(self, hints: List[str], fallback_from_facts: bool = True) -> List[str]:
        # In provided instances, hands are literally named left/right.
        # If not present, try to recover hands from predicates handEmpty/holding.
        hands = [h for h in hints if any(self._any_object_named(h) for _ in [0])]
        if hands:
            return hands
        if not fallback_from_facts:
            return []
        objs = set()
        for pred in ("handEmpty", "holding"):
            for args, _gid in self._fact_index.get(pred, []):
                if args:
                    objs.add(args[0])
        return sorted(objs)

    def _any_object_named(self, obj: str) -> bool:
        # We don't have explicit object lists; infer existence if it appears in any fact.
        for lst in self._fact_index.values():
            for args, _ in lst:
                if obj in args:
                    return True
        return False

    # -------------------- Domain structure extraction --------------------

    def _init_cocktail_parts(self) -> None:
        part1: Dict[str, str] = {}
        part2: Dict[str, str] = {}
        for args, _gid in self._fact_index.get("cocktailPart1", []):
            if len(args) == 2:
                part1[args[0]] = args[1]
        for args, _gid in self._fact_index.get("cocktailPart2", []):
            if len(args) == 2:
                part2[args[0]] = args[1]
        cocktails = set(part1) | set(part2)
        for c in cocktails:
            self._cocktail_parts[c] = (part1.get(c), part2.get(c))

    def _init_dispenses_map(self) -> None:
        for args, _gid in self._fact_index.get("dispenses", []):
            if len(args) == 2:
                disp, ing = args
                self._dispensers_for_ing.setdefault(ing, []).append(disp)

    # -------------------- Task parsing helpers --------------------

    def _parse_task(self, task_name: str) -> Tuple[str, List[str]]:
        """Parse a task/operator name like 'AchieveContainsShotCocktail shot2 cocktail1'
        into (functor, [args...]).

        Pytrich uses the 'name' field for both operators and abstract tasks; for tasks
        it typically contains the functor and grounded arguments separated by spaces.
        """
        s = task_name.strip()
        if not s:
            return "", []
        parts = s.split()
        return parts[0], parts[1:]

    # -------------------- Lower-bound cost models for key tasks --------------------

    def _lb_achieve_clean_shot(self, state: int, shot: str) -> int:
        # If already clean: 0
        if self._true_fact_exists(state, "clean", (shot,)):
            return 0
        # Optimistic: either just clean-shot (requires empty+used+holding+handEmpty)
        # or empty-shot then clean-shot. We lower-bound by at least 1 action (clean-shot)
        # but if it isn't empty, at least 2 (empty-shot + clean-shot).
        lb = 1
        if not self._true_fact_exists(state, "empty", (shot,)):
            lb = 2
        return lb

    def _lb_achieve_clean_shaker(self, state: int, shaker: str) -> int:
        if self._true_fact_exists(state, "clean", (shaker,)):
            return 0
        # Need clean-shaker, and if not empty we may need empty-shaker first.
        lb = 1
        if not self._true_fact_exists(state, "empty", (shaker,)):
            lb = 2
        return lb

    def _lb_achieve_holding(self, state: int, container: str) -> int:
        # If any hand already holds it: 0
        if self._any_true_with_args_prefix(state, "holding", ("left", container)) or \
           self._any_true_with_args_prefix(state, "holding", ("right", container)):
            return 0
        # Otherwise, at least a grasp is required.
        # We do not additionally count making hand empty / putting on table because that's extra,
        # and we want a safe LB.
        return 1

    def _lb_achieve_contains_shot_ingredient(self, state: int, shot: str, ing: str) -> int:
        if self._true_fact_exists(state, "contains", (shot, ing)):
            return 0
        # Minimal way is fill-shot: 1 action, but requires empty+clean+holding+handEmpty+dispenses.
        # As LB, if a dispenser exists we can still say >=1; if no dispenser, task is impossible
        # but heuristic should stay finite; return 1 anyway (optimistic).
        return 1

    def _lb_achieve_contains_shaker_ingredient(self, state: int, shaker: str, ing: str) -> int:
        if self._true_fact_exists(state, "contains", (shaker, ing)):
            return 0
        # To get ingredient into shaker: must have it in some shot and then pour to shaker.
        # Lower bound: at least 2 actions (fill-shot + pour-shot-to-*-shaker).
        # We ignore cleaning/holding/handEmpty requirements for LB.
        return 2

    def _lb_make_cocktail_in_shaker(self, state: int, shaker: str, cocktail: str) -> int:
        # If shaker already contains cocktail: 0
        if self._true_fact_exists(state, "contains", (shaker, cocktail)):
            return 0

        ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
        # If recipe unknown, be optimistic but non-zero
        if ing1 is None or ing2 is None:
            return 1

        # Lower bound components:
        # - ensure shaker clean (>=0)
        # - ensure two ingredients in shaker (each >=0/2)
        # - shake (>=1) if not already contains cocktail
        lb = 0
        lb += self._lb_achieve_clean_shaker(state, shaker)
        lb += self._lb_achieve_contains_shaker_ingredient(state, shaker, ing1)
        lb += self._lb_achieve_contains_shaker_ingredient(state, shaker, ing2)
        lb += 1  # shake
        return lb

    def _lb_pour_shaker_to_shot(self, state: int, shaker: str, shot: str, cocktail: str) -> int:
        # If shot already contains cocktail, 0
        if self._true_fact_exists(state, "contains", (shot, cocktail)):
            return 0
        # Pour action itself: 1
        return 1

    def _choose_best_shaker_for_cocktail(self, state: int, cocktail: str) -> str:
        # Prefer a shaker that already contains the cocktail, else any shaker.
        for sh in self._shakers:
            if self._true_fact_exists(state, "contains", (sh, cocktail)):
                return sh
        return self._shakers[0] if self._shakers else "shaker1"

    # -------------------- Main heuristic --------------------

    def __call__(self, state: int, task_network) -> int:
        """Compute heuristic value for (state, task_network).

        We sum optimistic lower bounds for each pending abstract task.
        Additionally, we add tiny enabling-cost signals for obvious blockers
        (e.g., needing to hold something).
        """
        h = 0

        # If no pending tasks, rely on goal facts only (Pytrich's goal_reached also checks tn empty).
        # Still, keep heuristic 0 in that case.
        if not task_network:
            return 0

        # For repeated subtasks, we can get some benefit by not double-counting exact same request.
        seen: set = set()

        for t in task_network:
            name = getattr(t, "name", "")
            functor, args = self._parse_task(name)
            key = (functor, tuple(args))
            if key in seen:
                continue
            seen.add(key)

            if functor == "AchieveContainsShotCocktail" and len(args) == 2:
                shot, cocktail = args
                if self._true_fact_exists(state, "contains", (shot, cocktail)):
                    continue
                shaker = self._choose_best_shaker_for_cocktail(state, cocktail)
                # Lower bound for making cocktail in shaker + cleaning shot + holding shaker + pour
                h += self._lb_make_cocktail_in_shaker(state, shaker, cocktail)
                h += self._lb_achieve_clean_shot(state, shot)
                h += self._lb_achieve_holding(state, shaker)
                h += self._lb_pour_shaker_to_shot(state, shaker, shot, cocktail)

            elif functor == "AchieveContainsShakerCocktail" and len(args) == 2:
                shaker, cocktail = args
                h += self._lb_make_cocktail_in_shaker(state, shaker, cocktail)

            elif functor == "AchieveContainsShakerIngredient" and len(args) == 2:
                shaker, ing = args
                h += self._lb_achieve_contains_shaker_ingredient(state, shaker, ing)

            elif functor == "AchieveContainsShotIngredient" and len(args) == 2:
                shot, ing = args
                h += self._lb_achieve_contains_shot_ingredient(state, shot, ing)

            elif functor == "AchieveCleanShot" and len(args) == 1:
                h += self._lb_achieve_clean_shot(state, args[0])

            elif functor == "AchieveCleanShaker" and len(args) == 1:
                h += self._lb_achieve_clean_shaker(state, args[0])

            elif functor == "AchieveHolding" and len(args) == 2:
                _hand, cont = args
                h += self._lb_achieve_holding(state, cont)

            elif functor == "AchieveOnTable" and len(args) == 1:
                cont = args[0]
                if not self._true_fact_exists(state, "ontable", (cont,)):
                    # At least one drop required to put it on the table
                    h += 1

            elif functor == "AchieveHandEmpty" and len(args) == 1:
                hand = args[0]
                # If already empty, 0; else at least one drop is needed.
                if not self._true_fact_exists(state, "handEmpty", (hand,)):
                    h += 1

            elif functor == "DoPourShakerToShot" and len(args) == 3:
                shaker, shot, cocktail = args
                h += self._lb_pour_shaker_to_shot(state, shaker, shot, cocktail)

            else:
                # Unknown task/operator: return 0 contribution (safe, optimistic)
                pass

        # Small stabilization: never negative
        if h < 0:
            h = 0
        return h
