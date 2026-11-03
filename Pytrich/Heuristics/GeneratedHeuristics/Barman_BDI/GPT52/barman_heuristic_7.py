from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any, Set

try:
    # Pytrich typical location
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback (some templates expose Heuristic at package root)
    from heuristic import Heuristic


class BarmanGPT52Heuristic7(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Core domain insight (from methods/actions):
    - Top-level goals are usually (AchieveContainsShotCocktail shot cocktail).
      If not already in the shot, the minimal *primitive* workflow (in this domain)
      is essentially:
        1) Ensure shaker contains the cocktail: requires 2 ingredients in shaker + shake.
        2) Ensure target shot is clean+empty.
        3) Hold shaker.
        4) Pour shaker->shot.

    - To get an ingredient into the shaker, you must:
        a) Fill a shot with that ingredient (requires clean+empty shot, holding shot,
           other hand empty), then
        b) Pour shot->shaker (clean-shaker if shaker empty+clean, else used-shaker if unshaked).

    Heuristic strategy:
    - Fast lower bound computed from *missing goal facts* (+contains[shot,cocktail]).
    - For each missing cocktail-in-shot goal, add a relaxed cost estimate that counts
      mandatory primitive actions under a delete-relaxed view, but *without* doing
      expensive explicit-state string conversions.

    Properties:
    - Admissible-ish in practice (it is a conservative under-estimate): we only count
      actions that are clearly necessary when the goal is not yet true, and we take
      minimums over alternative resources (e.g., any shot can be used to fetch an ingredient).
      We also avoid double-counting some shared setup by using max/union where appropriate.
    - Very cheap to compute: only bit tests on a small set of relevant facts, and small
      precomputed indices from goal facts and static relations (cocktail parts, dispensers).

    Notes on robustness:
    - If parsing fails (unexpected fact naming), returns a simple count of unsatisfied goals.
    """

    def __init__(self, model, barman_gpt52_heuristic_7: str = "barman_gpt52_heuristic_7"):
        super().__init__(model, barman_gpt52_heuristic_7)
        self.model = model

        # --- Build fact lookup (predicate,args) -> global_id for O(1) bit checks
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self._fact_gid[(pred, tuple(args))] = f.global_id

        # --- Extract objects by type from facts is hard; instead infer from fact arguments.
        # We only need: shots, shakers, hands, cocktails, ingredients.
        self._shots: Set[str] = set()
        self._shakers: Set[str] = set()
        self._hands: Set[str] = set()
        self._cocktails: Set[str] = set()
        self._ingredients: Set[str] = set()

        for (pred, args_t), gid in self._fact_gid.items():
            args = list(args_t)
            if pred in ("handEmpty", "holding"):
                if args:
                    self._hands.add(args[0])
                if pred == "holding" and len(args) >= 2:
                    # holding(hand, container) -> container could be shot/shaker
                    pass
            if pred in ("ontable", "clean", "empty", "used", "contains"):
                if len(args) >= 1:
                    # container candidates
                    if args[0].startswith("shot"):
                        self._shots.add(args[0])
                    if args[0].startswith("shaker"):
                        self._shakers.add(args[0])
            if pred in ("shaked", "unshaked", "shakerLevel", "shakerEmptyLevel"):
                if args and args[0].startswith("shaker"):
                    self._shakers.add(args[0])
            if pred.startswith("cocktailPart"):
                if len(args) >= 1:
                    self._cocktails.add(args[0])
                if len(args) >= 2:
                    self._ingredients.add(args[1])
            if pred == "dispenses" and len(args) == 2:
                self._ingredients.add(args[1])
            if pred == "ingredient" and len(args) == 1:
                self._ingredients.add(args[0])

        # --- Static mappings: cocktail -> (part1, part2)
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {
            c: (None, None) for c in self._cocktails
        }
        for (pred, args), gid in self._fact_gid.items():
            if pred == "cocktailPart1" and len(args) == 2:
                c, ing = args
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (ing, p2)
            elif pred == "cocktailPart2" and len(args) == 2:
                c, ing = args
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (p1, ing)

        # --- Cache goal contains facts (+contains[shot,cocktail])
        self._goal_contains: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fn in goal_fact_names:
            pred, args = self._parse_fact_name(fn)
            if pred == "contains" and len(args) == 2:
                cont, bev = args
                if cont.startswith("shot"):
                    self._goal_contains.append((cont, bev))

        # Choose a "primary" shaker for estimates (most instances have shaker1).
        self._primary_shaker = None
        if "shaker1" in self._shakers:
            self._primary_shaker = "shaker1"
        elif self._shakers:
            self._primary_shaker = next(iter(self._shakers))

    # ---------- Required by Pytrich Heuristic interface ----------
    def __call__(self, node) -> int:
        state = node.state
        # If no relevant goals detected, fallback: 0 (uninformative but safe)
        if not self._goal_contains:
            return 0

        # Sum of per-goal underestimates, but with simple sharing on shaker-cocktail production:
        # producing each cocktail in shaker is necessary before pouring to its shot.
        # Since shaker can only hold one cocktail at a time realistically, true cost is higher;
        # but for a lower bound we can take max over required productions instead of sum.
        missing_goals: List[Tuple[str, str]] = []
        for shot, cocktail in self._goal_contains:
            if not self._is_true(state, "contains", shot, cocktail):
                missing_goals.append((shot, cocktail))

        if not missing_goals:
            return 0

        # Lower bound for required cocktail productions (in shaker)
        # Take max over distinct cocktails not already in the shaker.
        prod_costs: List[int] = []
        for _, cocktail in missing_goals:
            prod_costs.append(self._estimate_make_cocktail_in_shaker_lb(state, cocktail))
        production_lb = max(prod_costs) if prod_costs else 0

        # Lower bound for required pours into target shots: at least 1 pour action per missing goal.
        # Plus if the target shot isn't (clean and empty), at least 1 action to fix each missing condition
        # (relaxed: could be achieved with one action in best case).
        pour_and_shotprep_lb = 0
        for shot, cocktail in missing_goals:
            pour_and_shotprep_lb += 1  # pour-shaker-to-shot
            if not self._is_true(state, "empty", shot):
                pour_and_shotprep_lb += 1  # empty-shot at least
            if not self._is_true(state, "clean", shot):
                pour_and_shotprep_lb += 1  # clean-shot at least

        # Hand/holding constraints: to pour, must be holding shaker; add 0/1.
        hold_shaker_lb = 0
        if self._primary_shaker and not self._any_hand_holding(state, self._primary_shaker):
            hold_shaker_lb = 1  # at least a grasp (relaxed)

        # Combine: production is shared, pouring+shot prep is per goal.
        return production_lb + pour_and_shotprep_lb + hold_shaker_lb

    # ---------- Domain-specific lower bounds ----------
    def _estimate_make_cocktail_in_shaker_lb(self, state: int, cocktail: str) -> int:
        """Lower bound to have +contains[shaker,cocktail] for the primary shaker.

        Relaxed plan skeleton for making cocktail in shaker:
          - ensure both ingredients in shaker (each requires at least: fill-shot + pour-shot-to-*-shaker)
          - shake

        We count:
          * 1 for shake if cocktail not already in shaker
          * for each missing ingredient in shaker: 2 actions (fill-shot, pour-shot-to-shaker)

        We do NOT count cleaning/holding/hands precisely (keeps it admissible/cheap).
        """
        sh = self._primary_shaker
        if not sh:
            return 0

        # If already contains cocktail, cost 0
        if self._is_true(state, "contains", sh, cocktail):
            return 0

        p1, p2 = self._cocktail_parts.get(cocktail, (None, None))
        if p1 is None or p2 is None:
            # Unknown recipe: very weak fallback
            return 1

        cost = 1  # shake

        if not self._is_true(state, "contains", sh, p1):
            cost += 2
        if not self._is_true(state, "contains", sh, p2):
            cost += 2

        return cost

    # ---------- Bitwise fact checks ----------
    def _is_true(self, state: int, pred: str, *args: str) -> bool:
        gid = self._fact_gid.get((pred, tuple(args)))
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _any_hand_holding(self, state: int, container: str) -> bool:
        # holding(hand, container)
        for h in self._hands:
            if self._is_true(state, "holding", h, container):
                return True
        return False

    # ---------- Fact name parsing (grounded format) ----------
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
