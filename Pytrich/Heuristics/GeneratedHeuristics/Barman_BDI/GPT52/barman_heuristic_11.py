from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Iterable, Set

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic11(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Rationale (domain insights -> heuristic):
    - Top-level goals are typically `+contains[shotX,cocktailY]`.
    - To achieve one missing cocktail-in-shot, the *primitive* work is roughly:
        (1) build cocktail in shaker: add two ingredients + shake
        (2) pour shaker to the target shot
      This is a stable backbone across instances.
    - The biggest dynamic hurdles are:
        * whether the shaker is already holding the correct cocktail and shaked
        * whether the shaker already has one/both required ingredients
        * whether the target shot is already clean+empty
        * whether someone is holding required containers / hands empty

    The heuristic computes a *lower bound* by summing per-goal lower bounds,
    but uses a conservative max-aggregation for shared resources (one shaker,
    two hands) to avoid gross over-counting.

    It remains admissible by:
    - counting only actions that are certainly required in any solution of the
      relaxed view (never counting optional cleanings/empties unless logically
      necessary)
    - using minima / optimistic assumptions for holding/handEmpty

    Implementation notes:
    - Uses only fast bitwise checks against fact.global_id.
    - Parses goal facts from model.goals via model.state_explicit_repr.
    """

    def __init__(self, model, barman_gpt52_heuristic_11: str = "barman_gpt52_heuristic_11"):
        super().__init__(model, barman_gpt52_heuristic_11)
        self.model = model
        self.name = barman_gpt52_heuristic_11

        # Cache fact ids by (pred, args...)
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_id[(pred, tuple(args))] = f.global_id

        # Cache cocktail recipe mapping: cocktail -> (part1 ingredient, part2 ingredient)
        # These are static across states.
        self._cocktail_part1: Dict[str, str] = {}
        self._cocktail_part2: Dict[str, str] = {}
        for (pred, args), _gid in self._fact_id.items():
            if pred == "cocktailPart1" and len(args) == 2:
                self._cocktail_part1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self._cocktail_part2[args[0]] = args[1]

        # Collect hands (for cheap min checks)
        self._hands: Set[str] = set()
        for (pred, args), _gid in self._fact_id.items():
            if pred in ("handEmpty", "holding"):
                if args:
                    self._hands.add(args[0])

        # Identify shaker objects (usually one)
        self._shakers: Set[str] = set()
        for (pred, args), _gid in self._fact_id.items():
            if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and args:
                self._shakers.add(args[0])

        # Goal facts (explicit) are stable; for HTN problems they are encoded in model.goals
        self._goal_fact_names: List[str] = self.model.state_explicit_repr(self.model.goals)
        self._goal_contains: List[Tuple[str, str]] = []  # (shot, cocktail)
        for n in self._goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "contains" and len(args) == 2:
                # In this domain goals are typically contains(shot,cocktail)
                self._goal_contains.append((args[0], args[1]))

    # -------------------------- core interface --------------------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        # If no explicit goals detected, fall back to 0 (uninformative but safe)
        if not self._goal_contains:
            return 0

        # For each goal shot/cocktail, compute an admissible lower bound.
        # Since all cocktails share the same shaker in provided instances, summing
        # would wildly overcount. We therefore take:
        #   h = max(per_goal_cost) + (num_missing_goals - 1)
        # The + (k-1) term captures that at least one pour per additional goal is needed,
        # but stays conservative.
        per_goal: List[int] = []
        missing_goals = 0
        for shot, cocktail in self._goal_contains:
            if self._is_true(state_bitwise, "contains", shot, cocktail):
                per_goal.append(0)
                continue
            missing_goals += 1
            per_goal.append(self._lb_make_and_pour_one(state_bitwise, shot, cocktail))

        if missing_goals == 0:
            return 0

        # Conservative aggregation for shared resources
        h = max(per_goal) if per_goal else 0

        # At least one successful pour for each missing goal; the max already includes one.
        # Additional goals require at least one more pour each (optimistic: shaker already ready).
        if missing_goals > 1:
            h += (missing_goals - 1)
        return int(h)

    # -------------------------- lower bound pieces --------------------------
    def _lb_make_and_pour_one(self, s: int, shot: str, cocktail: str) -> int:
        """Lower bound for achieving contains(shot,cocktail) from current state."""
        # If shaker already contains cocktail and is shaked, then only pouring matters.
        shaker = self._choose_some_shaker()
        if shaker is None:
            # No shaker detected -> can't reason; return 0 to stay admissible
            return 0

        if self._is_true(s, "contains", shaker, cocktail) and self._is_true(s, "shaked", shaker):
            return self._lb_pour(s, shaker, shot, cocktail)

        # Otherwise: need to make cocktail in shaker then pour.
        lb_make = self._lb_make_cocktail_in_shaker(s, shaker, cocktail)
        lb_pour = self._lb_pour(s, shaker, shot, cocktail)
        return lb_make + lb_pour

    def _lb_make_cocktail_in_shaker(self, s: int, shaker: str, cocktail: str) -> int:
        """Lower bound to reach (contains shaker cocktail) and (shaked shaker)."""
        # If already done
        if self._is_true(s, "contains", shaker, cocktail) and self._is_true(s, "shaked", shaker):
            return 0

        # Need recipe
        ing1 = self._cocktail_part1.get(cocktail)
        ing2 = self._cocktail_part2.get(cocktail)
        if ing1 is None or ing2 is None:
            # Unknown recipe -> don't overestimate
            return 0

        # If shaker currently has both ingredients and unshaked, at least one shake.
        if self._is_true(s, "contains", shaker, ing1) and self._is_true(s, "contains", shaker, ing2):
            # shake requires: holding some hand shaker, other hand empty, unshaked
            if self._is_true(s, "unshaked", shaker):
                return self._lb_hold_shaker_and_free_other_hand(s, shaker) + 1
            # If neither unshaked nor shaked known, be optimistic
            return 1

        # Otherwise, need to add missing ingredients.
        missing_ings = 0
        if not self._is_true(s, "contains", shaker, ing1):
            missing_ings += 1
        if not self._is_true(s, "contains", shaker, ing2):
            missing_ings += 1

        # Each missing ingredient requires (at least): fill-shot + pour-shot-to-*-shaker.
        # We ignore (cleaning shots / holding shots / handEmpty) because they may already hold.
        # But at least 2 primitive actions per ingredient are unavoidable in this domain.
        lb_add = 2 * missing_ings

        # After adding both, one shake is required.
        lb_shake = 1

        # Additionally, to shake we need to hold shaker and have other hand empty.
        lb_hold = self._lb_hold_shaker_and_free_other_hand(s, shaker)

        # Cleaning shaker can be required if method chooses clean-first; however adding to used shaker
        # does not require clean, and instances start clean. To remain admissible, don't force cleaning.
        return lb_add + lb_hold + lb_shake

    def _lb_pour(self, s: int, shaker: str, shot: str, cocktail: str) -> int:
        """Lower bound for pour-shaker-to-shot to achieve contains(shot,cocktail)."""
        if self._is_true(s, "contains", shot, cocktail):
            return 0

        # pour-shaker-to-shot requires: holding(hand,shaker), shaked(shaker), contains(shaker,cocktail),
        # and shot clean+empty.
        lb = 1  # at least the pour action itself

        # Ensure shot is clean and empty (if not, might need empty-shot and/or clean-shot, but we stay optimistic)
        # However, if shot is known non-empty with some beverage, then at least one empty-shot is required.
        # If shot is known not clean but empty and used, then at least one clean-shot is required.
        # These are safe lower bounds.
        if not self._is_true(s, "empty", shot):
            # if we can detect it contains something, then empty-shot is necessary.
            if self._shot_contains_any(s, shot):
                lb += 1
        if not self._is_true(s, "clean", shot):
            # clean-shot requires empty + used(shot,bev). If used is present for some bev, cleaning needed.
            if self._shot_used_any(s, shot):
                lb += 1

        # Need shaker held by some hand (optimistically 0 if already held)
        if not self._is_shaker_held_by_some_hand(s, shaker):
            # grasp requires ontable(shaker) and handEmpty(hand); we count just 1 as a LB for grasp
            lb += 1

        return lb

    def _lb_hold_shaker_and_free_other_hand(self, s: int, shaker: str) -> int:
        """LB to satisfy shake preconditions regarding hands:
        - one hand holds shaker
        - the other hand is empty
        We count minimal actions: if already satisfied => 0, else optimistic 1.
        """
        # If some hand holds shaker and some (other or same) hand is empty, we can likely arrange.
        if self._is_shaker_held_by_some_hand(s, shaker):
            # Need an empty hand distinct from holding hand; if we can prove both hands non-empty, add 1.
            if self._exists_empty_hand(s):
                return 0
            # If no empty hand known, dropping something might be needed.
            return 1

        # Not held: need at least a grasp (plus potentially freeing a hand).
        if self._exists_empty_hand(s):
            return 1
        return 2

    # -------------------------- helpers (bitwise, parsing) --------------------------
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

    def _gid(self, pred: str, *args: str) -> Optional[int]:
        return self._fact_id.get((pred, tuple(args)))

    def _is_true(self, s: int, pred: str, *args: str) -> bool:
        gid = self._gid(pred, *args)
        if gid is None:
            return False
        return ((s >> gid) & 1) == 1

    def _choose_some_shaker(self) -> Optional[str]:
        if not self._shakers:
            return None
        # Deterministic choice
        return next(iter(sorted(self._shakers)))

    def _exists_empty_hand(self, s: int) -> bool:
        for h in self._hands:
            gid = self._gid("handEmpty", h)
            if gid is not None and ((s >> gid) & 1):
                return True
        return False

    def _is_shaker_held_by_some_hand(self, s: int, shaker: str) -> bool:
        for h in self._hands:
            gid = self._gid("holding", h, shaker)
            if gid is not None and ((s >> gid) & 1):
                return True
        return False

    def _shot_contains_any(self, s: int, shot: str) -> bool:
        # Scan only facts that match contains[shot,*] by iterating cached keys.
        # This stays cheap because domain is small in predicate variety.
        for (pred, args), gid in self._fact_id.items():
            if pred == "contains" and len(args) == 2 and args[0] == shot:
                if ((s >> gid) & 1) == 1:
                    return True
        return False

    def _shot_used_any(self, s: int, shot: str) -> bool:
        for (pred, args), gid in self._fact_id.items():
            if pred == "used" and len(args) == 2 and args[0] == shot:
                if ((s >> gid) & 1) == 1:
                    return True
        return False
