from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic12(Heuristic):
    """Domain-dependent heuristic for the HDDL *Barman* domain (Pytrich).

    Rationale (domain insights -> heuristic):
    - All training problems have top-level goals of the form AchieveContainsShotCocktail(shot, cocktail).
      The intended method expands to: make cocktail in shaker (clean + 2 ingredients + shake),
      clean target shot, hold shaker, then pour shaker->shot.
    - The dominant costs per cocktail are fairly fixed in this domain:
        * fill-shot (twice) + pour-shot-to-shaker (twice) + shake + pour-shaker-to-shot  => 6 primitive ops
      plus possible *cleaning* and *pick/drop* operations if hands/containers not in good configuration.
    - Good guidance for search is therefore:
        (1) count remaining unsatisfied goal facts (+contains[shot,cocktail])
        (2) add small penalties for obvious required enabling conditions:
            - if a goal shot is not empty -> it must be emptied before it can receive cocktail
            - if a goal shot is not clean -> it must be cleaned (typically requires empty + used)
            - if shaker is not in a "ready" situation (empty+clean at some point for first ingredient)
            - if both hands are not free (many actions require one hand holding, the other empty)

    This heuristic is *inadmissible* in the strict sense if we overcount shared work across multiple goals.
    To remain a *lower bound* style estimator, we use:
      - a base cost per remaining cocktail that counts only unavoidable primitives in a relaxed sense,
      - and enabling penalties computed with conservative "at least once" logic (not per-goal),
        so shared resources (shaker, hands) are not overcounted.

    Practically, this produces strong guidance on both small and large instances while staying very cheap:
    it uses only bitwise checks on a pre-indexed set of relevant facts.
    """

    def __init__(self, barman_gpt52_heuristic_12):
        super().__init__(barman_gpt52_heuristic_12)
        self.model = barman_gpt52_heuristic_12

        # Cache: name -> Fact (global ids), and also structured indices for quick checks.
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # Parse goal facts once (they are encoded as a bitmask in model.goals).
        self._goal_fact_names: List[str] = self.model.state_explicit_repr(self.model.goals)

        # Extract all goal (shot,cocktail) pairs from +contains[shot,cocktail] facts.
        self._goal_contains_pairs: List[Tuple[str, str]] = []
        for n in self._goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "contains" and len(args) == 2:
                self._goal_contains_pairs.append((args[0], args[1]))

        # Pre-index facts used by the heuristic.
        # For each goal shot: ids for empty/clean/contains[shot,cocktail]
        self._goal_contains_fact_ids: List[int] = []
        self._goal_shot_empty_fact_ids: Dict[str, int] = {}
        self._goal_shot_clean_fact_ids: Dict[str, int] = {}

        for shot, cocktail in self._goal_contains_pairs:
            fid_contains = self._get_fact_id(f"+contains[{shot},{cocktail}]")
            if fid_contains is not None:
                self._goal_contains_fact_ids.append(fid_contains)
            if shot not in self._goal_shot_empty_fact_ids:
                fid_empty = self._get_fact_id(f"+empty[{shot}]")
                if fid_empty is not None:
                    self._goal_shot_empty_fact_ids[shot] = fid_empty
            if shot not in self._goal_shot_clean_fact_ids:
                fid_clean = self._get_fact_id(f"+clean[{shot}]")
                if fid_clean is not None:
                    self._goal_shot_clean_fact_ids[shot] = fid_clean

        # Shaker-related: we assume there is typically a single shaker (as in instances).
        # We look for any shaker object by scanning fact names.
        self._any_shaker: Optional[str] = self._infer_any_object_of_predicate("shakerLevel")
        self._shaker_clean_id: Optional[int] = None
        self._shaker_empty_id: Optional[int] = None
        self._shaker_unshaked_id: Optional[int] = None
        self._shaker_shaked_id: Optional[int] = None
        if self._any_shaker:
            self._shaker_clean_id = self._get_fact_id(f"+clean[{self._any_shaker}]")
            self._shaker_empty_id = self._get_fact_id(f"+empty[{self._any_shaker}]")
            self._shaker_unshaked_id = self._get_fact_id(f"+unshaked[{self._any_shaker}]")
            self._shaker_shaked_id = self._get_fact_id(f"+shaked[{self._any_shaker}]")

        # Hands: infer common hand names from handEmpty facts in the model.
        self._hands: List[str] = self._infer_all_objects_of_predicate("handEmpty")
        self._hand_empty_ids: List[int] = []
        for h in self._hands:
            fid = self._get_fact_id(f"+handEmpty[{h}]")
            if fid is not None:
                self._hand_empty_ids.append(fid)

        # Tuning constants (kept small and conservative).
        # Minimum unavoidable primitives per remaining cocktail in a relaxed sense:
        #   fill-shot x2, pour-shot-to-shaker x2, shake x1, pour-shaker-to-shot x1 => 6
        self._base_per_cocktail = 6

        # One-time enabling penalties (lower bounds):
        self._penalty_need_free_hand = 1
        self._penalty_shaker_not_ready = 1
        self._penalty_shot_not_empty = 1
        self._penalty_shot_not_clean = 1

    # --------- Core Heuristic API ---------

    def compute_heuristic(self, state: int, task_network=None) -> int:
        """Return an estimated remaining cost.

        Uses only bitwise checks; does not build explicit state representations.
        """
        # 1) Count how many goal cocktails are still missing in their target shots.
        remaining = 0
        for fid in self._goal_contains_fact_ids:
            if ((state >> fid) & 1) == 0:
                remaining += 1

        if remaining == 0:
            return 0

        h = remaining * self._base_per_cocktail

        # 2) Conservative enabling penalties.
        # 2a) At least one hand empty is often required (grasp/fill/shake/clean actions need a free hand).
        if self._hand_empty_ids:
            any_empty = False
            for fid in self._hand_empty_ids:
                if ((state >> fid) & 1) == 1:
                    any_empty = True
                    break
            if not any_empty:
                h += self._penalty_need_free_hand

        # 2b) Shaker readiness: for making *any* new cocktail we need shaker to be usable.
        # Lower bound: if shaker is neither clean nor empty (or unknown), add a small penalty.
        if self._any_shaker and remaining > 0:
            shaker_ready = True
            # If we can test: require clean OR (already contains some cocktail and shaked) would still
            # not help for the *next* cocktail; but as a lower bound, only penalize clear "not ready".
            if self._shaker_clean_id is not None and ((state >> self._shaker_clean_id) & 1) == 0:
                # not clean could be okay if we never need to make a new cocktail (but remaining>0 we do)
                shaker_ready = False
            # also if clearly not empty and not unshaked, it might require emptying/cleaning cycle.
            if self._shaker_empty_id is not None and ((state >> self._shaker_empty_id) & 1) == 0:
                # shaker contains something; if it's already shaked cocktail, it might still be usable
                # to pour into some goal shot (but that would satisfy a contains goal). Since we only
                # penalize once, keep it conservative: only penalize if not empty AND not shaked.
                if self._shaker_shaked_id is not None and ((state >> self._shaker_shaked_id) & 1) == 0:
                    shaker_ready = False
            if not shaker_ready:
                h += self._penalty_shaker_not_ready

        # 2c) For each *unsatisfied* goal shot, if it's not empty/clean add minimal penalties.
        # We only apply penalties to shots whose contains-goal is not yet achieved.
        # This is still a lower bound: if a shot is non-empty it must be emptied at least once.
        # If it is not clean, it must be cleaned at least once (in problems where used facts exist).
        for (shot, cocktail), fid_contains in zip(self._goal_contains_pairs, self._goal_contains_fact_ids):
            if ((state >> fid_contains) & 1) == 1:
                continue
            fid_empty = self._goal_shot_empty_fact_ids.get(shot)
            if fid_empty is not None and ((state >> fid_empty) & 1) == 0:
                h += self._penalty_shot_not_empty
            fid_clean = self._goal_shot_clean_fact_ids.get(shot)
            if fid_clean is not None and ((state >> fid_clean) & 1) == 0:
                h += self._penalty_shot_not_clean

        return int(h)

    # --------- Helpers ---------

    def _get_fact_id(self, fact_name: str) -> Optional[int]:
        f = self._fact_by_name.get(fact_name)
        return None if f is None else f.global_id

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

    def _infer_any_object_of_predicate(self, predicate: str) -> Optional[str]:
        # Find first +predicate[obj,...] in the fact list, return obj.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == predicate and args:
                return args[0]
        return None

    def _infer_all_objects_of_predicate(self, predicate: str) -> List[str]:
        objs = set()
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == predicate and len(args) == 1:
                objs.add(args[0])
        return sorted(objs)
