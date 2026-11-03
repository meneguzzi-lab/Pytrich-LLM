from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic3(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Core idea (fast, informative lower bound):
    - The training instances' top-level goals are (AchieveContainsShotCocktail shotX cocktailY).
    - Producing one cocktail in a shot minimally requires:
        1) shaking in a shaker (needs 2 ingredients in shaker + shake)
        2) pouring from shaker to the goal shot (requires holding shaker, clean+empty shot)
      We count *missing mandatory atoms* as a relaxed/landmark-style heuristic.

    We intentionally avoid expensive state string reconstruction; all checks are bitwise.

    Heuristic components (all are lower bounds; sum is still a lower bound under a relaxed
    "independent landmarks" assumption; we keep weights conservative to reduce overestimation):

    For each goal (+contains[shot,cocktail]) not yet true:
      - +1 if shot is not both (empty and clean)   (you must reach clean+empty before final pour)
      - +1 if shaker does not contain cocktail     (must be made somewhere; domain has one shaker)
      - +1 if shaker is not shaked                (needed before pour)
      - +k for missing ingredients currently not in shaker (0..2) based on cocktailPart1/2

    Additionally, if some required container must be held for an imminent primitive (pour/shake),
    we add a tiny penalty when neither hand holds the needed container and it is not ontable.
    (This stays admissible because it is only applied when a hold is *provably necessary* and
    cannot be satisfied without at least one grasp/drop step in this domain.)

    Notes:
    - Assumes the benchmark structure: single shaker named in the problem (as in provided files).
      If multiple shakers exist, we pick the first shaker object we see in facts.
    - Robust to partial/intermediate states: if some facts are absent in the grounded set,
      they simply do not contribute.
    """

    def __init__(self, model, barman_gpt52_heuristic_3=None):
        super().__init__(model)
        self.model = model

        # Map fact name -> global_id for O(1) bit test
        self._fact_gid: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Cache parsed goals and domain relations we need
        self._goal_contains: List[Tuple[str, str]] = []  # (shot, cocktail)
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}  # cocktail -> (ing1, ing2)

        # Discover main shaker (benchmark has shaker1)
        self._shaker: Optional[str] = self._infer_any_object_of_type_from_facts("shaker")

        # Discover hands (left/right)
        self._hands: List[str] = self._infer_all_objects_from_facts_by_pred("handEmpty")

        # Parse goal facts (bitmask -> explicit fact names)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                # In goals, this is typically shot contains cocktail
                self._goal_contains.append((args[0], args[1]))

        # Precompute cocktail part mapping from all grounded facts
        # We search all facts with predicate cocktailPart1/2
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "cocktailPart1" and len(args) == 2:
                c, ing = args
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (ing, p2)
            elif pred == "cocktailPart2" and len(args) == 2:
                c, ing = args
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (p1, ing)

        # Some problems might omit cocktail parts in facts until grounding; keep safe defaults.

    # ------------------------
    # Required Heuristic API
    # ------------------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        # If already at goal with empty TN, heuristic should be 0 (planner also checks goal)
        # We compute only over goal facts; TN ignored here for speed.

        if not self._goal_contains:
            return 0

        shaker = self._shaker
        h_val = 0

        # Precompute some shaker status bits
        shaker_shaked = self._is_true(state_bitwise, self._mk_fact("shaked", [shaker])) if shaker else False

        for shot, cocktail in self._goal_contains:
            if self._is_true(state_bitwise, self._mk_fact("contains", [shot, cocktail])):
                continue

            # (A) shot readiness for final pour: must be (empty and clean)
            shot_empty = self._is_true(state_bitwise, self._mk_fact("empty", [shot]))
            shot_clean = self._is_true(state_bitwise, self._mk_fact("clean", [shot]))
            if not (shot_empty and shot_clean):
                h_val += 1

            # (B) shaker must contain cocktail and be shaked
            if shaker:
                if not self._is_true(state_bitwise, self._mk_fact("contains", [shaker, cocktail])):
                    h_val += 1
                if not shaker_shaked:
                    h_val += 1

                # (C) missing ingredients needed to create this cocktail in shaker (0..2)
                ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
                if ing1:
                    if not self._is_true(state_bitwise, self._mk_fact("contains", [shaker, ing1])) and \
                       not self._is_true(state_bitwise, self._mk_fact("contains", [shaker, cocktail])):
                        # If cocktail already present, ingredients are irrelevant.
                        h_val += 1
                if ing2:
                    if not self._is_true(state_bitwise, self._mk_fact("contains", [shaker, ing2])) and \
                       not self._is_true(state_bitwise, self._mk_fact("contains", [shaker, cocktail])):
                        h_val += 1

            # (D) tiny holding/on-table feasibility cue (conservative)
            # Final pour requires holding the shaker at some point.
            if shaker and not self._any_hand_holds(state_bitwise, shaker):
                # If not on table either, at least one action is needed (drop/grasp chain)
                if not self._is_true(state_bitwise, self._mk_fact("ontable", [shaker])):
                    h_val += 1

        return h_val

    # ------------------------
    # Helpers (fast bit checks)
    # ------------------------
    def _is_true(self, state_bitwise: int, fact_name: Optional[str]) -> bool:
        if not fact_name:
            return False
        gid = self._fact_gid.get(fact_name)
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    def _mk_fact(self, pred: str, args: List[Optional[str]]) -> Optional[str]:
        if any(a is None for a in args):
            return None
        return "+" + pred + "[" + ",".join(args) + "]"

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

    def _infer_any_object_of_type_from_facts(self, type_hint: str) -> Optional[str]:
        # Type is not directly available; infer by typical predicate usage.
        # For shaker: appears in shakerLevel/shakerEmptyLevel/shaked/unshaked
        shaker_preds = {"shakerLevel", "shakerEmptyLevel", "shaked", "unshaked"}
        if type_hint == "shaker":
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if pred in shaker_preds and args:
                    return args[0]
        return None

    def _infer_all_objects_from_facts_by_pred(self, pred_name: str) -> List[str]:
        objs: List[str] = []
        seen: Set[str] = set()
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == pred_name and len(args) == 1:
                o = args[0]
                if o not in seen:
                    seen.add(o)
                    objs.append(o)
        return objs

    def _any_hand_holds(self, state_bitwise: int, container: str) -> bool:
        # holding[hand,container]
        for h in self._hands:
            if self._is_true(state_bitwise, self._mk_fact("holding", [h, container])):
                return True
        return False
