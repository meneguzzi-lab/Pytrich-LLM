from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic1(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Rationale (domain insights -> heuristic design):
    - Training instances have HTN goals of the form AchieveContainsShotCocktail(shotX, cocktailY).
      In this domain, producing one cocktail in a shot minimally requires a fixed *pipeline*:
        1) Have shaker contain the cocktail (requires: clean shaker, add two ingredients via shots, shake)
        2) Ensure target shot is clean+empty
        3) Hold shaker and pour shaker -> shot
      This yields a strong *lower bound* by counting missing necessary conditions.

    - The main resource/constraint is hands and holding/ontable, but in the provided instances
      hands start empty and everything is on the table and clean/empty. Therefore, a cheap and
      effective estimate is to count the remaining "must-achieve" facts for each demanded cocktail.

    - We keep the heuristic computationally light:
        * No state->explicit fact list conversion (uses bit checks).
        * Precomputes goal cocktail pairs and cocktail ingredients from static facts.

    Heuristic value:
    - For each required (shot, cocktail) pair:
        * If already contains(shot,cocktail): 0
        * Else add a conservative lower bound (inadmissible risk avoided by using ONLY necessary
          achievements and using min-cost assumptions):

      We estimate remaining primitive steps with a relaxed "count of missing landmarks" model.
      Each missing landmark contributes 1, and we cap/structure it so it remains a lower bound.

      Landmarks used (all are necessary in any valid plan for unsatisfied goal):
        A) shaker must contain(cocktail) at some point before pour (unless shot already contains it)
        B) shaker must be shaked (precondition to pour and to create cocktail)
        C) target shot must be empty AND clean before pour
        D) shaker must have its two ingredients before shake

      We map these to facts:
        - contains(shaker1,cocktail)
        - shaked(shaker1)
        - clean(shot), empty(shot)
        - contains(shaker1, ing1), contains(shaker1, ing2)  (from cocktailPart1/2)

    Note:
    - Instances provided have exactly one shaker (shaker1); we exploit this for speed.
      If multiple shakers exist, we pick the first seen shaker as the "main" one.

    This heuristic is not guaranteed admissible in all pathological states (due to interactions
    between multiple cocktails and shaker reuse), but it is a strong domain-guidance heuristic
    for reducing node expansions in the given benchmark style (many independent cocktail goals).
    """

    def __init__(self, model, barman_gpt52_heuristic_1=None):
        super().__init__(model, barman_gpt52_heuristic_1)
        self.model = model

        # --- Build fast lookup: fact_name -> Fact(global_id) ---
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # --- Identify a shaker to reason about (benchmarks: shaker1 exists) ---
        self._main_shaker: Optional[str] = None
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "ontable" and args and args[0].startswith("shaker"):
                self._main_shaker = args[0]
                break
        if self._main_shaker is None:
            # fallback: any object named shaker1 might not appear in ontable
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if args and any(a.startswith("shaker") for a in args):
                    self._main_shaker = next(a for a in args if a.startswith("shaker"))
                    break

        # --- Extract goal contains(shot, cocktail) facts from model.goals ---
        self._goal_pairs: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "contains" and len(args) == 2:
                shot, bev = args
                if shot.startswith("shot") and bev.startswith("cocktail"):
                    self._goal_pairs.append((shot, bev))

        # --- Precompute cocktail -> (ing1, ing2) using static facts cocktailPart1/2 ---
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "cocktailPart1" and len(args) == 2:
                c, ing = args
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (ing, p2)
            elif pred == "cocktailPart2" and len(args) == 2:
                c, ing = args
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (p1, ing)

        # Cache Fact objects for frequently queried predicates to avoid dict overhead in loops
        self._fact_cache: Dict[Tuple[str, Tuple[str, ...]], Optional[int]] = {}

    # ------------------------- helpers -------------------------
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

    def _fact_gid(self, predicate: str, *args: str) -> Optional[int]:
        key = (predicate, tuple(args))
        if key in self._fact_cache:
            return self._fact_cache[key]
        fname = f"+{predicate}[{','.join(args)}]"
        fact = self._fact_by_name.get(fname)
        gid = fact.global_id if fact is not None else None
        self._fact_cache[key] = gid
        return gid

    def _is_true(self, state: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    # ------------------------- heuristic -------------------------
    def __call__(self, state: int, task_network=None) -> int:
        # If no relevant goals extracted, fall back to 0 (uninformative but safe)
        if not self._goal_pairs:
            return 0

        shaker = self._main_shaker
        if shaker is None:
            return 0

        h = 0

        gid_shaked = self._fact_gid("shaked", shaker)

        for shot, cocktail in self._goal_pairs:
            # If goal already satisfied in current state, no cost for this pair
            gid_goal = self._fact_gid("contains", shot, cocktail)
            if self._is_true(state, gid_goal):
                continue

            # Necessary: shot must be clean and empty before pour
            if not self._is_true(state, self._fact_gid("clean", shot)):
                h += 1
            if not self._is_true(state, self._fact_gid("empty", shot)):
                h += 1

            # Necessary: shaker must end up with cocktail and be shaked to allow pouring
            gid_shaker_has_cocktail = self._fact_gid("contains", shaker, cocktail)
            if not self._is_true(state, gid_shaker_has_cocktail):
                h += 1

            if not self._is_true(state, gid_shaked):
                # shaked is a strict precondition for pouring
                h += 1

            # Necessary: cocktail ingredients must have been in shaker at some point before shake.
            # In relaxed landmark counting, we count missing ingredient-in-shaker facts.
            ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
            if ing1 is not None:
                if not self._is_true(state, self._fact_gid("contains", shaker, ing1)):
                    h += 1
            if ing2 is not None:
                if not self._is_true(state, self._fact_gid("contains", shaker, ing2)):
                    h += 1

        # Very small tie-break encouragement: if hands are not empty, add 1.
        # (Often needed to grasp/fill/shake/pour, but we keep it tiny.)
        # This is guidance, not a strong lower bound.
        if task_network is not None:
            # detect any handEmpty fact; if none true, nudge
            left_empty = self._is_true(state, self._fact_gid("handEmpty", "left"))
            right_empty = self._is_true(state, self._fact_gid("handEmpty", "right"))
            if not (left_empty or right_empty):
                h += 1

        return h
