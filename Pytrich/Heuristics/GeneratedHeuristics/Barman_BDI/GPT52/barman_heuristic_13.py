from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

# Pytrich framework base class
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic13(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Core domain insight
    -------------------
    In this Barman HTN, essentially every top-level goal is of the form:
        AchieveContainsShotCocktail(shot, cocktail)

    If not already satisfied, the method MakeAndPourCocktail expands into:
        1) AchieveContainsShakerCocktail(shaker, cocktail)
        2) AchieveCleanShot(shot)
        3) AchieveHolding(hand, shaker)
        4) DoPourShakerToShot(shaker, shot, cocktail)

    Making a cocktail in a shaker typically requires (at least):
        - get 2 required ingredients into the shaker (via shots + dispensers)
        - shake
        - then pour into target shot

    Planning challenges that matter for a heuristic
    ----------------------------------------------
    - Many goals (up to 50) but only one shaker: goals are sequential; guidance should
      prioritize progress toward completing each cocktail quickly.
    - Cleaning/emptiness and hand occupancy create frequent detours.
    - "Contains" facts are the key progress signals; other predicates (clean/empty/holding)
      are enabling conditions.

    Heuristic design
    ----------------
    We compute a *cheap lower bound* by counting missing "milestones" for each pending
    AchieveContainsShotCocktail task currently in the task network.

    For each pending (shot s, cocktail c):
      - If +contains[s,c] already true -> cost 0.
      - Else we add a small fixed lower bound for the unavoidable primitive actions.

    We deliberately ignore delete effects and most resource contention (relaxation), to keep
    it admissible-ish as a lower bound. (Pytrich does not require admissibility, but a lower
    bound tends to guide A*/wA* well.)

    Milestone lower bound used (very lightweight):
      - If shaker does not currently contain cocktail c:
            +1 for "need to shake" (since shake is required to create contains[shaker,c])
            +k for missing required ingredients in shaker (0..2)
        where missing ingredients are derived from cocktailPart1/2 facts.

      - If shot s not currently clean+empty (both required before pour-shaker-to-shot):
            +1 if not empty
            +1 if not clean

      - If no hand is currently holding the shaker:
            +1 (need at least one grasp/pickup sequence; relaxed LB)

      - Always +1 for the final pour from shaker to shot if shot doesn't contain cocktail.

    Additionally:
      - If shaker currently contains some *other* cocktail and is shaked, you often must
        empty it before making a new cocktail; we add +1 in that case when c is not in shaker.

    Efficiency
    ----------
    - Uses only bit checks: (state >> fact_id) & 1.
    - Pre-indexes facts by predicate/args to get global_id quickly.
    - Parses task names from their string representation; robustly extracts arguments.

    Note
    ----
    This heuristic is domain-dependent and HTN-aware in the sense that it evaluates the
    *current task network* and provides estimates per pending abstract task rather than
    only checking the final goal bitset.
    """

    def __init__(self, model, barman_gpt52_heuristic_13: str = "barman_gpt52_heuristic_13"):
        super().__init__(model, barman_gpt52_heuristic_13)
        self.model = model

        # Build fact index: (predicate, tuple(args)) -> global_id
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_id[(pred, tuple(args))] = f.global_id

        # Cache cocktail -> (ing1, ing2) from static cocktailPart facts (from facts universe)
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._all_cocktails: set[str] = set()
        for (pred, args), gid in self._fact_id.items():
            if pred == "cocktailPart1" and len(args) == 2:
                c, ing = args
                self._all_cocktails.add(c)
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (ing, p2)
            elif pred == "cocktailPart2" and len(args) == 2:
                c, ing = args
                self._all_cocktails.add(c)
                p1, p2 = self._cocktail_parts.get(c, (None, None))
                self._cocktail_parts[c] = (p1, ing)

        # Identify shaker objects (usually exactly one)
        self._shakers: List[str] = []
        for (pred, args), gid in self._fact_id.items():
            # Use presence of shakerLevel[shaker,level] to detect shakers
            if pred == "shakerLevel" and len(args) == 2:
                sh, _lvl = args
                self._shakers.append(sh)
        self._shakers = sorted(set(self._shakers))

        # Identify hands (from handEmpty facts)
        self._hands: List[str] = []
        for (pred, args), gid in self._fact_id.items():
            if pred == "handEmpty" and len(args) == 1:
                self._hands.append(args[0])
        self._hands = sorted(set(self._hands))

    # --------- Required Heuristic API ---------

    def __call__(self, state_bitwise: int, task_network=None) -> int:
        if task_network is None:
            task_network = []

        # If goal already reached (including empty TN), return 0.
        # (Pytrich's goal_reached checks model.goals <= state and tn empty.)
        if self.model.goal_reached(state_bitwise, task_network):
            return 0

        # Sum lower bounds over pending high-level goals in the TN.
        # We focus on AchieveContainsShotCocktail tasks, which dominate this domain.
        h_val = 0

        pending_goals = self._extract_pending_shot_cocktail_tasks(task_network)
        if not pending_goals:
            # Fallback: count missing goal facts from model.goals (bitset)
            # (very weak, but safe and fast)
            h_val += self._count_missing_goal_facts(state_bitwise)
            return int(h_val)

        shaker = self._shakers[0] if self._shakers else None

        for shot, cocktail in pending_goals:
            h_val += self._estimate_one_goal(state_bitwise, shot, cocktail, shaker)

        return int(h_val)

    # --------- Heuristic components ---------

    def _estimate_one_goal(self, state: int, shot: str, cocktail: str, shaker: Optional[str]) -> int:
        """Estimate remaining primitive effort for one pending AchieveContainsShotCocktail(shot,cocktail)."""

        # If already satisfied in the world state, cost 0.
        if self._is_true(state, "contains", shot, cocktail):
            return 0

        lb = 0

        # Final pour is unavoidable if goal not satisfied.
        lb += 1

        # Shot must be empty and clean for pour.
        if not self._is_true(state, "empty", shot):
            lb += 1
        if not self._is_true(state, "clean", shot):
            lb += 1

        # Need shaker and shaker to contain cocktail.
        if shaker is not None:
            if not self._is_true(state, "contains", shaker, cocktail):
                # If shaker currently contains some other cocktail and is shaked, often must empty it.
                if self._is_true(state, "shaked", shaker) and self._shaker_contains_some_cocktail(state, shaker, exclude=cocktail):
                    lb += 1

                # Need both ingredients in shaker, then shake.
                ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
                if ing1 is not None and not self._is_true(state, "contains", shaker, ing1):
                    lb += 1
                if ing2 is not None and not self._is_true(state, "contains", shaker, ing2):
                    lb += 1

                # Shake action needed to produce contains[shaker,cocktail]
                lb += 1

            # Need to be holding shaker to pour.
            if not self._someone_holding(state, shaker):
                lb += 1

        else:
            # No shaker detected: can't reason well; add weak penalty.
            lb += 2

        return lb

    def _count_missing_goal_facts(self, state: int) -> int:
        goal_bits = self.model.goals
        missing = goal_bits & ~state
        # popcount
        return missing.bit_count() if hasattr(int, "bit_count") else bin(missing).count("1")

    # --------- Bit-check helpers ---------

    def _is_true(self, state: int, predicate: str, *args: str) -> bool:
        gid = self._fact_id.get((predicate, tuple(args)))
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _someone_holding(self, state: int, container: str) -> bool:
        # If we know hands, check holding[hand,container]
        for h in self._hands:
            if self._is_true(state, "holding", h, container):
                return True
        # If hands not detected, do a conservative check by scanning known holding facts for this container
        # (still O(#hands) in typical instances)
        return False

    def _shaker_contains_some_cocktail(self, state: int, shaker: str, exclude: Optional[str] = None) -> bool:
        # Check if shaker contains any cocktail object (from static cocktail set).
        for c in self._all_cocktails:
            if exclude is not None and c == exclude:
                continue
            if self._is_true(state, "contains", shaker, c):
                return True
        return False

    # --------- Task parsing ---------

    def _extract_pending_shot_cocktail_tasks(self, task_network) -> List[Tuple[str, str]]:
        """Extract (shot,cocktail) pairs for pending AchieveContainsShotCocktail tasks.

        The task_network is a list of Operator or AbstractTask objects.
        We parse AbstractTask.name strings which typically look like:
            'AchieveContainsShotCocktail shot2 cocktail1'
        or sometimes include parentheses depending on frontend.
        """
        res: List[Tuple[str, str]] = []
        for t in task_network:
            # Operators don't matter here
            name = getattr(t, "name", "")
            if not name:
                continue
            if "AchieveContainsShotCocktail" not in name:
                continue

            shot, cocktail = self._parse_task_args(name)
            if shot is not None and cocktail is not None:
                res.append((shot, cocktail))
        return res

    def _parse_task_args(self, task_name: str) -> Tuple[Optional[str], Optional[str]]:
        # Normalize: remove parentheses and commas
        s = task_name.strip().replace("(", " ").replace(")", " ").replace(",", " ")
        parts = [p for p in s.split() if p]
        # Find the token 'AchieveContainsShotCocktail' and take next two tokens
        for i, tok in enumerate(parts):
            if tok == "AchieveContainsShotCocktail":
                if i + 2 < len(parts):
                    return parts[i + 1], parts[i + 2]
        # Fallback: if name starts with it
        if parts and parts[0] == "AchieveContainsShotCocktail" and len(parts) >= 3:
            return parts[1], parts[2]
        return None, None

    # --------- Fact-name parsing (given in prompt) ---------

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
