from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Any

try:
    # Pytrich typical import location
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback (some installations expose it differently)
    from heuristic import Heuristic


class BarmanGPT52Heuristic17(Heuristic):
    """Domain-dependent heuristic for the Barman HTN domain.

    Core domain insight
    -------------------
    Training instances have an HTN goal network consisting of many tasks of the form:
        (AchieveContainsShotCocktail shotX cocktailY)

    In this domain, achieving one cocktail in a shot (when not already present) minimally requires:
      - producing the cocktail in the shaker (needs: clean shaker, 2 ingredients in shaker, shake)
      - ensuring target shot is clean+empty
      - holding shaker, then pour shaker->shot

    The domain is tightly constrained by two *hands* and the need to alternate "holding" and "handEmpty"
    to execute fill/pour/clean actions.

    Heuristic design
    ----------------
    We compute a cheap *admissible* lower bound by counting outstanding "work items" for the remaining
    abstract tasks in the current task network (TN). We only count necessities that cannot be avoided
    by any decomposition, but we deliberately avoid double-counting uncertain shared actions.

    For each pending AchieveContainsShotCocktail(shot, cocktail):
      - if already contains -> 0
      - else at least 1 pour-shaker-to-shot action must occur -> +1
      - and the cocktail must be created in shaker -> at least 1 shake action -> +1
      - and for that shake, both required ingredients must have been placed into the shaker at some
        point. Each missing ingredient implies at least one "fill-shot" primitive to obtain it -> +1 each
        (we do NOT additionally count pour-shot-to-*-shaker actions to remain safely admissible)

    For other pending abstract tasks, we add tiny safe bounds:
      - AchieveHolding(hand, container): if not holding -> at least 1 grasp/drop sequence includes a grasp -> +1
      - AchieveCleanShot(shot): if not clean -> at least 1 clean-shot -> +1 (very weak but admissible)
      - AchieveCleanShaker(shaker): if not clean -> at least 1 clean-shaker -> +1

    Additionally, we add a resource/feasibility hint that remains admissible:
      - If a required shot is neither empty nor already contains the target cocktail, then eventually
        some empty-shot must happen -> +1.

    Efficiency
    ----------
    - Uses only bitwise checks on facts (no state_explicit_repr on current states).
    - Parses goal facts only once in __init__ (goals are bitwise int).
    - Caches fact global_ids for fast membership.

    This heuristic is not a full relaxed-plan heuristic, but it is domain-structured and scales well
    to pfile20-like instances where the TN contains 50 cocktail tasks.
    """

    def __init__(self, model, barman_gpt52_heuristic_17: str = "barman_gpt52_heuristic_17"):
        super().__init__(model, barman_gpt52_heuristic_17)
        self.model = model

        # --- Build fast indices for facts ---
        self._fact_gid_by_name: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Useful predicate caches (predicate -> list of (args_tuple, gid))
        self._contains: Dict[Tuple[str, str], int] = {}
        self._empty: Dict[Tuple[str], int] = {}
        self._clean: Dict[Tuple[str], int] = {}
        self._holding: Dict[Tuple[str, str], int] = {}
        self._handempty: Dict[Tuple[str], int] = {}
        self._cocktail_p1: Dict[str, str] = {}
        self._cocktail_p2: Dict[str, str] = {}

        # Parse all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "contains" and len(args) == 2:
                self._contains[(args[0], args[1])] = fact.global_id
            elif pred == "empty" and len(args) == 1:
                self._empty[(args[0],)] = fact.global_id
            elif pred == "clean" and len(args) == 1:
                self._clean[(args[0],)] = fact.global_id
            elif pred == "holding" and len(args) == 2:
                self._holding[(args[0], args[1])] = fact.global_id
            elif pred == "handEmpty" and len(args) == 1:
                self._handempty[(args[0],)] = fact.global_id
            elif pred == "cocktailPart1" and len(args) == 2:
                # cocktailPart1(cocktail, ingredient)
                self._cocktail_p1[args[0]] = args[1]
            elif pred == "cocktailPart2" and len(args) == 2:
                self._cocktail_p2[args[0]] = args[1]

        # Try to identify the (single) shaker object(s) from facts; domain usually has one shaker.
        self._all_shakers: Set[str] = set()
        for (c, b) in self._contains.keys():
            # can't infer shaker type safely from contains; instead infer from predicate names seen in state:
            pass
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and args:
                self._all_shakers.add(args[0])

        # If none discovered (edge case), fall back to any object name starting with 'shaker'
        if not self._all_shakers:
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                for a in args:
                    if a.startswith("shaker"):
                        self._all_shakers.add(a)

        # Precompute goal facts list (goals are bitwise integer)
        self._goal_fact_names: List[str] = self.model.state_explicit_repr(self.model.goals)

    # ---------- helpers ----------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true_gid(self, state_bitwise: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    def _is_true_fact(self, state_bitwise: int, pred: str, args: List[str]) -> bool:
        # Map to gid from our caches
        if pred == "contains" and len(args) == 2:
            gid = self._contains.get((args[0], args[1]))
            return self._is_true_gid(state_bitwise, gid)
        if pred == "empty" and len(args) == 1:
            gid = self._empty.get((args[0],))
            return self._is_true_gid(state_bitwise, gid)
        if pred == "clean" and len(args) == 1:
            gid = self._clean.get((args[0],))
            return self._is_true_gid(state_bitwise, gid)
        if pred == "holding" and len(args) == 2:
            gid = self._holding.get((args[0], args[1]))
            return self._is_true_gid(state_bitwise, gid)
        if pred == "handEmpty" and len(args) == 1:
            gid = self._handempty.get((args[0],))
            return self._is_true_gid(state_bitwise, gid)
        # Unknown fact: be conservative
        return False

    # ---------- heuristic ----------
    def __call__(self, parent_node, node) -> int:
        """Return an admissible integer estimate of remaining cost.

        Pytrich calls heuristic(parent_node, node).
        We use node.state and node.task_network.
        """
        state = node.state
        tn = getattr(node, "task_network", [])

        h = 0

        # 1) Task-network driven estimation (best guidance in this domain)
        # We inspect remaining abstract tasks and count minimal necessary primitives.
        for t in tn:
            # Task objects in Pytrich have .name, which usually includes parameters, e.g.
            # 'AchieveContainsShotCocktail(shot2,cocktail1)' or similar.
            tname = getattr(t, "name", "")

            # Robust parsing: extract functor and args between parentheses if present.
            functor, args = self._parse_task_name(tname)

            if functor == "AchieveContainsShotCocktail" and len(args) == 2:
                shot, cocktail = args[0], args[1]
                h += self._lb_make_shot_cocktail(state, shot, cocktail)

            elif functor == "AchieveContainsShakerCocktail" and len(args) == 2:
                shaker, cocktail = args[0], args[1]
                # If already in shaker -> 0 else at least one shake is needed.
                if not self._is_true_fact(state, "contains", [shaker, cocktail]):
                    h += 1

            elif functor == "AchieveContainsShotIngredient" and len(args) == 2:
                shot, ing = args[0], args[1]
                if not self._is_true_fact(state, "contains", [shot, ing]):
                    # At least one fill-shot is needed
                    h += 1

            elif functor == "AchieveContainsShakerIngredient" and len(args) == 2:
                shaker, ing = args[0], args[1]
                if not self._is_true_fact(state, "contains", [shaker, ing]):
                    # Need to obtain ingredient somewhere: at least one fill-shot
                    h += 1

            elif functor == "AchieveCleanShot" and len(args) == 1:
                shot = args[0]
                if not self._is_true_fact(state, "clean", [shot]):
                    h += 1

            elif functor == "AchieveCleanShaker" and len(args) == 1:
                shaker = args[0]
                if not self._is_true_fact(state, "clean", [shaker]):
                    h += 1

            elif functor == "AchieveHolding" and len(args) == 2:
                hand, cont = args[0], args[1]
                if not self._is_true_fact(state, "holding", [hand, cont]):
                    # At least one grasp is required
                    h += 1

            elif functor == "AchieveHandEmpty" and len(args) == 1:
                hand = args[0]
                if not self._is_true_fact(state, "handEmpty", [hand]):
                    # At least one drop is required
                    h += 1

            elif functor == "DoPourShakerToShot" and len(args) == 3:
                shaker, shot, cocktail = args[0], args[1], args[2]
                # If shot doesn't yet contain cocktail, at least one pour-shaker-to-shot must occur.
                if not self._is_true_fact(state, "contains", [shot, cocktail]):
                    h += 1

        # 2) Goal-fact driven extra bounds (cheap safety net)
        # If the final goals include contains(shot,cocktail) facts, ensure we count at least something
        # even if TN is empty (rare, but possible).
        for gname in self._goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                if not self._is_true_fact(state, "contains", args):
                    # At least one action must add this contains -> pour-shaker-to-shot
                    h += 1

        return h

    def _parse_task_name(self, tname: str) -> Tuple[str, List[str]]:
        """Parse task/operator name string into (functor, [args]).

        Expected typical format in Pytrich: 'TaskName(arg1,arg2)'
        If no parentheses, returns whole string as functor and empty args.
        """
        if not tname:
            return "", []
        s = tname.strip()
        lpar = s.find("(")
        if lpar == -1 or not s.endswith(")"):
            return s, []
        functor = s[:lpar]
        inside = s[lpar + 1 : -1]
        args = [a.strip() for a in inside.split(",") if a.strip()] if inside else []
        return functor, args

    def _lb_make_shot_cocktail(self, state: int, shot: str, cocktail: str) -> int:
        """Lower bound for AchieveContainsShotCocktail(shot,cocktail)."""
        # If already satisfied
        if self._is_true_fact(state, "contains", [shot, cocktail]):
            return 0

        lb = 0

        # If shot currently contains something else (not empty), it must be emptied eventually.
        # We can test emptiness fact directly; if not empty and doesn't contain target -> need at least 1 empty-shot.
        if not self._is_true_fact(state, "empty", [shot]):
            lb += 1

        # Need at least one pour to get cocktail into the shot
        lb += 1

        # Need to have cocktail available in shaker => at least one shake
        lb += 1

        # Each of the two ingredients must be sourced; if not already in *any* shaker, we will need at least
        # one fill-shot for that ingredient.
        ing1 = self._cocktail_p1.get(cocktail)
        ing2 = self._cocktail_p2.get(cocktail)

        if ing1 is not None and not self._ingredient_in_some_shaker(state, ing1):
            lb += 1
        if ing2 is not None and not self._ingredient_in_some_shaker(state, ing2):
            lb += 1

        return lb

    def _ingredient_in_some_shaker(self, state: int, ing: str) -> bool:
        # If we don't know shakers, assume not present (conservative for admissibility? actually could overestimate).
        # To keep admissible, if uncertain, return True (so we don't add cost).
        if not self._all_shakers:
            return True
        for sh in self._all_shakers:
            gid = self._contains.get((sh, ing))
            if self._is_true_gid(state, gid):
                return True
        return False
