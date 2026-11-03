from typing import Dict, List, Tuple, Set, Optional, Union, DefaultDict
from collections import defaultdict

# Pytrich base class import (assumed to be available in the runtime)
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimal base to avoid import errors in static analysis contexts
    class Heuristic:
        def __init__(self, model, *args, **kwargs):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
            return 0


class BarmanGPT5Heuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design rationale and domain insights:
    - Top-level goal tasks are AchieveContainsShotCocktail(shot, cocktail). To achieve this when not already true,
      the (common, minimal) plan is:
        1) Prepare the cocktail in the shaker by adding two ingredients and shaking
           (lower bound: if the shaker has none of the needed ingredients -> 5 primitives: fill + pour_to_clean + fill + pour_to_used + shake).
           If it already has both ingredients -> 1 (shake). If it has one of them -> 3 (fill + pour_to_used + shake).
           If it already contains the cocktail (shaked), prep cost is 0.
           If the shaker is empty and not clean, cleaning the shaker (clean-shaker) adds +1 (strictly necessary before pouring-to-clean-shaker).
        2) Ensure target shot is clean before pouring (lower bound: 0 if clean; 2 if it contains something; 1 if it is empty but used).
        3) Pour shaker to shot (1 primitive: pour-shaker-to-shot).

    - We compute a safe, admissible lower bound by:
        * Counting all already-present primitive operators in the task network: each must be executed (unit cost assumed).
        * For high-level abstract tasks related to cocktails, we estimate their minimal completion cost from the current state.
          To avoid double-counting, we:
            - Group AchieveContainsShotCocktail tasks by cocktail, adding the shaker cocktail preparation cost at most once per cocktail,
              then add one pour cost per remaining shot and the minimal shot cleaning cost per such shot.
            - Add costs for AchieveContainsShakerCocktail tasks only for cocktails that do not appear among the grouped shot-cocktail tasks.
            - Add costs for DoPourShakerToShot tasks that are not covered by AchieveContainsShotCocktail tasks.
        * We do not add lower bounds for other abstract tasks (e.g., AchieveHolding/HandEmpty/etc.) to avoid potential double counting with
          cocktail tasks and to keep admissibility.

    Efficiency considerations:
    - All state membership checks use fast bitwise operations via pre-indexed Fact objects (no string-in-list scans).
    - Static structures (cocktail recipes, shakers, container-to-fact indices) are precomputed once in __init__.

    Notes:
    - This heuristic is a delete-relaxation style lower bound tailored to the domain’s structure. It intentionally
      underestimates shared resource/hand logistics (grasps, drops), keeping it admissible and fast.
    - It handles scaling by grouping identical cocktail needs and reusing shaker content (prep paid once per cocktail).
    """

    def __init__(self, model, barman_gpt5_heuristic_12: Optional[dict] = None):
        super().__init__(model, barman_gpt5_heuristic_12)
        self.model = model

        # Fact indices: predicate -> args_tuple -> Fact
        self.fact_index: Dict[str, Dict[Tuple[str, ...], object]] = defaultdict(dict)

        # Container -> list of contains facts by beverage (for fast scan) and used facts
        self.contains_by_container: DefaultDict[str, Dict[str, object]] = defaultdict(dict)
        self.used_by_container: DefaultDict[str, Dict[str, object]] = defaultdict(dict)

        # Static mapping: cocktail -> (ing1, ing2)
        self.cocktail_to_ings: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Known shakers (collected from shaker-level predicates)
        self.shakers: List[str] = []
        self.default_shaker: Optional[str] = None

        # Precompute indices once
        self._index_facts()
        self._index_static_mappings()
        self._identify_shakers()

    # -----------------------------
    # Heuristic core API
    # -----------------------------
    def evaluate(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        if not task_network:
            # No tasks left; goal reached at HTN layer
            return 0

        # 1) Base lower bound: each primitive operator still in the task network must be executed at least once
        primitive_lb = 0
        abstract_tasks: List[object] = []
        for t in task_network:
            # The framework provides Operator and AbstractTask classes; use isinstance checks
            if self._is_operator(t):
                primitive_lb += 1
            elif self._is_abstract_task(t):
                abstract_tasks.append(t)

        # 2) Handle abstract tasks domain-specifically (grouped to avoid double counting on cocktails)
        # 2.1) Collect AchieveContainsShotCocktail tasks and group by cocktail for shared shaker prep
        shot_cocktail_pairs: List[Tuple[str, str]] = []
        for at in abstract_tasks:
            pred, args = self._parse_task_name(at.name)
            if pred == 'AchieveContainsShotCocktail' and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                shot_cocktail_pairs.append((shot, cocktail))

        # Group by cocktail
        by_cocktail: DefaultDict[str, List[str]] = defaultdict(list)
        for shot, cok in shot_cocktail_pairs:
            # Skip pairs already true in the state (no cost for that task)
            if not self._state_has(state_bitwise, 'contains', (shot, cok)):
                by_cocktail[cok].append(shot)

        abstract_lb = 0

        # Compute group-wise lower bound for each required cocktail
        for cok, shots in by_cocktail.items():
            if not shots:
                continue
            # prep cost: if any shaker already contains this cocktail (shaked), cost 0; else estimate prep
            prep_cost = 0
            if not self._any_shaker_has_cocktail(state_bitwise, cok):
                sh = self.default_shaker if self.default_shaker is not None else self._pick_any_shaker()
                if sh is not None:
                    prep_cost = self._estimate_prep_cost(state_bitwise, sh, cok)
                else:
                    # No shaker identified: cannot prepare; set 0 to remain admissible
                    prep_cost = 0

            # Each remaining shot needs: clean if necessary + 1 pour
            pour_and_clean_cost = 0
            for shot in shots:
                clean_c = self._estimate_shot_clean_cost(state_bitwise, shot)
                pour_and_clean_cost += 1 + clean_c

            abstract_lb += prep_cost + pour_and_clean_cost

        # 2.2) For AchieveContainsShakerCocktail tasks not covered by the above cocktail groups, group them as well
        #      (pay prep at most once per cocktail unless already present in shaker)
        shaker_cocktail_tasks: DefaultDict[str, List[str]] = defaultdict(list)  # cok -> list of shaker names (unused here)
        for at in abstract_tasks:
            pred, args = self._parse_task_name(at.name)
            if pred == 'AchieveContainsShakerCocktail' and len(args) >= 2:
                sh, cok = args[0], args[1]
                if cok not in by_cocktail:  # avoid double counting if same cocktail appears in shot-goals
                    # Also skip if state already has contains(shaker, cocktail)
                    if not self._state_has(state_bitwise, 'contains', (sh, cok)):
                        shaker_cocktail_tasks[cok].append(sh)

        for cok, sh_list in shaker_cocktail_tasks.items():
            if self._any_shaker_has_cocktail(state_bitwise, cok):
                continue
            sh = sh_list[0] if sh_list else (self.default_shaker if self.default_shaker is not None else self._pick_any_shaker())
            if sh is None:
                continue
            abstract_lb += self._estimate_prep_cost(state_bitwise, sh, cok)

        # 2.3) For DoPourShakerToShot tasks not covered by AchieveContainsShotCocktail, add 1+clean(shot)
        for at in abstract_tasks:
            pred, args = self._parse_task_name(at.name)
            if pred == 'DoPourShakerToShot' and len(args) >= 3:
                sh, shot, cok = args[0], args[1], args[2]
                # If this (shot, cok) pair is part of the AchieveContainsShotCocktail groups, it's already accounted for
                if cok in by_cocktail and shot in by_cocktail[cok]:
                    continue
                # If already satisfied, no cost
                if self._state_has(state_bitwise, 'contains', (shot, cok)):
                    continue
                # else: pour (1) + minimal cleaning for that shot
                abstract_lb += 1 + self._estimate_shot_clean_cost(state_bitwise, shot)

        # Final heuristic value
        return primitive_lb + abstract_lb

    # -----------------------------
    # Domain-specific estimators
    # -----------------------------
    def _estimate_prep_cost(self, state: int, shaker: str, cocktail: str) -> int:
        """
        Lower bound on the number of actions to make the shaker contain the target cocktail.
        Cases (delete-relaxed lower bound, ignoring grasps/hand logistics):
          - If shaker already contains the cocktail: 0.
          - If shaker already contains both required ingredients: 1 (shake).
          - If shaker already contains exactly one required ingredient: 3 (fill-shot + pour-to-used + shake).
          - Else: 5 (fill + pour-to-clean + fill + pour-to-used + shake),
                   and if the shaker is empty but not clean, add +1 to clean it before first pour-to-clean.
        """
        # Already has cocktail? -> 0
        if self._state_has(state, 'contains', (shaker, cocktail)):
            return 0

        ing1, ing2 = self.cocktail_to_ings.get(cocktail, (None, None))
        if ing1 is None or ing2 is None:
            # Missing static mapping; return 0 to remain admissible
            return 0

        has1 = self._state_has(state, 'contains', (shaker, ing1))
        has2 = self._state_has(state, 'contains', (shaker, ing2))

        # If already both ingredients inside -> 1 (shake)
        if has1 and has2:
            return 1

        # Exactly one ingredient present -> 3 (fill + pour_to_used + shake)
        if has1 or has2:
            return 3

        # No required ingredients currently in shaker: need two transfers + shake -> 5
        base = 5
        # If shaker is empty and not clean, we must clean it before the first pour-to-clean-shaker (strictly necessary)
        if self._state_has(state, 'empty', (shaker,)) and (not self._state_has(state, 'clean', (shaker,))):
            base += 1
        return base

    def _estimate_shot_clean_cost(self, state: int, shot: str) -> int:
        """
        Lower bound actions to make a shot clean (precondition needed before pouring):
          - If clean(shot): 0
          - Else if contains(shot, any_beverage): 2 (empty-shot + clean-shot)
          - Else if empty(shot) and used(shot, any_beverage): 1 (clean-shot)
          - Else: 0 (fallback: unknown corner state; keep admissible)
        """
        if self._state_has(state, 'clean', (shot,)):
            return 0

        # If it contains anything -> need to empty then clean => at least 2 actions
        if self._any_contains_for_container(state, shot):
            return 2

        # If empty and used with some beverage -> clean-shot once
        if self._any_used_for_container(state, shot):
            if self._state_has(state, 'empty', (shot,)):
                return 1

        # Unknown/degenerate: return 0 to keep admissible
        return 0

    def _any_shaker_has_cocktail(self, state: int, cocktail: str) -> bool:
        for sh in self.shakers:
            if self._state_has(state, 'contains', (sh, cocktail)):
                return True
        return False

    # -----------------------------
    # Indexing and utilities
    # -----------------------------
    def _index_facts(self):
        # Build predicate -> args -> Fact mapping and container->contains/used indices
        for f in self.model.facts:
            predicate, args = self._parse_fact_name(f.name)
            if predicate is None:
                continue
            args_t = tuple(args)
            self.fact_index[predicate][args_t] = f

            if predicate == 'contains' and len(args_t) == 2:
                container, beverage = args_t[0], args_t[1]
                self.contains_by_container[container][beverage] = f

            if predicate == 'used' and len(args_t) == 2:
                container, beverage = args_t[0], args_t[1]
                self.used_by_container[container][beverage] = f

    def _index_static_mappings(self):
        # Extract cocktail recipes: cocktailPart1/2 facts
        part1: DefaultDict[str, Set[str]] = defaultdict(set)
        part2: DefaultDict[str, Set[str]] = defaultdict(set)
        for (args_t, f) in self.fact_index.get('cocktailPart1', {}).items():
            if len(args_t) == 2:
                cok, ing = args_t[0], args_t[1]
                part1[cok].add(ing)
        for (args_t, f) in self.fact_index.get('cocktailPart2', {}).items():
            if len(args_t) == 2:
                cok, ing = args_t[0], args_t[1]
                part2[cok].add(ing)
        # Choose arbitrary representative (there should be exactly one per cocktail in provided instances)
        for cok in set(list(part1.keys()) + list(part2.keys())):
            i1 = next(iter(part1[cok])) if part1.get(cok) else None
            i2 = next(iter(part2[cok])) if part2.get(cok) else None
            self.cocktail_to_ings[cok] = (i1, i2)

    def _identify_shakers(self):
        # Identify shakers via presence in shakerLevel/shakerEmptyLevel facts
        shakers_set: Set[str] = set()
        for (args_t, f) in self.fact_index.get('shakerLevel', {}).items():
            if len(args_t) == 2:
                shakers_set.add(args_t[0])
        for (args_t, f) in self.fact_index.get('shakerEmptyLevel', {}).items():
            if len(args_t) == 2:
                shakers_set.add(args_t[0])
        self.shakers = list(shakers_set)
        self.default_shaker = self.shakers[0] if self.shakers else None

    def _any_contains_for_container(self, state: int, container: str) -> bool:
        facts = self.contains_by_container.get(container, {})
        for bev, fact in facts.items():
            if self._bit_true(state, fact.global_id):
                return True
        return False

    def _any_used_for_container(self, state: int, container: str) -> bool:
        facts = self.used_by_container.get(container, {})
        for bev, fact in facts.items():
            if self._bit_true(state, fact.global_id):
                return True
        return False

    def _state_has(self, state: int, predicate: str, args: Tuple[str, ...]) -> bool:
        f = self.fact_index.get(predicate, {}).get(tuple(args))
        return self._bit_true(state, f.global_id) if f is not None else False

    @staticmethod
    def _bit_true(state: int, bit_index: int) -> bool:
        # Fast bitwise membership test
        return ((state >> bit_index) & 1) != 0

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            posneg = name[0]
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _parse_task_name(name: str) -> Tuple[str, List[str]]:
        """
        Robust-ish parser for task/operator names. Supports formats like:
          - "AchieveContainsShotCocktail(shot2,cocktail1)"
          - "AchieveContainsShotCocktail shot2 cocktail1"
        Returns (predicate, [args]).
        """
        if not name:
            return "", []
        name = name.strip()
        if "(" in name and name.endswith(")"):
            pred = name[:name.find("(")].strip()
            inside = name[name.find("(") + 1:-1]
            args = [a.strip() for a in inside.split(",")] if inside else []
            return pred, args
        parts = name.split()
        if not parts:
            return "", []
        pred = parts[0]
        args = parts[1:]
        return pred, args

    @staticmethod
    def _is_operator(obj) -> bool:
        # Duck-type check by attributes present in Operator
        return hasattr(obj, 'pos_precons') and hasattr(obj, 'add_effects') and hasattr(obj, 'apply')

    @staticmethod
    def _is_abstract_task(obj) -> bool:
        # Duck-type check by attributes present in AbstractTask
        return hasattr(obj, 'decompositions') and hasattr(obj, 'task_network') is False if hasattr(obj, 'task_network') else hasattr(obj, 'decompositions')

    def _pick_any_shaker(self) -> Optional[str]:
        return self.shakers[0] if self.shakers else None


# Optional export alias for frameworks that import by symbol
HEURISTIC_CLASS = BarmanGPT5Heuristic12
