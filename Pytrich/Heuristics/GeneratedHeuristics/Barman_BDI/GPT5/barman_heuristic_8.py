try:
    from Pytrich.Heuristics.heuristic import Heuristic
except ImportError:
    try:
        # Alternate relative import path
        from .Heuristic import Heuristic
    except ImportError:
        try:
            from heuristic import Heuristic
        except ImportError:
            # Fallback minimal base (for static checks); the planner will supply the real one at runtime
            class Heuristic:
                def __init__(self, model=None):
                    self.model = model

from typing import Dict, Tuple, List, Optional, Set


class BarmanGPT5Heuristic8(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Key ideas (admissible lower bounds):
    - Use a max-combination of independent lower bounds to avoid double-counting:
      h(n) = max(sum_costs_of_primitive_ops_in_TN, max_per_abstract_task_LB)
      This stays admissible because both are individual lower bounds on cost-to-go.
    - Provide tight, state-aware lower bounds for the domain's core abstract tasks, exploiting
      shaker/shots structure and current 'contains', 'shaked', 'unshaked', 'clean', 'empty' facts.
    - Fast bitwise state checks via global_id and cached fact indices, no explicit state materialization.

    Lower-bound patterns used (derived from the domain decomposition structure):
    - AchieveContainsShotCocktail(shot, cocktail):
        0 if already contains(shot, cocktail)
        1 if some shaker already has that cocktail and is shaked (just need the pour)
        2 if some shaker already has both required ingredients and is unshaked (shake + pour)
        5 if some shaker has exactly one of the two required ingredients (2 for fill-shot + 1 pour-to-shaker + 1 shake + 1 pour-to-shot)
        9 otherwise (two ingredients via shot [3+3], +1 shake, +1 final pour)
    - AchieveContainsShakerCocktail(shaker, cocktail):
        0 if contains(shaker, cocktail)
        1 if shaker already holds both required ingredients and is unshaked (just shake)
        4 if shaker holds exactly one of the two required ingredients (2 for fill-shot + 1 pour-to-shaker + 1 shake)
        7 otherwise (two ingredients via shot [3+3] + 1 shake; we intentionally do not add grasp cost to keep LB admissible even if shaker is already held)
    - AchieveContainsShakerIngredient(shaker, ingredient):
        0 if already contains(shaker, ingredient)
        3 otherwise (hold-shot + fill-shot + pour-shot-to-shaker)
    - AchieveContainsShotIngredient(shot, ingredient):
        0 if already contains(shot, ingredient)
        1 if the shot is currently held by any hand (only need fill-shot)
        2 otherwise (grasp + fill-shot)
    - DoPourShakerToShot(shaker, shot, cocktail):
        0 if shot already contains(cocktail)
        1 otherwise (pour-shaker-to-shot)
    - AchieveCleanShot(shot):
        0 if clean(shot)
        1 otherwise (at minimum, one clean-shot is required eventually)
    - AchieveCleanShaker(shaker):
        0 if clean(shaker)
        1 otherwise (at minimum, one clean-shaker is required eventually)
    - AchieveHolding(hand, container):
        0 if holding(hand, container)
        1 otherwise (at minimum, one grasp)
    - AchieveOnTable(container):
        0 if ontable(container)
        1 otherwise (at minimum, one drop)
    - AchieveHandEmpty(hand):
        0 if handEmpty(hand)
        1 otherwise (at minimum, one drop)

    Notes:
    - We never sum across abstract tasks to avoid overestimations due to action sharing; we take the max
      per-abstract-task LB. We separately consider the exact sum of primitive operators already present in
      the current task network (these are unavoidable costs) and combine with max() for admissibility.
    - We rely on grounded fact names in +predicate[arg1,...] format and bitwise state checks for speed.
    """

    def __init__(self, model, barman_gpt5_heuristic_8: bool = True):
        super().__init__(model)
        self.model = model
        self.enabled = barman_gpt5_heuristic_8

        # Cached indices for fast access
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self._shakers: Set[str] = set()
        self._hands: Set[str] = set()
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Build caches
        self._build_fact_index_and_metadata()

    # ------------------------------
    # Public heuristic API
    # ------------------------------
    def __call__(self, state_bitwise: int, task_network: List[object]) -> int:
        return self.evaluate(state_bitwise, task_network)

    def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
        if not self.enabled:
            return 0

        # 1) Exact lower bound from primitive operators already present in the TN: sum of their costs
        h_ops = 0
        for tn_el in task_network:
            # Robustly identify operators: they have a 'cost' attribute in Pytrich's Operator class
            if hasattr(tn_el, 'cost'):
                try:
                    h_ops += int(getattr(tn_el, 'cost', 0))
                except Exception:
                    # If cost is not properly set, ignore (keep admissible)
                    pass

        # 2) For abstract tasks, compute a domain-aware per-task LB and take the maximum
        h_tasks = 0
        for tn_el in task_network:
            if hasattr(tn_el, 'cost'):
                continue  # operator already accounted in h_ops
            # AbstractTask or unknown: use name-based dispatch with safe parsing
            name = getattr(tn_el, 'name', '')
            task_head, params = self._parse_task_name(name)
            lb = self._lower_bound_for_task(task_head, params, state_bitwise)
            if lb > h_tasks:
                h_tasks = lb

        # Admissible combination to avoid double counting between existing operators and abstract tasks
        return max(h_ops, h_tasks)

    # ------------------------------
    # Internal: task lower bounds
    # ------------------------------
    def _lower_bound_for_task(self, head: str, params: List[str], state: int) -> int:
        # Dispatch per task head (robust to missing/extra params by always checking array bounds)
        try:
            if head == 'AchieveContainsShotCocktail' and len(params) >= 2:
                shot, cocktail = params[0], params[1]
                return self._lb_achieve_contains_shot_cocktail(state, shot, cocktail)

            if head == 'AchieveContainsShakerCocktail' and len(params) >= 2:
                shaker, cocktail = params[0], params[1]
                return self._lb_achieve_contains_shaker_cocktail(state, shaker, cocktail)

            if head == 'AchieveContainsShakerIngredient' and len(params) >= 2:
                shaker, ingredient = params[0], params[1]
                # 0 if already satisfied, else 3 (hold-shot + fill + pour-to-shaker)
                if self._is_true(state, 'contains', (shaker, ingredient)):
                    return 0
                return 3

            if head == 'AchieveContainsShotIngredient' and len(params) >= 2:
                shot, ingredient = params[0], params[1]
                if self._is_true(state, 'contains', (shot, ingredient)):
                    return 0
                # If shot already held by any hand, at minimum just 1 action (fill-shot)
                if self._is_shot_held(state, shot):
                    return 1
                # Otherwise: grasp + fill-shot = 2
                return 2

            if head == 'DoPourShakerToShot' and len(params) >= 3:
                shaker, shot, cocktail = params[0], params[1], params[2]
                # If already satisfied at the target, 0; else at least 1 pour action
                if self._is_true(state, 'contains', (shot, cocktail)):
                    return 0
                return 1

            if head == 'AchieveCleanShot' and len(params) >= 1:
                shot = params[0]
                return 0 if self._is_true(state, 'clean', (shot,)) else 1

            if head == 'AchieveCleanShaker' and len(params) >= 1:
                shaker = params[0]
                return 0 if self._is_true(state, 'clean', (shaker,)) else 1

            if head == 'AchieveHolding' and len(params) >= 2:
                hand, container = params[0], params[1]
                return 0 if self._is_true(state, 'holding', (hand, container)) else 1

            if head == 'AchieveOnTable' and len(params) >= 1:
                container = params[0]
                return 0 if self._is_true(state, 'ontable', (container,)) else 1

            if head == 'AchieveHandEmpty' and len(params) >= 1:
                hand = params[0]
                return 0 if self._is_true(state, 'handEmpty', (hand,)) else 1

            # Unknown or unsupported abstract task: no contribution (keeps admissible)
            return 0
        except Exception:
            # Fail-safe: return 0 to keep admissible if anything goes wrong
            return 0

    def _lb_achieve_contains_shot_cocktail(self, state: int, shot: str, cocktail: str) -> int:
        # 0 if target shot already contains the cocktail
        if self._is_true(state, 'contains', (shot, cocktail)):
            return 0

        # 1 if a shaker already has the cocktail and is shaked (we can at least do the pour)
        if self._any_shaker_has_cocktail_shaked(state, cocktail):
            return 1

        # 2 if some shaker contains both required ingredients and is currently unshaked
        if self._any_shaker_has_both_ingredients_unshaked(state, cocktail):
            return 2  # shake + pour

        # 5 if some shaker has exactly one needed ingredient (2 fill shot + 1 pour-to-shaker + 1 shake + 1 pour)
        if self._any_shaker_has_exactly_one_ingredient(state, cocktail):
            return 5

        # Otherwise, from scratch: two ingredients via shot [3+3], +1 shake, +1 pour = 9
        return 9

    def _lb_achieve_contains_shaker_cocktail(self, state: int, shaker: str, cocktail: str) -> int:
        # 0 if shaker already contains the cocktail
        if self._is_true(state, 'contains', (shaker, cocktail)):
            return 0

        # 1 if shaker has both ingredients and is unshaked
        parts = self._cocktail_parts.get(cocktail, (None, None))
        ing1, ing2 = parts
        if ing1 and ing2:
            if self._is_true(state, 'contains', (shaker, ing1)) and \
               self._is_true(state, 'contains', (shaker, ing2)) and \
               self._is_true(state, 'unshaked', (shaker,)):
                return 1

            # 4 if it has exactly one of the two ingredients already (then: fill+pour + shake)
            has1 = self._is_true(state, 'contains', (shaker, ing1))
            has2 = self._is_true(state, 'contains', (shaker, ing2))
            if (has1 ^ has2):
                return 4

        # Otherwise, from scratch: two ingredients via shot [3+3] + 1 shake = 7
        return 7

    # ------------------------------
    # Internal: state queries over bitset
    # ------------------------------
    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        fact = self._fact_by_key.get((predicate, tuple(args)))
        if fact is None:
            return False
        return bool((state_bitwise >> fact.global_id) & 1)

    def _is_shot_held(self, state: int, shot: str) -> bool:
        # True if any hand holds this shot
        for h in self._hands:
            if self._is_true(state, 'holding', (h, shot)):
                return True
        return False

    def _any_shaker_has_cocktail_shaked(self, state: int, cocktail: str) -> bool:
        for s in self._shakers:
            if self._is_true(state, 'contains', (s, cocktail)) and self._is_true(state, 'shaked', (s,)):
                return True
        return False

    def _any_shaker_has_both_ingredients_unshaked(self, state: int, cocktail: str) -> bool:
        ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
        if not ing1 or not ing2:
            return False
        for s in self._shakers:
            if self._is_true(state, 'contains', (s, ing1)) and \
               self._is_true(state, 'contains', (s, ing2)) and \
               self._is_true(state, 'unshaked', (s,)):
                return True
        return False

    def _any_shaker_has_exactly_one_ingredient(self, state: int, cocktail: str) -> bool:
        ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
        if not ing1 or not ing2:
            return False
        for s in self._shakers:
            has1 = self._is_true(state, 'contains', (s, ing1))
            has2 = self._is_true(state, 'contains', (s, ing2))
            if has1 ^ has2:
                return True
        return False

    # ------------------------------
    # Build indices and parse helpers
    # ------------------------------
    def _build_fact_index_and_metadata(self):
        # Build (predicate, args) -> Fact mapping, and collect types info from predicate names
        for fact in getattr(self.model, 'facts', []):
            pred, args = self._parse_fact_name(getattr(fact, 'name', ''))
            if not pred:
                continue
            key = (pred, tuple(args))
            self._fact_by_key[key] = fact

            # Gather shaker names
            if pred in ('shakerLevel', 'shakerEmptyLevel', 'shaked', 'unshaked') and len(args) >= 1:
                self._shakers.add(args[0])

            # Gather hands
            if pred == 'handEmpty' and len(args) == 1:
                self._hands.add(args[0])
            elif pred == 'holding' and len(args) == 2:
                self._hands.add(args[0])

            # Gather cocktail parts
            if pred == 'cocktailPart1' and len(args) == 2:
                cocktail, ing = args[0], args[1]
                p1, p2 = self._cocktail_parts.get(cocktail, (None, None))
                self._cocktail_parts[cocktail] = (ing, p2)
            elif pred == 'cocktailPart2' and len(args) == 2:
                cocktail, ing = args[0], args[1]
                p1, p2 = self._cocktail_parts.get(cocktail, (None, None))
                self._cocktail_parts[cocktail] = (p1, ing)

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name in format +predicate[arg1,arg2]. Returns (predicate, [args...]).
        Returns (None, []) if parsing fails.
        """
        try:
            s = name.strip()
            if s.startswith('+') or s.startswith('-'):
                s = s[1:]
                bracket_idx = s.find('[')
                if bracket_idx > 0 and s.endswith(']'):
                    predicate = s[:bracket_idx]
                    args_str = s[bracket_idx + 1:-1]
                    args = [a.strip() for a in args_str.split(',')] if args_str else []
                    return predicate, args
        except Exception:
            pass
        return None, []

    @staticmethod
    def _parse_task_name(name: str) -> Tuple[str, List[str]]:
        """
        Parse task/ATN element names. Expected common formats:
          - 'AchieveContainsShotCocktail(shot2,cocktail1)'
          - 'AchieveHolding left shot1'
        Returns (head, [params...]). Falls back gracefully if format is unexpected.
        """
        if not name:
            return '', []
        name = name.strip()
        # Parenthesized format
        l = name.find('(')
        r = name.rfind(')')
        if l != -1 and r != -1 and r > l:
            head = name[:l].strip()
            inside = name[l + 1:r]
            params = [p.strip() for p in inside.split(',') if p.strip()]
            return head, params
        # Space-separated fallback
        parts = name.split()
        if not parts:
            return '', []
        head = parts[0]
        params = parts[1:]
        return head, params
