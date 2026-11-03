from typing import Dict, Tuple, List, Set, Optional

# Heuristic base is provided by the framework
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback for some environments
    class Heuristic:
        def __init__(self, *args, **kwargs):
            pass


class BarmanGPT5Heuristic16(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design goals and rationale:
    - Carefully exploits the domain's structure: a cocktail is made by adding two specific
      ingredients to a shaker and then shaking; pouring into shots requires the shot to be
      empty and clean, and the shaker to be shaked and containing the cocktail.
    - Shares work across multiple AchieveContainsShotCocktail tasks of the same cocktail:
      preparing the cocktail in the shaker once can serve many pours (large instances).
    - Admissible lower bound (we ignore hand/grasp/ordering subtleties and count only the
      absolutely necessary primitive effects), yet informative enough to prune search.

    Key lower-bound approximations (all ignore hand/holding requirements to remain safe):
    - Making cocktail in shaker from scratch (no required ingredients currently in shaker and
      shaker not shaked with a different cocktail):
        * For each of the 2 missing ingredients: at least one fill-shot (1) + one pour-to-shaker (1) = 2
        * One shake (1)
      => 2*2 + 1 = 5 as the strictly necessary action effects; we conservatively add 0-1 extra for
         minimal robustness if shaker is currently shaked with some other cocktail (requires empty): +1
      In code, this is computed more precisely using the current state's partial progress
      (existing required ingredients already in the shaker), and whether the shaker is shaked.

    - Preparing a target shot for pour (to meet clean & empty):
        * If already clean & empty: 0
        * If contains something: empty (1) + clean (1) = 2
        * If empty but dirty: clean (1) = 1

    - Pouring from shaker to shot: at least 1 primitive action.

    Grouping and sharing:
    - For each distinct cocktail present in the remaining task network (AchieveContainsShotCocktail
      and/or AchieveContainsShakerCocktail), we count the shaker-preparation ("make") cost at most once.
      Every requested shot of that cocktail only adds the per-shot preparation and pour cost.
    - We do NOT charge inter-cocktail switch costs (e.g., empty/clean between different cocktails),
      which keeps the heuristic admissible and scalable.

    Robustness:
    - Fully bitwise checks for state membership using model's facts global_id.
    - Parses task/operator names to extract bound arguments.

    Expected interface:
    - __call__(model, state_bitwise, task_network) -> non-negative int lower bound.
    - evaluate(...) and h(...) as aliases for compatibility with different planners.

    Notes:
    - This heuristic is domain-dependent, but remains lightweight: no search, purely arithmetic
      over the current (state, tn) with precomputed indices from the model.
    """

    def __init__(self, barman_gpt5_heuristic_16: bool = True):
        super().__init__()
        # Runtime caches
        self._indexed_model_id: Optional[int] = None
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._facts_parsed: bool = False

        # Domain-specific indices
        self._cocktail_parts: Dict[str, Tuple[str, str]] = {}          # cocktail -> (ing1, ing2)
        self._contains_by_beverage: Dict[str, List[Tuple[str, int]]] = {}  # bev -> list of (container, fact_id)
        self._contains_by_container: Dict[str, List[int]] = {}         # container -> list of fact_ids (contains[container,*])
        self._shakers: Set[str] = set()
        self._hands: Set[str] = set()

    # --------------------------- Public API ---------------------------
    def __call__(self, model, state_bitwise: int, task_network: List) -> int:
        return self.evaluate(model, state_bitwise, task_network)

    def evaluate(self, model, state_bitwise: int, task_network: List) -> int:
        # Goals in HTN problems here are typically empty; success requires empty TN.
        if getattr(model, 'goal_reached', None):
            try:
                if model.goal_reached(state_bitwise, task_network):
                    return 0
            except Exception:
                pass

        self._ensure_indexes(model)

        # If there are multiple shakers, pick an arbitrary one for estimates; in provided instances it's 1.
        shaker_name = next(iter(self._shakers)) if self._shakers else None

        # Group AchieveContainsShotCocktail by cocktail to share shaker preparation cost.
        cocktail_to_shots: Dict[str, List[str]] = {}
        # Additional tasks (shaker/shot/ingredient primitives) are handled per-task.

        h_val = 0
        # Count primitive operators in TN as 1 each (they must be executed)
        for t in task_network:
            # Operator tasks are primitive => add exact cost 1 as LB
            if hasattr(t, 'pos_precons') and hasattr(t, 'add_effects'):
                h_val += 1

        # First pass: collect grouped top-level cocktail goals, and also collect other abstract tasks
        other_abstract_tasks: List[Tuple[str, List[str]]] = []  # (task_name, args)
        for t in task_network:
            if hasattr(t, 'task_network') or hasattr(t, 'decompositions'):
                tname, targs = self._parse_task_or_op_name(getattr(t, 'name', ''))
                if tname == 'AchieveContainsShotCocktail' and len(targs) == 2:
                    shot, cocktail = targs
                    cocktail_to_shots.setdefault(cocktail, []).append(shot)
                else:
                    other_abstract_tasks.append((tname, targs))

        # Account once per distinct cocktail: shaker preparation (if necessary)
        counted_cocktails: Set[str] = set()
        for cocktail, shots in cocktail_to_shots.items():
            if shaker_name is not None:
                make_cost = self._lb_make_cocktail_in_shaker(model, state_bitwise, shaker_name, cocktail)
                if make_cost > 0:
                    h_val += make_cost
            # per-shot: prepare and pour
            for shot in shots:
                h_val += self._lb_prep_shot_for_pour(model, state_bitwise, shot) + 1  # +1 pour
            counted_cocktails.add(cocktail)

        # Handle other abstract tasks with specific LBs
        for tname, targs in other_abstract_tasks:
            if tname == 'AchieveContainsShakerCocktail' and len(targs) == 2:
                shaker, cocktail = targs
                # If this cocktail already counted via AchieveContainsShotCocktail group, do not double count
                if cocktail in counted_cocktails:
                    continue
                h_val += self._lb_make_cocktail_in_shaker(model, state_bitwise, shaker, cocktail)
            elif tname == 'AchieveContainsShakerIngredient' and len(targs) == 2:
                shaker, ing = targs
                h_val += self._lb_add_ingredient_to_shaker(model, state_bitwise, shaker, ing)
            elif tname == 'AchieveContainsShotIngredient' and len(targs) == 2:
                shot, ing = targs
                # If already contains, 0; else at least one fill-shot (ignore grasp/hand as LB)
                if not self._is_true(model, state_bitwise, 'contains', (shot, ing)):
                    h_val += 1
            elif tname == 'AchieveCleanShot' and len(targs) == 1:
                shot = targs[0]
                h_val += self._lb_clean_shot(model, state_bitwise, shot)
            elif tname == 'AchieveCleanShaker' and len(targs) == 1:
                shaker = targs[0]
                h_val += self._lb_clean_shaker(model, state_bitwise, shaker)
            elif tname == 'DoPourShakerToShot' and len(targs) == 3:
                shaker, shot, cocktail = targs
                # At least one pour + preparation of the shot for pour
                h_val += 1 + self._lb_prep_shot_for_pour(model, state_bitwise, shot)
            elif tname == 'AchieveHolding' and len(targs) == 2:
                hand, container = targs
                if not self._is_true(model, state_bitwise, 'holding', (hand, container)):
                    h_val += 1
            elif tname == 'AchieveOnTable' and len(targs) == 1:
                container = targs[0]
                if not self._is_true(model, state_bitwise, 'ontable', (container,)):
                    h_val += 1
            elif tname == 'AchieveHandEmpty' and len(targs) >= 1:
                hand = targs[0]
                if not self._is_true(model, state_bitwise, 'handEmpty', (hand,)):
                    h_val += 1
            else:
                # Unknown/other abstract task: give zero (keeps admissible)
                pass

        # Ensure non-negative
        return max(0, int(h_val))

    # Backwards compatibility aliases
    def h(self, model, state_bitwise: int, task_network: List) -> int:
        return self.evaluate(model, state_bitwise, task_network)

    # ------------------------ Internal helpers ------------------------
    def _ensure_indexes(self, model):
        # Re-index per model instance
        mid = id(model)
        if self._indexed_model_id == mid and self._facts_parsed:
            return

        # Reset
        self._fact_by_key.clear()
        self._cocktail_parts.clear()
        self._contains_by_beverage.clear()
        self._contains_by_container.clear()
        self._shakers.clear()
        self._hands.clear()

        # Build indices over facts
        for fact in getattr(model, 'facts', []):
            name = getattr(fact, 'name', '')
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_by_key[key] = fact.global_id

            if pred == 'cocktailPart1' and len(args) == 2:
                c, i1 = args
                i2 = self._cocktail_parts.get(c, (None, None))[1]
                self._cocktail_parts[c] = (i1, i2)
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, i2 = args
                i1 = self._cocktail_parts.get(c, (None, None))[0]
                self._cocktail_parts[c] = (i1, i2)
            elif pred == 'contains' and len(args) == 2:
                cont, bev = args
                self._contains_by_beverage.setdefault(bev, []).append((cont, fact.global_id))
                self._contains_by_container.setdefault(cont, []).append(fact.global_id)
            elif pred in ('shakerLevel', 'shakerEmptyLevel') and len(args) == 2:
                cont = args[0]
                self._shakers.add(cont)
            elif pred == 'handEmpty' and len(args) == 1:
                self._hands.add(args[0])

        self._indexed_model_id = mid
        self._facts_parsed = True

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name of format: +predicate[arg1,arg2]
        Returns (predicate, [args...]) or (None, []) on failure.
        """
        try:
            name = name.strip()
            if name.startswith('+') or name.startswith('-'):
                name2 = name[1:]
                b = name2.find('[')
                if b > 0 and name2.endswith(']'):
                    pred = name2[:b]
                    args_str = name2[b+1:-1]
                    args = args_str.split(',') if args_str else []
                    return pred, args
        except Exception:
            pass
        return None, []

    def _parse_task_or_op_name(self, name: str) -> Tuple[str, List[str]]:
        """
        Parse operator/abstract task name: e.g., AchieveContainsShotCocktail(shot2,cocktail1)
        Returns (task_name, [args...]). If not parsable, returns (name, []).
        """
        try:
            name = name.strip()
            i = name.find('(')
            if i == -1 or not name.endswith(')'):
                return name, []
            task = name[:i]
            args_str = name[i+1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return task, args
        except Exception:
            return name, []

    def _fact_id(self, model, pred: str, args: Tuple[str, ...]) -> Optional[int]:
        return self._fact_by_key.get((pred, args))

    def _is_true(self, model, state_bitwise: int, pred: str, args: Tuple[str, ...]) -> bool:
        fid = self._fact_id(model, pred, args)
        if fid is None:
            return False
        return ((state_bitwise >> fid) & 1) == 1

    def _container_contains_any(self, model, state_bitwise: int, container: str) -> bool:
        ids = self._contains_by_container.get(container, [])
        for fid in ids:
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    def _any_shot_contains(self, model, state_bitwise: int, beverage: str, exclude_containers: Set[str]) -> bool:
        lst = self._contains_by_beverage.get(beverage, [])
        for cont, fid in lst:
            if cont in exclude_containers:
                continue
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    # ---------------------- Domain LB components ----------------------
    def _lb_prep_shot_for_pour(self, model, state_bitwise: int, shot: str) -> int:
        """
        Minimal actions to ensure shot is clean and empty (required for pour-shaker-to-shot).
        Ignore hand/holding requirements for admissibility.
        - If already clean & empty => 0
        - If contains something => need empty (1); likely also dirty, and pour precond needs clean => +1 => 2
        - Else if empty but dirty => clean (1)
        - Else if clean but not empty => empty (1)
        """
        clean = self._is_true(model, state_bitwise, 'clean', (shot,))
        empty = self._is_true(model, state_bitwise, 'empty', (shot,))
        contains_any = self._container_contains_any(model, state_bitwise, shot)
        if clean and empty:
            return 0
        cost = 0
        if contains_any:
            cost += 1  # empty-shot
        if not clean:
            cost += 1  # clean-shot
        if not contains_any and clean and not empty:
            cost = max(cost, 1)  # need at least one empty if only non-empty missing
        return cost

    def _lb_clean_shot(self, model, state_bitwise: int, shot: str) -> int:
        """
        Minimal actions to have (clean shot):
        - If already clean: 0
        - If contains something: empty (1) + clean (1) = 2
        - Else (empty but dirty): clean (1) = 1
        """
        if self._is_true(model, state_bitwise, 'clean', (shot,)):
            return 0
        if self._container_contains_any(model, state_bitwise, shot):
            return 2
        return 1

    def _lb_clean_shaker(self, model, state_bitwise: int, shaker: str) -> int:
        """
        Minimal actions to have (clean shaker):
        - If already clean: 0
        - If contains something: need empty (1) + clean (1) = 2
        - Else (empty but dirty): clean (1) = 1
        """
        if self._is_true(model, state_bitwise, 'clean', (shaker,)):
            return 0
        if self._container_contains_any(model, state_bitwise, shaker):
            return 2
        return 1

    def _lb_add_ingredient_to_shaker(self, model, state_bitwise: int, shaker: str, ing: str) -> int:
        """
        Minimal actions to ensure shaker contains a given ingredient (ignoring method constraints):
        - If already contains: 0
        - Else: need at least one pour from some shot (1). If no shot currently contains the ingredient,
          we must also fill a shot with that ingredient first (1). If shaker is shaked (contains some
          cocktail), we need to empty it once to be able to proceed (1). We ignore cleaning.
        """
        if self._is_true(model, state_bitwise, 'contains', (shaker, ing)):
            return 0
        cost = 0
        if self._is_true(model, state_bitwise, 'shaked', (shaker,)):
            cost += 1  # need to empty at least once to unshake
        # pour from shot to shaker
        cost += 1
        # if no shot already has ing, need to fill a shot with ing first
        if not self._any_shot_contains(model, state_bitwise, ing, exclude_containers={shaker}):
            cost += 1
        return cost

    def _lb_make_cocktail_in_shaker(self, model, state_bitwise: int, shaker: str, cocktail: str) -> int:
        """
        Minimal actions to ensure shaker contains the given cocktail (ignoring grasp/hand constraints):
        Cases:
        - If already contains shaker,cocktail => 0.
        - Else compute missing-ingredients cost + possibly one reset if shaker is shaked (different cocktail).
        - Let m = number of required ingredients currently present in shaker (when unshaked).
          For each missing ingredient: need at least pour from some shot (1) + if no shot currently
          contains it then fill such a shot (1). After all required are in, need one shake (1).
        """
        if self._is_true(model, state_bitwise, 'contains', (shaker, cocktail)):
            return 0

        parts = self._cocktail_parts.get(cocktail, (None, None))
        i1, i2 = parts
        if i1 is None or i2 is None:
            # Missing static knowledge: be optimistic but safe
            return 1

        cost = 0
        # If shaker currently shaked (i.e., has some cocktail), we need at least one empty to proceed
        if self._is_true(model, state_bitwise, 'shaked', (shaker,)):
            cost += 1  # empty-shaker lower bound

        # Count how many required ingredients are already in shaker
        have_i1 = self._is_true(model, state_bitwise, 'contains', (shaker, i1))
        have_i2 = self._is_true(model, state_bitwise, 'contains', (shaker, i2))
        have_cnt = int(have_i1) + int(have_i2)
        missing = max(0, 2 - have_cnt)

        # For each missing ingredient: at least 1 pour; add 1 fill if nobody currently contains it
        def per_missing_cost(ing: str) -> int:
            base = 1  # pour-shot-to-(clean/used)-shaker
            if not self._any_shot_contains(model, state_bitwise, ing, exclude_containers={shaker}):
                base += 1  # need a fill-shot with that ingredient
            return base

        if not have_i1:
            cost += per_missing_cost(i1)
        if not have_i2:
            cost += per_missing_cost(i2)

        # Need one shake to transform ingredients to the cocktail
        cost += 1
        return cost


# Factory function (optional depending on framework conventions)
def get_heuristic(barman_gpt5_heuristic_16: bool = True):
    return BarmanGPT5Heuristic16(barman_gpt5_heuristic_16=barman_gpt5_heuristic_16)
