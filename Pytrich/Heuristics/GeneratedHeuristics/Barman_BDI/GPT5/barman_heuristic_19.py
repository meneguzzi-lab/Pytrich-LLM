from typing import Dict, Tuple, List, Set, Optional, Union

# Try importing Pytrich's base Heuristic; provide a safe fallback if not found (useful for static checks)
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:  # pragma: no cover
    class Heuristic:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return 0


class BarmanGPT5Heuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Key ideas (admissible lower bounds):
    - For primitive operators: each outstanding operator requires at least 1 step.
    - For abstract tasks, we exploit the domain's structure and available state facts:
      * AchieveContainsShotCocktail(shot, cocktail):
        - If shot already contains cocktail -> 0
        - Otherwise, we need to pour from a shaker at least once (1), and often first make the
          cocktail in a shaker. Making the cocktail requires at least:
            + One shake (to transform ingredients into the cocktail) if the shaker doesn't already contain the cocktail
            + One pour from a shot for each missing ingredient (among the two required)
            + For each missing ingredient that is not available in any shot, at least one fill-shot
        - Additionally, shot may need cleaning before pour: add 1 if it's not clean (lower bound)
        - Finally, we need the shaker to be held once for all pours of the same cocktail, so we add a
          group-level cost of 0/1 (0 if any hand is already holding a shaker, else 1). We only charge
          the "make cocktail" and this 0/1 holding once per cocktail across all tasks in the network,
          since a single shaker content can be reused to pour multiple identical cocktails.
      * AchieveContainsShakerCocktail(s, cocktail): similar lower bound to making the cocktail in a
        given shaker (ignoring hand/clean constraints):
            LB = (0 if contains(s, cocktail) else 1 shake)
               + (# of missing required ingredients currently in s)
               + (# of missing ingredients not yet available in any shot -> need fill-shot for each)
      * AchieveContainsShakerIngredient(s, i): 0 if already in shaker, else at least 1 pour; if no shot
        currently contains i, add at least 1 fill-shot.
      * AchieveContainsShotIngredient(shot, i): 0 if already in shot, else at least 1 fill-shot.
      * AchieveCleanShaker(s) / AchieveCleanShot(shot): 0 if already clean, else at least 1 action.
      * AchieveHolding(hand, container): 0 if already holding, else at least 1 grasp.
      * AchieveHandEmpty(hand): 0 if already empty, else at least 1 drop.
      * AchieveOnTable(container): 0 if already on table, else at least 1 drop.
      * DoPourShakerToShot: always at least 1 action.

    Notes on admissibility and hierarchy:
    - We always under-estimate by ignoring many preconditions (e.g., hand conflicts, ordering, emptiness)
      and count only the minimal unavoidable primitives. This yields an admissible lower bound.
    - We also aggregate costs across identical cocktails so that the "make cocktail in shaker" is charged
      at most once per cocktail, preventing overestimation when multiple shots require the same cocktail.

    Implementation details:
    - Efficient bitwise state checks via Fact.global_id.
    - Pre-index facts by (predicate, args) for O(1) lookup, parse grounded fact names of the form '+pred[arg1,arg2]'.
    - Extract object sets (shakers, containers, shots, hands) by scanning grounded facts.
    - Extract cocktail -> (ingredient1, ingredient2) mapping from static facts (cocktailPart1/2).

    Public interface:
    - __init__(model, barman_gpt5_heuristic_19=None): Prepares indexes if model is given.
    - evaluate(model, state_bitwise, task_network, depth=0): Returns h-value.
    - __call__ delegates to evaluate for planner compatibility.
    """

    def __init__(self, model=None, barman_gpt5_heuristic_19: Optional[dict] = None):
        super().__init__()
        self.model = None
        self.fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self.shakers: Set[str] = set()
        self.containers: Set[str] = set()
        self.shots: Set[str] = set()
        self.hands: Set[str] = set()
        self.cocktail_to_ingredients: Dict[str, Tuple[str, str]] = {}
        if model is not None:
            self.set_model(model)

    # ---------------------- Model preparation and parsing utilities ----------------------
    def set_model(self, model):
        self.model = model
        self._prepare_indexes()

    def _prepare_indexes(self):
        self.fact_by_key.clear()
        self.shakers.clear()
        self.containers.clear()
        self.shots.clear()
        self.hands.clear()
        self.cocktail_to_ingredients.clear()

        # Build fact index and extract typed object sets from grounded facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            # Store the positive fact object (assumed unique per ground atom)
            self.fact_by_key[key] = fact

            # Extract object roles by predicate
            if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and len(args) >= 1:
                self.shakers.add(args[0])
            if pred in ("ontable", "empty", "clean") and len(args) == 1:
                self.containers.add(args[0])
            if pred == "holding" and len(args) == 2:
                self.hands.add(args[0])
                self.containers.add(args[1])
            if pred == "handEmpty" and len(args) == 1:
                self.hands.add(args[0])
            if pred in ("contains", "used") and len(args) >= 1:
                self.containers.add(args[0])
            if pred == "cocktailPart1" and len(args) == 2:
                c, i1 = args
                self.cocktail_to_ingredients.setdefault(c, [None, None])[0] = i1
            if pred == "cocktailPart2" and len(args) == 2:
                c, i2 = args
                self.cocktail_to_ingredients.setdefault(c, [None, None])[1] = i2

        # From containers, derive shots as those that are not identified as shakers
        self.shots = set(self.containers) - set(self.shakers)

        # Convert cocktail ingredient lists to tuples and clean missing entries (robustness)
        c_map: Dict[str, Tuple[str, str]] = {}
        for c, pair in self.cocktail_to_ingredients.items():
            i1, i2 = pair
            if i1 is not None and i2 is not None:
                c_map[c] = (i1, i2)
        self.cocktail_to_ingredients = c_map

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """Parse grounded fact name of the form '+predicate[arg1,arg2]'."""
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
        bracket_idx = s.find("[")
        if bracket_idx > 0 and s.endswith("]"):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1 : -1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    @staticmethod
    def _parse_task_name(name: str) -> Tuple[str, List[str]]:
        """Parse task/operator component name 'Task(args,...)' or fallback 'Task arg1 arg2'."""
        if not isinstance(name, str):
            return "", []
        s = name.strip()
        # Parentheses style
        lb = s.find("(")
        if lb != -1 and s.endswith(")"):
            tname = s[:lb]
            args_str = s[lb + 1 : -1]
            args = args_str.split(",") if args_str else []
            return tname, args
        # Space separated fallback
        parts = s.split()
        if not parts:
            return "", []
        return parts[0], parts[1:]

    def _get_fact(self, pred: str, args: Tuple[str, ...]):
        return self.fact_by_key.get((pred, tuple(args)))

    def _is_true(self, state_bitwise: int, pred: str, args: Tuple[str, ...]) -> bool:
        f = self._get_fact(pred, args)
        if f is None:
            return False
        return ((state_bitwise >> f.global_id) & 1) == 1

    # ---------------------- Domain convenience checks ----------------------
    def _contains(self, state: int, cont: str, bev: str) -> bool:
        return self._is_true(state, "contains", (cont, bev))

    def _is_clean(self, state: int, cont: str) -> bool:
        return self._is_true(state, "clean", (cont,))

    def _is_on_table(self, state: int, cont: str) -> bool:
        return self._is_true(state, "ontable", (cont,))

    def _is_empty(self, state: int, cont: str) -> bool:
        return self._is_true(state, "empty", (cont,))

    def _holding(self, state: int, hand: str, cont: str) -> bool:
        return self._is_true(state, "holding", (hand, cont))

    def _any_hand_holds(self, state: int, cont: str) -> bool:
        for h in self.hands:
            if self._holding(state, h, cont):
                return True
        return False

    def _hand_empty(self, state: int, hand: str) -> bool:
        return self._is_true(state, "handEmpty", (hand,))

    def _exists_shot_with_ingredient(self, state: int, ingr: str, cache: Dict[str, bool]) -> bool:
        if ingr in cache:
            return cache[ingr]
        for sh in self.shots:
            if self._contains(state, sh, ingr):
                cache[ingr] = True
                return True
        cache[ingr] = False
        return False

    # Lower bound for making cocktail c in shaker s (ignoring hand/clean/order constraints)
    def _lb_make_cocktail_in_shaker(self, state: int, s: str, c: str, shot_has_ing_cache: Dict[str, bool]) -> int:
        # If shaker already contains the cocktail, no action is needed for making it
        if self._contains(state, s, c):
            return 0
        # Identify the required ingredients of the cocktail; if unknown, return 1 for at least one shake
        pair = self.cocktail_to_ingredients.get(c)
        if not pair:
            # Unknown composition -> minimally 1 action (shake)
            return 1
        i1, i2 = pair
        present = 0
        if self._contains(state, s, i1):
            present += 1
        if self._contains(state, s, i2):
            present += 1
        missing = max(0, 2 - present)
        # Each missing ingredient needs at least one pour from a shot
        lb_pours = missing
        # For each missing ingredient, if no shot has it, we need at least one fill-shot
        lb_fills = 0
        if not self._contains(state, s, i1):
            if not self._exists_shot_with_ingredient(state, i1, shot_has_ing_cache):
                lb_fills += 1
        if not self._contains(state, s, i2):
            if not self._exists_shot_with_ingredient(state, i2, shot_has_ing_cache):
                lb_fills += 1
        # One shake needed to transform the two ingredients into the cocktail
        lb_shake = 1
        return lb_shake + lb_pours + lb_fills

    # ---------------------- Main heuristic evaluation ----------------------
    def evaluate(self, model, state_bitwise: int, task_network: List[Union[object]], depth: int = 0) -> int:
        if self.model is None or self.model is not model:
            self.set_model(model)

        h = 0

        # Group AchieveContainsShotCocktail by cocktail to avoid double-counting the "make cocktail" and initial hold cost
        acc_shot_cocktail: Dict[str, List[str]] = {}  # cocktail -> list of shots in TN needing it (and not already containing it)

        # Cache: whether any shot currently contains a given ingredient
        shot_has_ing_cache: Dict[str, bool] = {}

        # First pass: collect special tasks and count other tasks
        other_tasks: List[Tuple[str, List[str]]] = []
        primitive_ops_count = 0

        for tn_item in task_network:
            name = getattr(tn_item, 'name', '')
            # Distinguish Operator vs AbstractTask by attribute presence (using planner's classes is safer but we avoid hard import)
            # Operators in Pytrich have attributes pos_precons/add_effects/etc.; AbstractTasks have decompositions
            is_operator = hasattr(tn_item, 'pos_precons') and hasattr(tn_item, 'add_effects')
            is_abstract = hasattr(tn_item, 'decompositions')

            if is_operator:
                # Each outstanding primitive operator needs at least one step
                primitive_ops_count += 1
                continue

            if is_abstract:
                tname, targs = self._parse_task_name(name)
                if tname == 'AchieveContainsShotCocktail' and len(targs) == 2:
                    shot, cocktail = targs
                    # If this shot already contains the cocktail, this task can be solved by the null method (LB 0)
                    if not self._contains(state_bitwise, shot, cocktail):
                        acc_shot_cocktail.setdefault(cocktail, []).append(shot)
                else:
                    other_tasks.append((tname, targs))
            else:
                # Unknown component type; conservatively assume at least 1
                primitive_ops_count += 1

        h += primitive_ops_count

        # Process grouped AchieveContainsShotCocktail tasks per cocktail
        for cocktail, shots_needing in acc_shot_cocktail.items():
            if not shots_needing:
                continue
            # Compute group base cost: making cocktail (in the best shaker) + one-time hold cost (0/1) in that same shaker
            group_base = 0
            if self.shakers:
                best = None
                for s in self.shakers:
                    make_cost = self._lb_make_cocktail_in_shaker(state_bitwise, s, cocktail, shot_has_ing_cache)
                    hold_cost = 0 if self._any_hand_holds(state_bitwise, s) else 1
                    total = make_cost + hold_cost
                    if best is None or total < best:
                        best = total
                group_base = best if best is not None else 0
            else:
                # No shaker recognized -> we cannot make cocktails; as LB, don't add anything here
                group_base = 0
            h += group_base

            # Per-shot lower bound: 1 pour + maybe 1 to clean shot (if not already clean)
            for shot in shots_needing:
                # We know shot doesn't contain cocktail yet (filtered above)
                pour_cost = 1
                clean_cost = 0 if self._is_clean(state_bitwise, shot) else 1
                h += pour_cost + clean_cost

        # Process other abstract tasks with local lower bounds
        for tname, targs in other_tasks:
            if tname == 'AchieveContainsShakerCocktail' and len(targs) == 2:
                s, c = targs
                # Lower bound ignoring hand/clean/order constraints
                lb = self._lb_make_cocktail_in_shaker(state_bitwise, s, c, shot_has_ing_cache)
                h += lb
            elif tname == 'AchieveContainsShakerIngredient' and len(targs) == 2:
                s, i = targs
                if not self._contains(state_bitwise, s, i):
                    lb = 1
                    # If no shot has the ingredient yet, we need at least one fill-shot
                    if not self._exists_shot_with_ingredient(state_bitwise, i, shot_has_ing_cache):
                        lb += 1
                    h += lb
            elif tname == 'AchieveContainsShotIngredient' and len(targs) == 2:
                sh, i = targs
                if not self._contains(state_bitwise, sh, i):
                    h += 1
            elif tname == 'AchieveCleanShaker' and len(targs) == 1:
                s = targs[0]
                if not self._is_clean(state_bitwise, s):
                    h += 1
            elif tname == 'AchieveCleanShot' and len(targs) == 1:
                sh = targs[0]
                if not self._is_clean(state_bitwise, sh):
                    h += 1
            elif tname == 'AchieveHolding' and len(targs) == 2:
                hand, cont = targs
                if not self._holding(state_bitwise, hand, cont):
                    h += 1
            elif tname == 'AchieveHandEmpty' and len(targs) == 1:
                hand = targs[0]
                if not self._hand_empty(state_bitwise, hand):
                    h += 1
            elif tname == 'AchieveOnTable' and len(targs) == 1:
                cont = targs[0]
                if not self._is_on_table(state_bitwise, cont):
                    h += 1
            elif tname == 'DoPourShakerToShot' and len(targs) == 3:
                # Must ultimately perform the pour; at least 1
                h += 1
            else:
                # Unknown/other abstract task: conservatively count 1 to keep guidance without overestimation
                h += 1

        return h

    # Alias for planner compatibility
    def __call__(self, model, state_bitwise: int, task_network: List[Union[object]], depth: int = 0) -> int:
        return self.evaluate(model, state_bitwise, task_network, depth)
