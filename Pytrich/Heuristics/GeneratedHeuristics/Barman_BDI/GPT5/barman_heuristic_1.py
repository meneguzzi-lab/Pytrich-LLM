from typing import Dict, Tuple, List, Optional, Union

# Fallback Heuristic base class if import path differs in the environment
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    class Heuristic:
        def __init__(self, model):
            self.model = model


class BarmanGPT5Heuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design goals and rationale:
    - Focus on the structure of AchieveContainsShotCocktail tasks, the dominant top-level goals in provided instances.
    - Use optimistic (admissible) lower bounds derived from the domain's action structure:
      * Making a cocktail in a shaker minimally requires adding two ingredients and one shake.
        - Each missing ingredient requires at least 2 primitive actions (fill-shot + pour-to-shaker), ignoring grasp costs for admissibility.
        - Shake requires 1 action.
      * Pouring a ready cocktail from shaker to target shot requires at least 1 action (pour-shaker-to-shot) plus preparing the shot (0..2 actions).
      * We do not charge for grasp/holding in the main estimate to avoid double-counting across multiple goals and to stay safely admissible.
    - The heuristic sums lower bounds over the current task network. Each abstract task contributes a safe, optimistic cost estimate.
    - Uses only bitwise state checks for speed on large instances (per instructions).

    Covered abstract tasks with domain-aware estimates:
    - AchieveContainsShotCocktail(s, c):
      * If +contains[s,c] already true -> 0.
      * Else, if shaker already contains c and is shaked -> cost = shot_prep_cost(s) + 1 (single pour).
      * Else, missing_ingredients*2 + 1 (shake) + shot_prep_cost(s) + 1 (pour).
        - shot_prep_cost(s): 0 if clean(s) and empty(s); 1 if empty(s) and not clean(s); 2 if not empty(s).
    - AchieveContainsShakerCocktail(sh, c): 0 if already +contains[sh,c]; else (missing_ingredients*2 + 1).
    - AchieveContainsShakerIngredient(sh, i): 0 if +contains[sh,i]; else 1 (optimistic lower bound when a shot with i may already be ready and in-hand).
    - AchieveContainsShotIngredient(shot, i): 0 if +contains[shot,i]; else 1 (fill action lower bound).
    - AchieveCleanShot(shot): 0 if +clean[shot]; else 1.
    - AchieveCleanShaker(sh): 0 if +clean[sh]; else 1.
    - AchieveHolding(h, x): 0 if +holding[h,x]; else 1.
    - AchieveHandEmpty(h): 0 if +handEmpty[h]; else 1.
    - AchieveOnTable(x): 0 if +ontable[x]; else 1.
    - DoPourShakerToShot(sh, shot, c): 1 (single pour action lower bound).

    Robust parsing and state checking:
    - Facts are grounded as +predicate[arg1,arg2]. We build fast indices from Model.facts at init.
    - We parse task names flexibly (supports 'Task a b', 'Task(a,b)', 'Task[a,b]').
    - All state membership tests use bitwise checks via fact.global_id.

    Efficiency:
    - All lookups are O(1) with precomputed dictionaries; per-node evaluation is linear in the current task network size.
    """

    def __init__(self, model, barman_gpt5_heuristic_1: bool = True):
        super().__init__(model)
        self.barman_gpt5_heuristic_1 = barman_gpt5_heuristic_1

        # Indexes for fast bitwise checks
        self.fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self.contains_by_container: Dict[str, List[object]] = {}
        self.clean_fact_by_container: Dict[str, object] = {}
        self.empty_fact_by_container: Dict[str, object] = {}
        self.ontable_fact_by_container: Dict[str, object] = {}
        self.holding_facts_by_container: Dict[str, List[object]] = {}
        self.holding_facts_by_hand: Dict[str, List[object]] = {}
        self.hand_empty_fact_by_hand: Dict[str, object] = {}
        self.shaked_fact_by_shaker: Dict[str, object] = {}
        self.unshaked_fact_by_shaker: Dict[str, object] = {}
        self.contains_fact: Dict[Tuple[str, str], object] = {}

        # Cocktail to its required ingredient pair
        self.cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Gather all shakers observed in facts (usually single shaker1 in provided instances)
        self.shakers: List[str] = []

        # Build indices from model facts
        self._build_fact_indices()

        # Pick a default shaker if available
        self.default_shaker: Optional[str] = self.shakers[0] if self.shakers else None

    # -------------------- Core Heuristic API --------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        h_val = 0
        if not task_network:
            return 0

        # Iterate over current task network items (component ids or objects)
        for t in task_network:
            comp = self._get_component(t)
            if comp is None:
                continue

            # Primitive operator: at least its own cost (usually 1)
            if self._is_operator(comp):
                op_cost = getattr(comp, 'cost', 1)
                h_val += max(1, op_cost)  # Minimal primitive step
                continue

            # Abstract task: domain-specific estimate
            if self._is_abstract_task(comp):
                name = comp.name
                # Normalize name for robust prefix checks
                if not isinstance(name, str):
                    continue

                # AchieveContainsShotCocktail(shot, cocktail)
                if name.startswith('AchieveContainsShotCocktail'):
                    args = self._parse_task_args(name)
                    if len(args) >= 2:
                        shot, cocktail = args[0], args[1]
                        h_val += self._h_achieve_contains_shot_cocktail(state_bitwise, shot, cocktail)
                    else:
                        # Fallback: optimistic minimal path: 1 action
                        h_val += 1
                    continue

                # AchieveContainsShakerCocktail(shaker, cocktail)
                if name.startswith('AchieveContainsShakerCocktail'):
                    args = self._parse_task_args(name)
                    if len(args) >= 2:
                        shaker, cocktail = args[0], args[1]
                        h_val += self._h_achieve_contains_shaker_cocktail(state_bitwise, shaker, cocktail)
                    else:
                        h_val += 1  # optimistic lower bound
                    continue

                # AchieveContainsShakerIngredient(shaker, ingredient)
                if name.startswith('AchieveContainsShakerIngredient'):
                    args = self._parse_task_args(name)
                    if len(args) >= 2:
                        shaker, ingredient = args[0], args[1]
                        h_val += self._h_achieve_contains_shaker_ingredient(state_bitwise, shaker, ingredient)
                    else:
                        h_val += 1
                    continue

                # DoPourShakerToShot(shaker, shot, cocktail)
                if name.startswith('DoPourShakerToShot'):
                    # Single pour is at least 1 action
                    h_val += 1
                    continue

                # AchieveContainsShotIngredient(shot, ingredient)
                if name.startswith('AchieveContainsShotIngredient'):
                    args = self._parse_task_args(name)
                    if len(args) >= 2:
                        shot, ingredient = args[0], args[1]
                        h_val += self._h_achieve_contains_shot_ingredient(state_bitwise, shot, ingredient)
                    else:
                        h_val += 1
                    continue

                # AchieveCleanShot(shot)
                if name.startswith('AchieveCleanShot'):
                    args = self._parse_task_args(name)
                    if len(args) >= 1:
                        shot = args[0]
                        h_val += 0 if self._is_true(state_bitwise, self.clean_fact_by_container.get(shot)) else 1
                    else:
                        h_val += 1
                    continue

                # AchieveCleanShaker(shaker)
                if name.startswith('AchieveCleanShaker'):
                    args = self._parse_task_args(name)
                    if len(args) >= 1:
                        shaker = args[0]
                        h_val += 0 if self._is_true(state_bitwise, self.clean_fact_by_container.get(shaker)) else 1
                    else:
                        h_val += 1
                    continue

                # AchieveHolding(hand, container)
                if name.startswith('AchieveHolding'):
                    args = self._parse_task_args(name)
                    if len(args) >= 2:
                        hand, container = args[0], args[1]
                        holding_fact = self._get_fact(('holding', (hand, container)))
                        h_val += 0 if self._is_true(state_bitwise, holding_fact) else 1
                    else:
                        h_val += 1
                    continue

                # AchieveHandEmpty(hand)
                if name.startswith('AchieveHandEmpty'):
                    args = self._parse_task_args(name)
                    if len(args) >= 1:
                        hand = args[0]
                        h_val += 0 if self._is_true(state_bitwise, self.hand_empty_fact_by_hand.get(hand)) else 1
                    else:
                        h_val += 1
                    continue

                # AchieveOnTable(container)
                if name.startswith('AchieveOnTable'):
                    args = self._parse_task_args(name)
                    if len(args) >= 1:
                        container = args[0]
                        h_val += 0 if self._is_true(state_bitwise, self.ontable_fact_by_container.get(container)) else 1
                    else:
                        h_val += 1
                    continue

                # Unknown/other abstract task: fallback minimal optimistic
                h_val += 1
                continue

            # Unknown component type: be optimistic
            h_val += 0

        return max(0, int(h_val))

    # -------------------- Per-task heuristic helpers --------------------
    def _h_achieve_contains_shot_cocktail(self, state: int, shot: str, cocktail: str) -> int:
        # If already satisfied
        if self._is_true(state, self._get_fact(('contains', (shot, cocktail)))):
            return 0

        # Compute shot preparation lower bound (to meet pour preconditions)
        shot_prep = self._shot_prep_cost(state, shot)

        # Option A: Shaker already holds the target cocktail and is shaked -> just pour
        cost_A = None
        shaker = self._choose_shaker()
        if shaker is not None:
            if self._is_true(state, self._get_fact(('contains', (shaker, cocktail)))) \
               and self._is_true(state, self.shaked_fact_by_shaker.get(shaker)):
                # Pour requires at least 1 action; holding preconditions ignored in lower bound
                cost_A = shot_prep + 1

        # Option B: Need to make the cocktail in shaker, then pour
        # Missing ingredients (2 total): each at least 2 actions (fill + pour), shake 1, then pour 1
        missing_ing = 2
        i1, i2 = self.cocktail_parts.get(cocktail, (None, None))
        if shaker is not None and i1 is not None and i2 is not None:
            have_1 = self._is_true(state, self._get_fact(('contains', (shaker, i1))))
            have_2 = self._is_true(state, self._get_fact(('contains', (shaker, i2))))
            missing_ing = 2 - int(have_1) - int(have_2)
            if missing_ing < 0:
                missing_ing = 0
        # Lower bound: per missing ingredient 2 actions; +1 shake; +1 pour; + shot_prep
        cost_B = missing_ing * 2 + 1 + 1 + shot_prep

        if cost_A is None:
            return cost_B
        return min(cost_A, cost_B)

    def _h_achieve_contains_shaker_cocktail(self, state: int, shaker: str, cocktail: str) -> int:
        # If shaker already contains the cocktail -> 0
        if self._is_true(state, self._get_fact(('contains', (shaker, cocktail)))):
            return 0
        # Compute missing ingredients for this shaker
        i1, i2 = self.cocktail_parts.get(cocktail, (None, None))
        missing_ing = 2
        if i1 is not None and i2 is not None:
            have_1 = self._is_true(state, self._get_fact(('contains', (shaker, i1))))
            have_2 = self._is_true(state, self._get_fact(('contains', (shaker, i2))))
            missing_ing = 2 - int(have_1) - int(have_2)
            if missing_ing < 0:
                missing_ing = 0
        # Lower bound: each missing ingredient at least 2 actions (fill + pour), then 1 shake
        return missing_ing * 2 + 1

    def _h_achieve_contains_shaker_ingredient(self, state: int, shaker: str, ingredient: str) -> int:
        # If already in shaker -> 0, else at least 1 action (pour), ignoring grasp/cleanup for admissibility
        return 0 if self._is_true(state, self._get_fact(('contains', (shaker, ingredient)))) else 1

    def _h_achieve_contains_shot_ingredient(self, state: int, shot: str, ingredient: str) -> int:
        # If already in shot -> 0, else at least 1 action (fill), optimistic lower bound
        return 0 if self._is_true(state, self._get_fact(('contains', (shot, ingredient)))) else 1

    # -------------------- Parsing and indexing utilities --------------------
    def _build_fact_indices(self):
        # Build dictionaries for fast access to key predicates
        facts = getattr(self.model, 'facts', [])
        for f in facts:
            if f is None:
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self.fact_by_key[key] = f

            # Maintain convenience maps for common predicates
            if pred == 'contains' and len(args) == 2:
                container, beverage = args[0], args[1]
                self.contains_fact[(container, beverage)] = f
                self.contains_by_container.setdefault(container, []).append(f)
            elif pred == 'clean' and len(args) == 1:
                self.clean_fact_by_container[args[0]] = f
            elif pred == 'empty' and len(args) == 1:
                self.empty_fact_by_container[args[0]] = f
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_fact_by_container[args[0]] = f
            elif pred == 'holding' and len(args) == 2:
                hand, container = args[0], args[1]
                self.holding_facts_by_container.setdefault(container, []).append(f)
                self.holding_facts_by_hand.setdefault(hand, []).append(f)
            elif pred == 'handEmpty' and len(args) == 1:
                self.hand_empty_fact_by_hand[args[0]] = f
            elif pred == 'shaked' and len(args) == 1:
                self.shaked_fact_by_shaker[args[0]] = f
                if args[0] not in self.shakers:
                    self.shakers.append(args[0])
            elif pred == 'unshaked' and len(args) == 1:
                self.unshaked_fact_by_shaker[args[0]] = f
                if args[0] not in self.shakers:
                    self.shakers.append(args[0])
            elif pred == 'shakerLevel' and len(args) == 2:
                sh = args[0]
                if sh not in self.shakers:
                    self.shakers.append(sh)
            elif pred == 'shakerEmptyLevel' and len(args) == 2:
                sh = args[0]
                if sh not in self.shakers:
                    self.shakers.append(sh)

        # Cocktail parts mapping
        # cocktailPart1(cocktail, ingredient) and cocktailPart2(cocktail, ingredient)
        for f in facts:
            if f is None:
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == 'cocktailPart1' and len(args) == 2:
                c, i = args[0], args[1]
                i1, i2 = self.cocktail_parts.get(c, (None, None))
                self.cocktail_parts[c] = (i, i2)
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, i = args[0], args[1]
                i1, i2 = self.cocktail_parts.get(c, (None, None))
                self.cocktail_parts[c] = (i1, i)

    def _parse_fact_name(self, name: str):
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            core = name[1:]
            bidx = core.find('[')
            if bidx > 0 and core.endswith(']'):
                pred = core[:bidx]
                args_str = core[bidx + 1:-1]
                args = tuple(a.strip() for a in args_str.split(',')) if args_str else tuple()
                return pred, args
        return None, []

    def _parse_task_args(self, name: str) -> List[str]:
        # Robust parsing for names like:
        # 'Task a b', 'Task(a,b)', 'Task[a,b]'
        if not isinstance(name, str):
            return []
        s = name.strip()
        # Try bracket/parenthesis form first
        for lb, rb in [("(", ")"), ("[", "]")]:
            lb_idx = s.find(lb)
            if lb_idx != -1 and s.endswith(rb):
                inside = s[lb_idx + 1:-1]
                parts = [p.strip() for p in inside.split(',') if p.strip()]
                return parts
        # Fallback: space-separated
        toks = s.split()
        if len(toks) <= 1:
            return []
        return [t for t in toks[1:] if t]

    def _get_component(self, comp: Union[int, object]) -> Optional[object]:
        # If it's an ID, resolve using model; otherwise assume it's already a component
        try:
            if isinstance(comp, int):
                return self.model.get_component(comp)
            return comp
        except Exception:
            return None

    def _is_operator(self, comp: object) -> bool:
        cls_name = comp.__class__.__name__
        return cls_name == 'Operator'

    def _is_abstract_task(self, comp: object) -> bool:
        cls_name = comp.__class__.__name__
        return cls_name == 'AbstractTask'

    def _get_fact(self, key: Tuple[str, Tuple[str, ...]]) -> Optional[object]:
        return self.fact_by_key.get(key)

    def _is_true(self, state_bitwise: int, fact: Optional[object]) -> bool:
        if fact is None:
            return False
        # Fast bitwise check
        gid = getattr(fact, 'global_id', None)
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    def _choose_shaker(self) -> Optional[str]:
        # Prefer the default shaker identified at init, else any available
        if self.default_shaker is not None:
            return self.default_shaker
        return self.shakers[0] if self.shakers else None

    def _shot_prep_cost(self, state: int, shot: str) -> int:
        # Lower bound on preparing a shot to be clean and empty (required to pour into it)
        # - If clean and empty: 0
        # - If empty but not clean: at least 1 (clean-shot)
        # - If not empty: at least 2 (empty-shot + clean-shot)
        is_clean = self._is_true(state, self.clean_fact_by_container.get(shot))
        is_empty = self._is_true(state, self.empty_fact_by_container.get(shot))
        if is_clean and is_empty:
            return 0
        if is_empty:
            return 1
        return 2


# Alias for potential dynamic loading systems (optional)
HeuristicClass = BarmanGPT5Heuristic1
