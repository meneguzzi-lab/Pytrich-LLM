from typing import Dict, Tuple, List, Set, Optional

try:
    # Normal import path inside Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        # Alternate relative path (depending on runner layout)
        from .Heuristic import Heuristic
    except Exception:
        # Fallback stub (for static analysis or docs generation only). The framework will provide the real class.
        class Heuristic:  # type: ignore
            pass


class BarmanGPT5Heuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design summary (admissible lower bound using domain structure):
    - We estimate the minimal remaining number of primitive actions needed to complete the current task network.
    - We provide state-dependent lower bounds for key abstract tasks using domain knowledge:
        * AchieveContainsShotCocktail(shot, cocktail)
            - If already contains -> 0
            - Else: choose a shaker s and plan: make cocktail in s (2 pours + 1 shake, plus creating ingredients in shots if needed),
              ensure the target shot is clean if needed, then pour shaker->shot (1). We take the cheapest s.
        * AchieveContainsShakerCocktail(shaker, cocktail)
            - If contains -> 0
            - Else: clean shaker if needed (LB), add both ingredients (each: possibly create in a shot via fill (1) plus pour to shaker (1)), then shake (1).
        * AchieveContainsShakerIngredient(shaker, ingredient)
            - If contains -> 0
            - Else: choose a shot that already contains ingredient (0) or can be made to contain it (fill 1 + minimal cleaning LB) and add 1 (pour to shaker).
        * AchieveContainsShotIngredient(shot, ingredient)
            - If contains -> 0
            - Else: minimal to prepare the shot (clean LB) plus 1 fill-shot.
        * AchieveCleanShot(shot)
            - If clean -> 0
            - Else if contains something -> 2 (empty-shot + clean-shot)
            - Else if empty and used -> 1 (clean-shot)
            - Else: 0 (very conservative for odd states, keeps admissible)
        * AchieveCleanShaker(shaker)
            - If clean -> 0
            - Else if empty -> 1 (clean-shaker)
            - Else if contains cocktail and shaked -> 2 (empty-shaker + clean-shaker)
            - Else: 0 (conservative in pre-shake states to keep admissible)
        * DoPourShakerToShot(shaker, shot, cocktail): if shot already contains cocktail -> 0, else 1 (pour)
        * AchieveHolding(hand, container): if already holding -> 0 else 1 (grasp)
        * AchieveOnTable(container): if ontable -> 0 else 1 (drop)
        * AchieveHandEmpty(hand): if handEmpty -> 0 else 1 (drop)
    - Primitive operators in the task network each contribute 1 (they still need to be executed).

    Implementation notes:
    - Uses bitwise checks for state membership via fact.global_id (fast), never converts state to explicit lists per node.
    - Pre-indexes all relevant facts at initialization for O(1) or small-O checks per query.
    - Parses grounded task names (e.g., 'AchieveContainsShotCocktail[shot2,cocktail1]') to extract arguments.

    Safety and admissibility:
    - We only count mandatory primitive actions that must occur, possibly underestimating due to ignoring some ordering/hand-holding costs.
    - This keeps the heuristic as a lower bound (admissible) while still providing strong guidance on large instances.
    """

    def __init__(self, model=None, barman_gpt5_heuristic_17: bool = True):
        # Configuration flag kept for compatibility with framework's constructor signature requirements
        self.enabled: bool = bool(barman_gpt5_heuristic_17)

        self.model = None

        # Indices for fast bitwise checks
        self.contains_index: Dict[Tuple[str, str], int] = {}
        self.clean_index: Dict[str, int] = {}
        self.empty_index: Dict[str, int] = {}
        self.used_index: Dict[Tuple[str, str], int] = {}
        self.used_by_container: Dict[str, List[int]] = {}
        self.shaked_index: Dict[str, int] = {}
        self.unshaked_index: Dict[str, int] = {}
        self.ontable_index: Dict[str, int] = {}
        self.handEmpty_index: Dict[str, int] = {}
        self.holding_index: Dict[Tuple[str, str], int] = {}
        self.shakerLevel_index: Dict[Tuple[str, str], int] = {}
        self.shakerEmptyLevel_index: Dict[Tuple[str, str], int] = {}
        self.dispenses_index: Dict[Tuple[str, str], int] = {}

        # Object enumerations
        self.all_shakers: Set[str] = set()
        self.all_shots: Set[str] = set()
        self.all_hands: Set[str] = set()
        self.all_cocktails: Set[str] = set()
        self.all_ingredients: Set[str] = set()

        # Cocktail -> set(ingredients)
        self.cocktail_parts: Dict[str, Set[str]] = {}

        # For quick 'any contains' checks
        self.contains_by_container: Dict[str, List[int]] = {}

        if model is not None:
            self._precompute(model)

    # Public API: support multiple common entry points used by the planner
    def __call__(self, model, state_bitwise: int, task_network: List) -> int:
        if not self.enabled:
            return 0
        if model is not self.model:
            self._precompute(model)
        return self.compute(model, state_bitwise, task_network)

    def evaluate(self, model, state_bitwise: int, task_network: List) -> int:
        return self.__call__(model, state_bitwise, task_network)

    def h(self, model, state_bitwise: int, task_network: List) -> int:
        return self.__call__(model, state_bitwise, task_network)

    # ------------------------------------------------------------------
    # Precomputation and utilities
    # ------------------------------------------------------------------
    def _precompute(self, model):
        self.model = model
        # Reset indices
        self.contains_index.clear()
        self.clean_index.clear()
        self.empty_index.clear()
        self.used_index.clear()
        self.used_by_container.clear()
        self.shaked_index.clear()
        self.unshaked_index.clear()
        self.ontable_index.clear()
        self.handEmpty_index.clear()
        self.holding_index.clear()
        self.shakerLevel_index.clear()
        self.shakerEmptyLevel_index.clear()
        self.dispenses_index.clear()
        self.all_shakers.clear()
        self.all_shots.clear()
        self.all_hands.clear()
        self.all_cocktails.clear()
        self.all_ingredients.clear()
        self.cocktail_parts.clear()
        self.contains_by_container.clear()

        # Scan facts and build indices
        for fact in model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'contains' and len(args) == 2:
                self.contains_index[(args[0], args[1])] = fact.global_id
                self.contains_by_container.setdefault(args[0], []).append(fact.global_id)
            elif pred == 'clean' and len(args) == 1:
                self.clean_index[args[0]] = fact.global_id
            elif pred == 'empty' and len(args) == 1:
                self.empty_index[args[0]] = fact.global_id
            elif pred == 'used' and len(args) == 2:
                self.used_index[(args[0], args[1])] = fact.global_id
                self.used_by_container.setdefault(args[0], []).append(fact.global_id)
            elif pred == 'shaked' and len(args) == 1:
                self.shaked_index[args[0]] = fact.global_id
            elif pred == 'unshaked' and len(args) == 1:
                self.unshaked_index[args[0]] = fact.global_id
            elif pred == 'ontable' and len(args) == 1:
                self.ontable_index[args[0]] = fact.global_id
            elif pred == 'handEmpty' and len(args) == 1:
                self.handEmpty_index[args[0]] = fact.global_id
                self.all_hands.add(args[0])
            elif pred == 'holding' and len(args) == 2:
                self.holding_index[(args[0], args[1])] = fact.global_id
                self.all_hands.add(args[0])
            elif pred == 'shakerLevel' and len(args) == 2:
                self.shakerLevel_index[(args[0], args[1])] = fact.global_id
                self.all_shakers.add(args[0])
            elif pred == 'shakerEmptyLevel' and len(args) == 2:
                self.shakerEmptyLevel_index[(args[0], args[1])] = fact.global_id
                self.all_shakers.add(args[0])
            elif pred == 'dispenses' and len(args) == 2:
                self.dispenses_index[(args[0], args[1])] = fact.global_id
                self.all_ingredients.add(args[1])
            elif pred == 'cocktailPart1' and len(args) == 2:
                self.cocktail_parts.setdefault(args[0], set()).add(args[1])
                self.all_cocktails.add(args[0])
                self.all_ingredients.add(args[1])
            elif pred == 'cocktailPart2' and len(args) == 2:
                self.cocktail_parts.setdefault(args[0], set()).add(args[1])
                self.all_cocktails.add(args[0])
                self.all_ingredients.add(args[1])

        # Deduce shots = containers that can be clean but are not shakers
        clean_containers = set(self.clean_index.keys())
        self.all_shots = clean_containers - self.all_shakers

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """Parse grounded fact name of format '+predicate[arg1,arg2]' (or '-') into (predicate, [args])."""
        if not name:
            return None, []
        n = name.strip()
        if n.startswith('+') or n.startswith('-'):
            n = n[1:]
        b = n.find('[')
        if b > 0 and n.endswith(']'):
            pred = n[:b]
            args_str = n[b + 1:-1]
            args = args_str.split(',') if args_str else []
            return pred, args
        return n, []

    @staticmethod
    def _parse_task_or_op_name(name: str) -> Tuple[str, List[str]]:
        """Parse grounded task/operator name like 'AchieveContainsShotCocktail[shot2,cocktail1]'."""
        if not name:
            return name, []
        n = name.strip()
        b = n.find('[')
        if b > 0 and n.endswith(']'):
            head = n[:b]
            args_str = n[b + 1:-1]
            args = args_str.split(',') if args_str else []
            return head, args
        return n, []

    def _is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _any_contains(self, state_bitwise: int, container: str) -> bool:
        ids = self.contains_by_container.get(container, [])
        for fid in ids:
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    def _any_used(self, state_bitwise: int, container: str) -> bool:
        ids = self.used_by_container.get(container, [])
        for fid in ids:
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    # ------------------------------------------------------------------
    # Lower-bound estimators for subgoals (domain-specific)
    # ------------------------------------------------------------------
    def _lb_clean_shot(self, state: int, shot: str) -> int:
        # If already clean -> 0
        if self._is_true(state, self.clean_index.get(shot)):
            return 0
        # If shot currently contains something -> need empty-shot (1) + clean-shot (1)
        if self._any_contains(state, shot):
            return 2
        # If empty and used -> can clean-shot (1)
        if self._is_true(state, self.empty_index.get(shot)) and self._any_used(state, shot):
            return 1
        # Otherwise (odd states) return 0 to stay conservative/admissible
        return 0

    def _lb_contains_shot_ingredient(self, state: int, shot: str, ing: str) -> int:
        # If already contains ingredient
        if self._is_true(state, self.contains_index.get((shot, ing))):
            return 0
        # Need to be able to perform fill-shot: minimal prep (clean/empty) + 1 action fill
        prep = 0
        if self._is_true(state, self.clean_index.get(shot)) and self._is_true(state, self.empty_index.get(shot)):
            prep = 0
        elif self._any_contains(state, shot):
            prep = 2  # empty + clean
        elif self._is_true(state, self.empty_index.get(shot)) and self._any_used(state, shot):
            prep = 1  # clean only
        else:
            prep = 0  # conservative fallback
        return prep + 1  # +1 for fill-shot

    def _lb_contains_shaker_ingredient(self, state: int, shaker: str, ing: str) -> int:
        # If shaker already contains the ingredient
        if self._is_true(state, self.contains_index.get((shaker, ing))):
            return 0
        # Otherwise, we need a shot that contains the ingredient (or can be made to), and pour to shaker (1)
        best_shot_cost = None
        for shot in self.all_shots:
            c = self._lb_contains_shot_ingredient(state, shot, ing)
            if best_shot_cost is None or c < best_shot_cost:
                best_shot_cost = c
        if best_shot_cost is None:
            # No shot known (shouldn't occur in provided instances); be safe but non-negative
            best_shot_cost = 1
        return best_shot_cost + 1  # +1 for pour-shot-to-*-shaker

    def _lb_clean_shaker(self, state: int, shaker: str) -> int:
        if self._is_true(state, self.clean_index.get(shaker)):
            return 0
        # If empty and not clean -> 1 clean-shaker
        if self._is_true(state, self.empty_index.get(shaker)):
            return 1
        # If contains a cocktail and is shaked -> need to empty (1) + clean (1)
        # We detect 'contains cocktail' by checking contains[shaker, c] for any known cocktail c
        contains_cocktail = False
        for c in self.all_cocktails:
            if self._is_true(state, self.contains_index.get((shaker, c))):
                contains_cocktail = True
                break
        if contains_cocktail and self._is_true(state, self.shaked_index.get(shaker)):
            return 2
        # Pre-shake with ingredients only is complicated; stay conservative
        return 0

    def _lb_contains_shaker_cocktail(self, state: int, shaker: str, cocktail: str) -> int:
        if self._is_true(state, self.contains_index.get((shaker, cocktail))):
            return 0
        # Ingredients required by the cocktail
        ings = list(self.cocktail_parts.get(cocktail, set()))
        # Defensive: ensure we consider up to 2 ingredients
        if len(ings) == 0:
            # Unknown cocktail structure; return a small positive lower bound to provide guidance
            return 1
        elif len(ings) == 1:
            ing1 = ings[0]
            ing2 = ings[0]
        else:
            ing1, ing2 = ings[0], ings[1]

        lb = self._lb_clean_shaker(state, shaker)
        lb += self._lb_contains_shaker_ingredient(state, shaker, ing1)
        lb += self._lb_contains_shaker_ingredient(state, shaker, ing2)
        lb += 1  # shake action
        return lb

    def _lb_pour_shaker_to_shot(self, state: int, shaker: str, shot: str, cocktail: str) -> int:
        if self._is_true(state, self.contains_index.get((shot, cocktail))):
            return 0
        return 1  # pour-shaker-to-shot

    def _lb_contains_shot_cocktail(self, state: int, shot: str, cocktail: str) -> int:
        # If shot already has the cocktail
        if self._is_true(state, self.contains_index.get((shot, cocktail))):
            return 0
        # For each shaker, consider: make cocktail in shaker + clean shot LB + pour to shot
        best = None
        clean_shot_lb = self._lb_clean_shot(state, shot)
        for shaker in self.all_shakers:
            make_lb = self._lb_contains_shaker_cocktail(state, shaker, cocktail)
            pour_lb = self._lb_pour_shaker_to_shot(state, shaker, shot, cocktail)
            total = make_lb + clean_shot_lb + pour_lb
            if best is None or total < best:
                best = total
        if best is None:
            # No shaker known (edge case) -> at least 1 action remains somewhere
            best = 1
        return best

    # ------------------------------------------------------------------
    # Main heuristic computation
    # ------------------------------------------------------------------
    def compute(self, model, state_bitwise: int, task_network: List) -> int:
        if not task_network:
            # Also consider standard fact-goal bitset (often empty for HTN instances)
            goals_left = 0
            if model.goals:
                # Count how many goal bits are not yet set in the state
                missing = model.goals & (~state_bitwise)
                # Lower bound: at least 1 per missing fact (very conservative)
                # But to keep consistent with domain-specific approach, just count bits
                goals_left = missing.bit_count() if hasattr(int, 'bit_count') else self._popcount(missing)
            return goals_left

        hval = 0
        for node in task_network:
            # If primitive operator: each costs at least 1 to execute
            # Avoid dynamic import checks; rely on class name comparison
            clsname = node.__class__.__name__
            if clsname == 'Operator':
                hval += 1
                continue

            # Abstract task: inspect its grounded name
            name, args = self._parse_task_or_op_name(getattr(node, 'name', ''))
            if not name:
                continue

            # Map known abstract tasks to LB calculators
            if name == 'AchieveContainsShotCocktail' and len(args) == 2:
                shot, cocktail = args
                hval += self._lb_contains_shot_cocktail(state_bitwise, shot, cocktail)
            elif name == 'AchieveContainsShakerCocktail' and len(args) == 2:
                shaker, cocktail = args
                hval += self._lb_contains_shaker_cocktail(state_bitwise, shaker, cocktail)
            elif name == 'AchieveContainsShakerIngredient' and len(args) == 2:
                shaker, ing = args
                hval += self._lb_contains_shaker_ingredient(state_bitwise, shaker, ing)
            elif name == 'AchieveContainsShotIngredient' and len(args) == 2:
                shot, ing = args
                hval += self._lb_contains_shot_ingredient(state_bitwise, shot, ing)
            elif name == 'AchieveCleanShot' and len(args) == 1:
                shot = args[0]
                hval += self._lb_clean_shot(state_bitwise, shot)
            elif name == 'AchieveCleanShaker' and len(args) == 1:
                shaker = args[0]
                hval += self._lb_clean_shaker(state_bitwise, shaker)
            elif name == 'DoPourShakerToShot' and len(args) == 3:
                shaker, shot, cocktail = args
                hval += self._lb_pour_shaker_to_shot(state_bitwise, shaker, shot, cocktail)
            elif name == 'AchieveHolding' and len(args) == 2:
                hand, container = args
                if not self._is_true(state_bitwise, self.holding_index.get((hand, container))):
                    hval += 1  # grasp
            elif name == 'AchieveOnTable' and len(args) == 1:
                container = args[0]
                if not self._is_true(state_bitwise, self.ontable_index.get(container)):
                    hval += 1  # drop
            elif name == 'AchieveHandEmpty' and len(args) == 1:
                hand = args[0]
                if not self._is_true(state_bitwise, self.handEmpty_index.get(hand)):
                    hval += 1  # drop
            else:
                # Unknown/other abstract task: return small admissible lower bound (0)
                # This is conservative and safe.
                hval += 0

        return hval

    @staticmethod
    def _popcount(x: int) -> int:
        # Fallback popcount if Python <3.8 (no int.bit_count)
        count = 0
        while x:
            x &= x - 1
            count += 1
        return count
