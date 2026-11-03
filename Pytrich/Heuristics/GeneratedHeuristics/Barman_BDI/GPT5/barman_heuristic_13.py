from typing import Dict, List, Tuple, Set, Optional

# Attempt to import the planner's Heuristic base class from common paths
try:
    from Heuristics.heuristic import Heuristic  # typical Pytrich path
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic  # fallback
    except Exception:
        # Minimal fallback to avoid import-time crash if path differs during static checks.
        class Heuristic:  # type: ignore
            def __init__(self, model, *args, **kwargs):
                self.model = model
            def evaluate(self, state_bitwise: int, task_network: List):
                return 0


class BarmanGPT5Heuristic13(Heuristic):
    """
    Domain-dependent HTN heuristic for the Barman domain.

    Key ideas (lower-bound, domain-aware, efficient):
    - Extract the first pending task and compute a sharp, admissible lower bound on the
      remaining cost-to-go using domain structure: shaker preparation, ingredient acquisition,
      shaking, and pouring.
    - Use fast bitwise state checks via precomputed Fact global_ids.
    - Handle both high-level tasks (Achieve*) and primitive actions. Fall back conservatively
      on unknown tasks.

    Core LB estimates (admissible):
    - AchieveContainsShotCocktail(shot,cocktail):
        * If shot already contains cocktail -> 0.
        * Else if shaker already contains cocktail and is shaked ->
            LB = 1 (pour) + [1 if not holding shaker] + [1 if shot not clean/empty]
        * Else ->
            LB = LB(AchieveContainsShakerCocktail) + [1 if shot not clean/empty] + 1 (pour)
          (Assume holding of shaker established inside shaker-cocktail preparation; don't double-count.)
    - AchieveContainsShakerCocktail(shaker,cocktail):
        * If shaker already contains cocktail -> 0.
        * Else:
            lb = LB(AchieveCleanShaker)                                           # forced by method hierarchy
               + sum over the two required ingredients i1,i2 of LB(AddToShaker(i))
               + [1 if not holding shaker] + 1 (shake)
          where LB(AddToShaker(i)) accounts for reusing an already full shot if available, otherwise filling a shot.
    - AchieveCleanShaker(shaker):
        * If clean -> 0
        * Else if shaker contains anything -> 2 (empty-shaker + clean-shaker)
        * Else -> 1 (clean-shaker)
    - AchieveContainsShakerIngredient(shaker,ingredient): lower bound of getting an ingredient into the shaker
        * If any shot contains that ingredient and is held -> 1 (pour to shaker)
        * Else if any shot contains that ingredient -> 2 (grasp + pour)
        * Else -> 3 (grasp + fill + pour)
      (Cleaning the shot etc. is ignored here for a lower bound; existence of at least one clean empty shot is
       true initially; underestimating keeps admissibility.)
    - DoPourShakerToShot(shaker,shot,cocktail): 1 (pour) + [1 if not holding shaker] + [1 if shot not clean/empty]
    - Simpler Achieve* tasks:
        * AchieveHolding(hand,container): 0 if already holding else 1
        * AchieveHandEmpty(hand): 0 if handEmpty else 1
        * AchieveCleanShot(shot): 0 if clean else 1
        * AchieveOnTable(container): 0 if ontable else 1
    - Primitive operator: 1

    Notes:
    - We respect the HTN structure: AchieveContainsShakerCocktail forces cleaning prior to mixing in the domain
      methods; our LB includes AchieveCleanShaker.
    - All checks are state-dependent and use bitwise operations for scalability.
    - The heuristic focuses on the first task in the current task network (ordered tasks) to preserve admissibility
      while still offering strong guidance.
    """

    def __init__(self, model, barman_gpt5_heuristic_13: bool = True):
        super().__init__(model)
        self.enabled = barman_gpt5_heuristic_13

        # Fast positive-fact lookup: (pred, tuple(args)) -> global_id
        self.fact_pos_map: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Precomputed indices/sets for quick queries
        self.hands: Set[str] = set()
        self.shakers: Set[str] = set()
        self.default_shaker: Optional[str] = None
        self.ingredients: Set[str] = set()
        self.cocktails: Set[str] = set()

        # Map cocktail -> (part1, part2)
        self.cocktail_parts: Dict[str, Tuple[str, str]] = {}

        # For each ingredient: list of (shot_name, fact_id of +contains[shot,ingredient])
        self.contains_shot_ing_ids: Dict[str, List[Tuple[str, int]]] = {}

        # For each container: list of fact_ids of +holding[hand,container]
        self.holding_ids_by_container: Dict[str, List[int]] = {}

        # For a shaker: list of fact_ids for contains[shaker,*] (any beverage)
        self.contains_shaker_any_ids: Dict[str, List[int]] = {}

        # Build all indices from grounded facts
        self._index_facts()

    # -----------------------------
    # Heuristic main entry point
    # -----------------------------
    def evaluate(self, state_bitwise: int, task_network: List) -> int:
        try:
            if not self.enabled:
                return 0
            if state_bitwise is None:
                return 0
            if task_network is None or len(task_network) == 0:
                return 0

            # Use the first pending task/operator (ordered HTN)
            first = task_network[0]
            name = getattr(first, 'name', str(first))
            head, args = self._parse_component_name(name)

            # Primitive operators (domain actions): cost at least 1
            if self._is_operator_name(head):
                return 1

            # Abstract tasks (domain-specific)
            if head == 'AchieveContainsShotCocktail' and len(args) == 2:
                shot, cocktail = args[0], args[1]
                shaker = self._pick_shaker()
                return self._lb_achieve_shot_cocktail(state_bitwise, shot, cocktail, shaker)

            if head == 'AchieveContainsShakerCocktail' and len(args) == 2:
                shaker, cocktail = args[0], args[1]
                return self._lb_achieve_shaker_cocktail(state_bitwise, shaker, cocktail)

            if head == 'AchieveContainsShakerIngredient' and len(args) == 2:
                shaker, ingredient = args[0], args[1]
                return self._lb_add_ingredient_to_shaker(state_bitwise, shaker, ingredient)

            if head == 'DoPourShakerToShot' and len(args) == 3:
                shaker, shot, cocktail = args[0], args[1], args[2]
                return self._lb_do_pour_shaker_to_shot(state_bitwise, shaker, shot, cocktail)

            if head == 'AchieveCleanShot' and len(args) == 1:
                shot = args[0]
                return 0 if self._is_true(state_bitwise, 'clean', (shot,)) else 1

            if head == 'AchieveCleanShaker' and len(args) == 1:
                shaker = args[0]
                return self._lb_achieve_clean_shaker(state_bitwise, shaker)

            if head == 'AchieveHolding' and len(args) == 2:
                hand, container = args[0], args[1]
                return 0 if self._is_true(state_bitwise, 'holding', (hand, container)) else 1

            if head == 'AchieveHandEmpty' and len(args) == 1:
                hand = args[0]
                return 0 if self._is_true(state_bitwise, 'handEmpty', (hand,)) else 1

            if head == 'AchieveOnTable' and len(args) == 1:
                cnt = args[0]
                return 0 if self._is_true(state_bitwise, 'ontable', (cnt,)) else 1

            # Unknown/other abstract task: be conservative but non-zero to guide search
            return 1
        except Exception:
            # Be safe: do not break the planner because of heuristic; default to 0 (admissible)
            return 0

    # -----------------------------
    # Lower-bound calculators
    # -----------------------------
    def _lb_achieve_shot_cocktail(self, state: int, shot: str, cocktail: str, shaker: Optional[str]) -> int:
        # If already satisfied
        if self._is_true(state, 'contains', (shot, cocktail)):
            return 0

        # If no shaker known, assume at least 1 to make progress (very conservative)
        if shaker is None:
            return 1

        # Case: shaker already has the cocktail and is shaked -> just pour (and potentially grasp/clean shot)
        if self._is_true(state, 'contains', (shaker, cocktail)) and self._is_true(state, 'shaked', (shaker,)):
            lb = 1  # pour-shaker-to-shot
            if not self._any_hand_holding_container(state, shaker):
                lb += 1  # need to grasp shaker
            if (not self._is_true(state, 'clean', (shot,))) or (not self._is_true(state, 'empty', (shot,))):
                lb += 1  # need shot prepared (empty & clean) at least 1 action
            return lb

        # Otherwise: we must prepare the shaker cocktail first (includes cleaning per HTN method), then pour
        lb = 0
        lb += self._lb_achieve_shaker_cocktail(state, shaker, cocktail)

        # AchieveCleanShot for the target shot (0 if already good, else 1)
        if (not self._is_true(state, 'clean', (shot,))) or (not self._is_true(state, 'empty', (shot,))):
            lb += 1

        # Finally, pour to the shot.
        # We assume AchieveContainsShakerCocktail sequence leaves us holding the shaker before shaking, and we can
        # reuse that for pouring; thus we only add the pour operator cost.
        lb += 1
        return lb

    def _lb_achieve_shaker_cocktail(self, state: int, shaker: str, cocktail: str) -> int:
        # If shaker already contains the cocktail -> 0
        if self._is_true(state, 'contains', (shaker, cocktail)):
            return 0

        parts = self.cocktail_parts.get(cocktail)
        if not parts:
            # No knowledge about this cocktail's ingredients; return a small positive LB to guide
            # (still admissible but weak)
            parts = (None, None)

        lb = 0
        # Clean before mixing (forced by MakeCocktail method)
        lb += self._lb_achieve_clean_shaker(state, shaker)

        # Add both ingredients
        for ing in parts:
            if ing is None:
                continue
            lb += self._lb_add_ingredient_to_shaker(state, shaker, ing)

        # Need to hold shaker and shake (at least 1 action to shake; +1 if currently not holding shaker)
        if not self._any_hand_holding_container(state, shaker):
            lb += 1
        lb += 1  # shake action

        # Optionally account for needing an empty hand to shake; as an LB we can safely add 0 if any hand is empty
        # If no hand is empty now, at least one drop would be needed eventually.
        if not self._any_hand_empty(state):
            # Add 1 to account for having to empty a hand before shaking; still a lower bound.
            lb += 1

        return lb

    def _lb_add_ingredient_to_shaker(self, state: int, shaker: str, ingredient: str) -> int:
        # If the shaker already has this ingredient, we can count 0; it will be consumed by shake.
        if self._is_true(state, 'contains', (shaker, ingredient)):
            return 0

        # If some shot already contains the ingredient
        shot_list = self.contains_shot_ing_ids.get(ingredient, [])
        held_exists = False
        present_exists = False
        for shot_name, fact_id in shot_list:
            if (state >> fact_id) & 1:
                present_exists = True
                if self._any_hand_holding_container(state, shot_name):
                    held_exists = True
                    break

        if held_exists:
            # Just pour-shot-to-...: 1 action (ignoring shaker cleanliness for LB)
            return 1
        if present_exists:
            # Need to grasp that shot then pour: 2 actions
            return 2

        # Otherwise, we must fill a shot then pour -> grasp + fill + pour: 3 actions
        return 3

    def _lb_do_pour_shaker_to_shot(self, state: int, shaker: str, shot: str, cocktail: str) -> int:
        lb = 1  # the pour operator itself
        if not self._any_hand_holding_container(state, shaker):
            lb += 1
        if (not self._is_true(state, 'clean', (shot,))) or (not self._is_true(state, 'empty', (shot,))):
            lb += 1
        return lb

    def _lb_achieve_clean_shaker(self, state: int, shaker: str) -> int:
        # Already clean -> 0
        if self._is_true(state, 'clean', (shaker,)):
            return 0
        # If shaker contains anything -> at least empty-shaker + clean-shaker = 2 actions (ignoring grasps)
        if self._shaker_contains_any(state, shaker):
            return 2
        # Otherwise, only clean-shaker is needed
        return 1

    # -----------------------------
    # Parsing and State helpers
    # -----------------------------
    def _is_operator_name(self, head: str) -> bool:
        # Domain actions (primitive)
        return head in {
            'clean-shaker', 'clean-shot', 'drop', 'empty-shaker', 'empty-shot',
            'fill-shot', 'grasp', 'pour-shaker-to-shot', 'pour-shot-to-clean-shaker', 'pour-shot-to-used-shaker',
            'shake'
        }

    def _parse_component_name(self, name: str) -> Tuple[str, List[str]]:
        # Supports forms like Head[arg1,arg2] or Head(arg1,arg2) or 'Head arg1 arg2'
        name = name.strip()
        # Bracket style
        lb, rb = name.find('['), name.rfind(']')
        if lb != -1 and rb != -1 and rb > lb:
            head = name[:lb].strip()
            args_str = name[lb + 1:rb].strip()
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return head, args
        # Parentheses style
        lp, rp = name.find('('), name.rfind(')')
        if lp != -1 and rp != -1 and rp > lp:
            head = name[:lp].strip()
            args_str = name[lp + 1:rp].strip()
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return head, args
        # Space-separated fallback
        toks = name.replace(')', '').replace(']', '').split()
        if len(toks) >= 1:
            return toks[0], toks[1:]
        return name, []

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        # Format: +predicate[arg1,arg2,...]
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name2 = name[1:]
            lb = name2.find('[')
            if lb > 0 and name2.endswith(']'):
                pred = name2[:lb]
                args_str = name2[lb + 1:-1]
                args = args_str.split(',') if args_str else []
                return pred, [a.strip() for a in args]
        return None, []

    def _is_true(self, state: int, pred: str, args: Tuple[str, ...]) -> bool:
        fid = self.fact_pos_map.get((pred, tuple(args)))
        if fid is None:
            return False
        return ((state >> fid) & 1) == 1

    def _any_hand_holding_container(self, state: int, container: str) -> bool:
        for fid in self.holding_ids_by_container.get(container, []):
            if (state >> fid) & 1:
                return True
        return False

    def _any_hand_empty(self, state: int) -> bool:
        for h in self.hands:
            if self._is_true(state, 'handEmpty', (h,)):
                return True
        return False

    def _shaker_contains_any(self, state: int, shaker: str) -> bool:
        for fid in self.contains_shaker_any_ids.get(shaker, []):
            if (state >> fid) & 1:
                return True
        return False

    def _pick_shaker(self) -> Optional[str]:
        # If there is a unique shaker, return it; else return the first known.
        if self.default_shaker is not None:
            return self.default_shaker
        if self.shakers:
            return next(iter(self.shakers))
        return None

    # -----------------------------
    # Fact indexing
    # -----------------------------
    def _index_facts(self):
        # Build positive fact map and extract typed symbols via static predicates
        # We'll scan all facts once.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            # Map only positive facts (the model holds only positive forms in its fact list)
            self.fact_pos_map[key] = f.global_id

            if pred == 'handEmpty' and len(args) == 1:
                self.hands.add(args[0])

            if pred == 'shakerLevel' and len(args) == 2:
                self.shakers.add(args[0])

            if pred == 'shakerEmptyLevel' and len(args) == 2:
                self.shakers.add(args[0])

            if pred == 'dispenses' and len(args) == 2:
                # dispenses[d,ingredient]
                self.ingredients.add(args[1])

            if pred == 'cocktailPart1' and len(args) == 2:
                self.cocktails.add(args[0])
                c = args[0]
                p1 = args[1]
                prev = self.cocktail_parts.get(c)
                if prev is None:
                    self.cocktail_parts[c] = (p1, None)
                else:
                    self.cocktail_parts[c] = (p1, prev[1])

            if pred == 'cocktailPart2' and len(args) == 2:
                self.cocktails.add(args[0])
                c = args[0]
                p2 = args[1]
                prev = self.cocktail_parts.get(c)
                if prev is None:
                    self.cocktail_parts[c] = (None, p2)
                else:
                    self.cocktail_parts[c] = (prev[0], p2)

        # Identify a default shaker if unique/available
        if len(self.shakers) == 1:
            self.default_shaker = next(iter(self.shakers))
        elif len(self.shakers) > 1:
            # Prefer a shaker with a level fact; already in set
            self.default_shaker = next(iter(self.shakers))

        # Build holding index per container and contains[shot,ingredient] per ingredient, and contains[shaker,*]
        # To build these efficiently, we need to know which names are shakers for the last mapping.
        # We also accumulate all containers that appear in holding facts.
        containers_seen: Set[str] = set()

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            if pred == 'holding' and len(args) == 2:
                hand, container = args[0], args[1]
                containers_seen.add(container)
                self.holding_ids_by_container.setdefault(container, []).append(f.global_id)

        # Build contains indices for shots per ingredient, and for shakers per any beverage
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            if pred == 'contains' and len(args) == 2:
                cnt, bev = args[0], args[1]
                # Shaker-any mapping
                if cnt in self.shakers:
                    self.contains_shaker_any_ids.setdefault(cnt, []).append(f.global_id)
                # Shot-ingredient mapping
                if bev in self.ingredients and cnt not in self.shakers:
                    # assume containers that are not shakers and contain an ingredient are shots
                    self.contains_shot_ing_ids.setdefault(bev, []).append((cnt, f.global_id))
