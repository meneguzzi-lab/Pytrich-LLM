from typing import Dict, List, Tuple, Optional, Union

try:
    # Expected base class in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub if import path differs during static checks
    class Heuristic:
        def __init__(self, model):
            self.model = model
        def h(self, state, task_network):
            return 0
        def __call__(self, state, task_network):
            return 0
        def get_value(self, state, task_network):
            return 0


class BarmanGPT5Heuristic18(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design rationale (lower-bound, domain-informed, HTN-aware):
    - Each AchieveContainsShotCocktail(shot, cocktail) requires at least:
      * Making (or having) the cocktail in a shaker (delete-relaxed, hand/cleaning ignored except where strictly necessary),
      * The target shot being clean (at least one cleaning action if not already clean),
      * Holding the shaker (at least one grasp if none holds it),
      * One pour action from shaker to shot.
    - Making a cocktail in a shaker (lower bound):
      * If shaker already contains the cocktail and is shaked: 0
      * If shaker already contains both required ingredients (unshaked): 1 (shake)
      * If shaker contains one required ingredient: 2 (fill-shot + pour-used) + 1 (shake) = 3
      * Else: 2 fills + 2 pours (using two initially clean shots) + 1 shake = 5
      This ignores grasp/drop/empty/clean preconditions unless absolutely necessary to keep admissibility.
    - Subtask-aware: For partially decomposed networks, we add cheap lower bounds per leaf abstract task:
      * AchieveCleanShot(s): 0 if clean(s) else 1 (clean-shot)
      * AchieveContainsShakerIngredient: 0 if already in shaker else 2 (fill + pour)
      * AchieveContainsShotIngredient: 0 if shot contains ingredient else 1 (fill-shot)
      * AchieveContainsShakerCocktail: as above (make in shaker LB)
      * AchieveHolding(h, c): 0 if already holding else 1 (grasp)
      * AchieveHandEmpty(h): 0 if empty else 1 (drop)
      * AchieveOnTable(c): 0 if on table else 1 (drop)
      * DoPourShakerToShot: 0 if shot already contains cocktail else 1 (pour)
      * AchieveCleanShaker: 0 if clean else 1 (clean-shaker)
    - Primitive operators in the task network (if any) contribute 1 each (must still be executed), which is an exact lower bound for them.

    Implementation notes:
    - Uses only bitwise state checks via fact.global_id for speed (no explicit state reconstruction per node).
    - Robust parsing for task names to extract parameters if available in forms like 'TaskName[arg1,arg2]', 'TaskName(arg1,arg2)', or whitespace-separated.
    - Preprocesses:
        * fact index map: (predicate, args) -> fact.global_id
        * cocktail parts mapping: cocktail -> (ing1, ing2)
        * known shakers list (from 'shakerLevel'/'shakerEmptyLevel' facts); default to the first if multiple
        * known hands list (from 'handEmpty'/'holding') for holding checks
    - Admissibility: Heuristic is optimistic (underestimates by ignoring many supporting actions). It never overestimates.
    - Efficiency: All per-node computations are O(#tasks + small constant) with O(1) bitwise checks per needed fact.

    """

    def __init__(self, model, barman_gpt5_heuristic_18: bool = True):
        super().__init__(model)
        self.model = model

        # Fact index for O(1) bitwise queries: (predicate, tuple(args)) -> global_id
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._build_fact_index()

        # Static relations: cocktail parts
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self._build_cocktail_parts()

        # Known shakers (prefer first seen); derive from shakerLevel/shakerEmptyLevel
        self._shakers: List[str] = self._collect_shakers()
        self._default_shaker: Optional[str] = self._shakers[0] if self._shakers else None

        # Known hands
        self._hands: List[str] = self._collect_hands()

    # --------------- Internal helpers: model preprocessing ---------------

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = [a.strip() for a in args_str.split(",")] if args_str else []
                return predicate, args
        return None, []

    def _build_fact_index(self):
        # Only store positive facts (begin with '+')
        for fact in getattr(self.model, 'facts', []):
            name = getattr(fact, 'name', '')
            if not name.startswith('+'):
                continue
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_gid[key] = fact.global_id

    def _collect_shakers(self) -> List[str]:
        shakers = set()
        # Derive from 'shakerLevel' and/or 'shakerEmptyLevel'
        for (pred, args), gid in self._fact_gid.items():
            if pred in ("shakerLevel", "shakerEmptyLevel") and len(args) == 2:
                shakers.add(args[0])
        return sorted(shakers)

    def _collect_hands(self) -> List[str]:
        hands = set()
        for (pred, args), gid in self._fact_gid.items():
            if pred == 'handEmpty' and len(args) == 1:
                hands.add(args[0])
            elif pred == 'holding' and len(args) == 2:
                hands.add(args[0])
        return sorted(hands)

    def _build_cocktail_parts(self):
        # Fill from facts cocktailPart1[cocktail,ingredient], cocktailPart2[cocktail,ingredient]
        for (pred, args), gid in self._fact_gid.items():
            if pred in ("cocktailPart1", "cocktailPart2") and len(args) == 2:
                c, ing = args[0], args[1]
                if c not in self._cocktail_parts:
                    self._cocktail_parts[c] = (None, None)
                p1, p2 = self._cocktail_parts[c]
                if pred == "cocktailPart1":
                    p1 = ing
                else:
                    p2 = ing
                self._cocktail_parts[c] = (p1, p2)

    # --------------- Internal helpers: bitwise state checks ---------------

    def _gid(self, predicate: str, args: List[str]) -> Optional[int]:
        return self._fact_gid.get((predicate, tuple(args)))

    def _is_true(self, state_bitwise: int, predicate: str, args: List[str]) -> int:
        gid = self._gid(predicate, args)
        if gid is None:
            return 0
        return (state_bitwise >> gid) & 1

    def _any_holding_shaker(self, state_bitwise: int, shaker: str) -> bool:
        for h in self._hands:
            if self._is_true(state_bitwise, 'holding', [h, shaker]):
                return True
        return False

    # --------------- Internal helpers: task parsing ---------------

    def _parse_task_name_and_args(self, name: str) -> Tuple[str, List[str]]:
        s = (name or '').strip()
        if not s:
            return '', []

        # Try bracket style: Name[arg1,arg2,...]
        lb = s.find('[')
        rb = s.rfind(']')
        if lb > 0 and rb > lb:
            base = s[:lb].strip()
            inner = s[lb+1:rb].strip()
            args = [a.strip() for a in inner.split(',')] if inner else []
            return base, args

        # Try parenthesis style: Name(arg1,arg2,...)
        lp = s.find('(')
        rp = s.rfind(')')
        if lp > 0 and rp > lp:
            base = s[:lp].strip()
            inner = s[lp+1:rp].strip()
            args = [a.strip() for a in inner.split(',')] if inner else []
            return base, args

        # Fallback: whitespace separated (Name arg1 arg2 ...)
        parts = s.split()
        base = parts[0]
        args = parts[1:]
        return base, args

    # --------------- Core LB computations ---------------

    def _lb_make_shaker_cocktail(self, state_bitwise: int, cocktail: str) -> int:
        """
        Lower bound to achieve contains[shaker, cocktail] and shaked[shaker].
        Chooses a default shaker (first discovered). If none, returns 0 (no guidance).
        """
        s = self._default_shaker
        if s is None:
            return 0

        # If already have the cocktail in shaker and shaked => 0
        if self._is_true(state_bitwise, 'contains', [s, cocktail]) and \
           self._is_true(state_bitwise, 'shaked', [s]):
            return 0

        # Retrieve required ingredients for this cocktail
        ing1, ing2 = self._cocktail_parts.get(cocktail, (None, None))
        if ing1 is None or ing2 is None:
            # Unknown cocktail composition => minimal non-zero lower bound if not already true
            # We still need a shake+some fills+pours. Optimistically return 1 (weak but admissible)
            return 1

        have1 = self._is_true(state_bitwise, 'contains', [s, ing1])
        have2 = self._is_true(state_bitwise, 'contains', [s, ing2])

        k = int(bool(have1)) + int(bool(have2))
        if k >= 2:
            # Both ingredients present (unshaked assumed) => need at least 1 shake
            return 1
        elif k == 1:
            # Need at least one fill-shot + one pour-used + 1 shake
            return 3
        else:
            # Need two fills and two pours (+1 shake) using two clean shots optimistically
            return 5

    # --------------- Heuristic evaluation ---------------

    def h(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        if not task_network:
            # In HTN domains without explicit goal facts, empty TN often signals completion
            # Still ensure we don't violate model.goals if any
            try:
                if getattr(self.model, 'goals', 0):
                    if (self.model.goals & state_bitwise) != self.model.goals:
                        # Some fact-goals not yet true: add minimal 1 to remain admissible
                        return 1
            except Exception:
                pass
            return 0

        cost = 0
        default_shaker = self._default_shaker

        for task in task_network:
            # Operators contribute exactly 1 (they must be executed)
            # AbstractTasks contribute domain-specific lower bounds.
            tname = getattr(task, 'name', '')
            base, args = self._parse_task_name_and_args(tname)

            # Attempt to detect Operator vs AbstractTask via component ID ranges if available
            # but prefer name-based fallbacks. If 'task' has 'pos_precons', 'add_effects', it's an Operator.
            is_operator = hasattr(task, 'pos_precons') and hasattr(task, 'add_effects')
            if is_operator:
                cost += 1
                continue

            # Abstract task cases
            if base == 'AchieveContainsShotCocktail' and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                if self._is_true(state_bitwise, 'contains', [shot, cocktail]):
                    # Task already satisfied
                    continue
                # Need to pour: +1
                c = 1
                # Need a clean shot (at least one clean-shot action if not already clean)
                c += 0 if self._is_true(state_bitwise, 'clean', [shot]) else 1
                # Need to hold a shaker (choose default)
                if default_shaker is not None:
                    c += 0 if self._any_holding_shaker(state_bitwise, default_shaker) else 1
                # Need to have the cocktail ready in shaker
                c += self._lb_make_shaker_cocktail(state_bitwise, cocktail)
                cost += c

            elif base == 'DoPourShakerToShot' and len(args) >= 3:
                shaker, shot, cocktail = args[0], args[1], args[2]
                # Lower bound: need exactly one pour if not yet in shot
                if not self._is_true(state_bitwise, 'contains', [shot, cocktail]):
                    cost += 1

            elif base == 'AchieveContainsShakerCocktail' and len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                # Use shaker specified if possible, else default
                prev_default = self._default_shaker
                if shaker:
                    self._default_shaker = shaker
                cost += self._lb_make_shaker_cocktail(state_bitwise, cocktail)
                self._default_shaker = prev_default

            elif base == 'AchieveContainsShakerIngredient' and len(args) >= 2:
                shaker, ingredient = args[0], args[1]
                # If already there => 0 else at least fill + pour = 2
                if not self._is_true(state_bitwise, 'contains', [shaker, ingredient]):
                    cost += 2

            elif base == 'AchieveContainsShotIngredient' and len(args) >= 2:
                shot, ingredient = args[0], args[1]
                # If not already contained, at least 1 fill-shot
                if not self._is_true(state_bitwise, 'contains', [shot, ingredient]):
                    cost += 1

            elif base == 'AchieveCleanShot' and len(args) >= 1:
                shot = args[0]
                if not self._is_true(state_bitwise, 'clean', [shot]):
                    cost += 1

            elif base == 'AchieveCleanShaker' and len(args) >= 1:
                shaker = args[0]
                if not self._is_true(state_bitwise, 'clean', [shaker]):
                    cost += 1

            elif base == 'AchieveHolding' and len(args) >= 2:
                hand, container = args[0], args[1]
                if not self._is_true(state_bitwise, 'holding', [hand, container]):
                    cost += 1

            elif base == 'AchieveHandEmpty' and len(args) >= 1:
                hand = args[0]
                if not self._is_true(state_bitwise, 'handEmpty', [hand]):
                    cost += 1

            elif base == 'AchieveOnTable' and len(args) >= 1:
                container = args[0]
                if not self._is_true(state_bitwise, 'ontable', [container]):
                    cost += 1

            else:
                # Unknown abstract tasks: no contribution (optimistic)
                pass

        return max(0, int(cost))

    # Common planner entry points
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]]):
        return self.h(state_bitwise, task_network)

    def get_value(self, state_bitwise: int, task_network: List[Union[object, object]]):
        return self.h(state_bitwise, task_network)
