from typing import Dict, Tuple, List, Set, Optional, Any

try:
    # Import the base Heuristic from Pytrich if available
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Safe fallback stub (in case of standalone testing). The planner will provide the actual Heuristic.
    class Heuristic:
        def __init__(self, model, *args, **kwargs):
            self.model = model
        def h(self, state_bitwise: int, task_network: List[Any]) -> int:
            return 0


class BarmanGPT5Heuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain (barman_htn).

    Key ideas (admissible, domain-specific lower bounds):
    - Target tasks are of the form AchieveContainsShotCocktail(shot, cocktail).
      To realize this, at least these primitive actions are needed (in the best case):
        * Make the cocktail in a shaker (if not already): requires transferring two ingredients (2 actions each: fill-shot, pour-to-shaker) + 1 shake -> lower bound 5.
          We refine this by detecting if the shaker already contains some required ingredient(s), and by accounting for minimal reset costs when a (different) cocktail is in the shaker.
        * Pour the cocktail from shaker to the target shot: at least 1 pour action + possibly 1 grasp (if currently no hand holds the shaker) + preparing the shot to be clean and empty (0/1/2 actions lower bound depending on current state).
    - We compute these costs using only current state facts via fast bitwise checks (no expensive explicit state enumeration).
    - For HTN task networks:
        * If the network consists solely of AchieveContainsShotCocktail tasks, we sum their individual lower bounds (cocktails are distinct in training instances, making the sum a safe and informative LB).
        * Otherwise, we conservatively take the max of the individual task LBs to avoid double-counting during partial decompositions.

    Important implementation details:
    - We pre-index facts for O(1) state membership checks: (predicate, args) -> Fact (global_id)
    - Parse component names of tasks/operators with a bracketed signature: Name[arg1,arg2,...]
    - Detect domain objects (shakers, hands, beverages, cocktails) from the grounded facts.
    - Use only bitwise checks to test if a fact holds in the state.

    Notes on admissibility:
    - All per-task estimates are constructed as lower bounds on the number of primitive actions still required.
    - We never count precondition-enforcing actions in a way that could overestimate; when uncertain, we choose the smaller (safer) LB.
    - Summation is only applied for top-level cocktail-achievement tasks; otherwise we take a max across tasks.
    """

    def __init__(self, model, barman_gpt5_heuristic_14: bool = True, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.barman_gpt5_heuristic_14 = barman_gpt5_heuristic_14

        # Fact index: (predicate, (arg1,...)) -> Fact
        self.fact_index: Dict[Tuple[str, Tuple[str, ...]], Any] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self.fact_index[(pred, tuple(args))] = f

        # Extract domain-specific sets
        self.hands: Set[str] = set()
        self.shakers: Set[str] = set()
        self.cocktails: Set[str] = set()
        self.ingredients: Set[str] = set()
        self.beverages: Set[str] = set()  # union of cocktails and ingredients (as they appear in facts)

        # Cocktail parts mapping: cocktail -> (p1, p2)
        self.cocktail_parts: Dict[str, Tuple[str, str]] = {}

        # Fill from grounded facts
        for (pred, args), fact in self.fact_index.items():
            if pred == 'handEmpty' and len(args) == 1:
                self.hands.add(args[0])
            elif pred in ('shakerLevel', 'shakerEmptyLevel', 'shaked', 'unshaked') and len(args) >= 1:
                self.shakers.add(args[0])
            elif pred == 'cocktailPart1' and len(args) == 2:
                c, ing = args[0], args[1]
                self.cocktails.add(c)
                self.ingredients.add(ing)
                # initialize if needed
                if c not in self.cocktail_parts:
                    self.cocktail_parts[c] = (ing, None)
                else:
                    # set part1
                    self.cocktail_parts[c] = (ing, self.cocktail_parts[c][1])
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, ing = args[0], args[1]
                self.cocktails.add(c)
                self.ingredients.add(ing)
                if c not in self.cocktail_parts:
                    self.cocktail_parts[c] = (None, ing)
                else:
                    self.cocktail_parts[c] = (self.cocktail_parts[c][0], ing)

        # Any beverages that appear in contains/used facts
        for (pred, args), fact in self.fact_index.items():
            if pred in ('contains', 'used') and len(args) == 2:
                # args[1] is a beverage (ingredient or cocktail)
                self.beverages.add(args[1])
        # Ensure beverages include both recorded cocktails and ingredients from parts
        self.beverages |= self.cocktails | self.ingredients

        # For convenience when no shaker detected (should not happen in provided instances)
        if not self.shakers:
            # Try to infer shaker by any occurrence in facts like clean/unshaked and container positions
            for (pred, args), fact in self.fact_index.items():
                if pred in ('clean', 'empty', 'ontable') and len(args) == 1:
                    # Can't distinguish type reliably, but store candidates
                    pass
            # Leave as empty set - methods handle gracefully

    # -------------------------------
    # Public heuristic entry point
    # -------------------------------
    def h(self, state_bitwise: int, task_network: List[Any]) -> int:
        if state_bitwise is None:
            return 0
        if not task_network:
            return 0

        # Map each task in the network to a per-task LB
        task_lbs: List[int] = []
        all_are_shot_cocktail = True

        for t in task_network:
            comp = self._get_component(t)
            name = getattr(comp, 'name', '')
            pred, args = self._parse_name(name)
            lb = 0

            if pred == 'AchieveContainsShotCocktail' and len(args) == 2:
                shot, cocktail = args
                lb = self._lb_achieve_shot_cocktail(state_bitwise, shot, cocktail)
            elif pred == 'AchieveContainsShakerCocktail' and len(args) == 2:
                shaker, cocktail = args
                lb = self._lb_make_cocktail(state_bitwise, shaker, cocktail)
                all_are_shot_cocktail = False
            elif pred == 'AchieveContainsShakerIngredient' and len(args) == 2:
                shaker, ing = args
                lb = self._lb_achieve_shaker_ingredient(state_bitwise, shaker, ing)
                all_are_shot_cocktail = False
            elif pred == 'AchieveCleanShot' and len(args) == 1:
                shot = args[0]
                lb = self._lb_clean_shot(state_bitwise, shot)
                all_are_shot_cocktail = False
            elif pred == 'AchieveCleanShaker' and len(args) == 1:
                shaker = args[0]
                lb = self._lb_clean_shaker(state_bitwise, shaker)
                all_are_shot_cocktail = False
            elif pred == 'DoPourShakerToShot' and len(args) == 3:
                shaker, shot, cocktail = args
                lb = self._lb_pour_to_shot(state_bitwise, shaker, shot, cocktail)
                all_are_shot_cocktail = False
            elif pred == 'AchieveContainsShotIngredient' and len(args) == 2:
                shot, ing = args
                lb = self._lb_achieve_shot_ingredient(state_bitwise, shot, ing)
                all_are_shot_cocktail = False
            elif pred == 'AchieveHolding' and len(args) == 2:
                hand, cont = args
                lb = 0 if self._is_true(state_bitwise, 'holding', (hand, cont)) else 1
                all_are_shot_cocktail = False
            elif pred == 'AchieveHandEmpty' and len(args) == 1:
                hand = args[0]
                lb = 0 if self._is_true(state_bitwise, 'handEmpty', (hand,)) else 1
                all_are_shot_cocktail = False
            elif pred == 'AchieveOnTable' and len(args) == 1:
                cont = args[0]
                lb = 0 if self._is_true(state_bitwise, 'ontable', (cont,)) else 1
                all_are_shot_cocktail = False
            else:
                # Primitive operator or unknown task: count as 1 step remaining
                # (still admissible as a LB for an unapplied primitive)
                lb = 1
                all_are_shot_cocktail = False

            task_lbs.append(lb)

        if not task_lbs:
            return 0

        # If the current TN is a set of top-level cocktail serving goals, sum them for stronger guidance.
        if all_are_shot_cocktail:
            return sum(task_lbs)
        # Otherwise, to avoid optimistic double counting across partially decomposed subtasks, take a safe max.
        return max(task_lbs)

    # -------------------------------
    # Domain-specific lower bounds
    # -------------------------------
    def _lb_achieve_shot_cocktail(self, state: int, shot: str, cocktail: str) -> int:
        # If already satisfied
        if self._is_true(state, 'contains', (shot, cocktail)):
            return 0

        best = None
        # Consider each shaker (usually 1 in the training set)
        for sh in (self.shakers if self.shakers else {None}):
            # Cost to ensure shaker contains the cocktail (0 if already)
            make_cost = 0
            if sh is not None:
                if not self._is_true(state, 'contains', (sh, cocktail)):
                    make_cost = self._lb_make_cocktail(state, sh, cocktail)
            else:
                # No shaker known: can't make realistically, but return minimal non-zero LB
                make_cost = 1

            # Cost to pour into the target shot once shaker has the cocktail
            pour_cost = 1  # the pour action itself
            # Need to hold the shaker to pour
            if sh is not None:
                pour_cost += self._lb_hold_shaker(state, sh)
            # Shot must be empty and clean
            pour_cost += self._lb_shot_ready(state, shot)

            total = make_cost + pour_cost
            if best is None or total < best:
                best = total

        return best if best is not None else 1

    def _lb_make_cocktail(self, state: int, shaker: str, cocktail: str) -> int:
        # If already in shaker
        if self._is_true(state, 'contains', (shaker, cocktail)):
            return 0

        # Determine required ingredients
        p1, p2 = self.cocktail_parts.get(cocktail, (None, None))
        missing = 0
        if p1 is None or p2 is None:
            # If info missing, return a minimal safe bound: need at least shake + one transfer
            return 2

        # Count how many of the required ingredients are currently missing in the shaker
        if not self._is_true(state, 'contains', (shaker, p1)):
            missing += 1
        if not self._is_true(state, 'contains', (shaker, p2)):
            missing += 1

        # Base cost: 2 actions per missing ingredient (fill-shot + pour-to-shaker), plus 1 shake
        base = 2 * missing + 1

        # Minimal reset costs for the shaker if it's currently in conflicting state
        reset = 0
        # If shaker currently contains some cocktail different than target, must empty (1) + clean (1)
        contains_any_cocktail = False
        if self.cocktails:
            for c in self.cocktails:
                if self._is_true(state, 'contains', (shaker, c)):
                    contains_any_cocktail = True
                    if c != cocktail:
                        reset = max(reset, 2)
                    break
        # If it's empty but dirty, need at least one cleaning action before first clean-pour
        if self._is_true(state, 'empty', (shaker,)) and not self._is_true(state, 'clean', (shaker,)):
            reset = max(reset, 1)

        return base + reset

    def _lb_pour_to_shot(self, state: int, shaker: str, shot: str, cocktail: str) -> int:
        # If already satisfied, nothing to do
        if self._is_true(state, 'contains', (shot, cocktail)):
            return 0
        # At least: 1 pour + possibly 1 grasp (if not holding shaker) + shot preparation
        cost = 1
        cost += self._lb_hold_shaker(state, shaker)
        cost += self._lb_shot_ready(state, shot)
        return cost

    def _lb_achieve_shaker_ingredient(self, state: int, shaker: str, ing: str) -> int:
        if self._is_true(state, 'contains', (shaker, ing)):
            return 0
        # If shaker contains some cocktail (shaked), must empty + clean first before adding new ingredients to an empty clean shaker
        if any(self._is_true(state, 'contains', (shaker, c)) for c in self.cocktails):
            reset = 2
        elif self._is_true(state, 'empty', (shaker,)) and not self._is_true(state, 'clean', (shaker,)):
            reset = 1
        else:
            reset = 0
        # LB: need to at least fill-shot (1) and pour to shaker (1)
        return 2 + reset

    def _lb_clean_shot(self, state: int, shot: str) -> int:
        if self._is_true(state, 'clean', (shot,)):
            return 0
        # If shot contains something -> need to empty (1) then clean (1)
        contains_any = any(self._is_true(state, 'contains', (shot, b)) for b in self.beverages)
        if contains_any:
            return 2
        # If shot is empty but dirty, and used is true for some beverage, we can clean in 1
        if self._is_true(state, 'empty', (shot,)):
            used_any = any(self._is_true(state, 'used', (shot, b)) for b in self.beverages)
            return 1 if used_any else 1  # conservative: at least 1
        # Default safe LB
        return 1

    def _lb_clean_shaker(self, state: int, shaker: str) -> int:
        if self._is_true(state, 'clean', (shaker,)):
            return 0
        # If shaker contains a cocktail (and is shaked), need to empty then clean at least
        if any(self._is_true(state, 'contains', (shaker, c)) for c in self.cocktails):
            return 2
        # If already empty (but dirty), one clean action suffices
        if self._is_true(state, 'empty', (shaker,)):
            return 1
        # Otherwise, at least 1 action will be needed (very conservative)
        return 1

    def _lb_achieve_shot_ingredient(self, state: int, shot: str, ing: str) -> int:
        if self._is_true(state, 'contains', (shot, ing)):
            return 0
        # At least one fill-shot action
        return 1

    def _lb_shot_ready(self, state: int, shot: str) -> int:
        """Lower bound to ensure shot is (clean AND empty)."""
        is_clean = self._is_true(state, 'clean', (shot,))
        is_empty = self._is_true(state, 'empty', (shot,))
        if is_clean and is_empty:
            return 0
        if is_empty:
            # Likely requires one clean-shot. If used flag is absent, 1 remains a safe LB.
            return 1
        # Not empty: must empty (1) and then clean (1)
        return 2

    def _lb_hold_shaker(self, state: int, shaker: str) -> int:
        """Lower bound on actions to ensure some hand is holding the shaker."""
        for h in self.hands:
            if self._is_true(state, 'holding', (h, shaker)):
                return 0
        # Need at least one grasp
        return 1

    # -------------------------------
    # Utilities
    # -------------------------------
    def _get_component(self, comp_or_id: Any) -> Any:
        # Task networks may carry either component objects or global IDs; normalize to object
        if isinstance(comp_or_id, int):
            return self.model.get_component(comp_or_id)
        return comp_or_id

    def _is_true(self, state: int, pred: str, args: Tuple[str, ...]) -> bool:
        fact = self.fact_index.get((pred, tuple(args)))
        if fact is None:
            return False
        return ((state >> fact.global_id) & 1) == 1

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names in the format +predicate[arg1,arg2] or -predicate[arg1,arg2].
        Returns (predicate, [args]) or (None, []) if cannot parse.
        """
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1:-1]
            args = args_str.split(',') if args_str else []
            return predicate, args
        return None, []

    @staticmethod
    def _parse_name(name: str) -> Tuple[str, List[str]]:
        """
        General parser for component names (operators, tasks, decompositions) in the form Name[arg1,arg2,...].
        If no bracket is present, returns (name, []).
        """
        if not name:
            return '', []
        s = name.strip()
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1:-1]
            args = args_str.split(',') if args_str else []
            return predicate, args
        return s, []
