from typing import Dict, List, Tuple, Union, Optional, Set

# Fallback base class in case the environment does not provide it during static checks
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:  # pragma: no cover
    class Heuristic:  # minimal shim
        def __init__(self, model):
            self.model = model


class BarmanGPT5Heuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain (Pytrich Planner).

    Design rationale (short):
    - We compute a safe (admissible) lower bound that leverages domain structure while
      remaining very fast.
    - We combine: (a) the number of already scheduled primitive operators in the current
      task network (they must all be executed), with (b) a conservative lower bound for
      each remaining abstract task using domain knowledge, and (c) a (very conservative)
      goal-derived lower bound.
    - Critical domain-specific improvements:
        * AchieveContainsShotCocktail[shot,cocktail]: if shot already contains cocktail -> 0;
          else if any shaker already contains that cocktail -> 1; else -> 2 (at least 1 action
          to create cocktail in shaker via shake + 1 action to pour to the shot).
        * AchieveHandEmpty[hand]: aggregated across all such tasks in the TN. Because the domain
          allows satisfying AchieveHandEmpty(null) if ANY hand is empty (see HandEmptyNull), a single
          drop can satisfy multiple AchieveHandEmpty tasks at once. Hence, if any AchieveHandEmpty
          tasks are present: 0 if any hand is currently empty, else 1.
        * Other Achieve* tasks: use 0 if already satisfied in the state; otherwise 1. This is a safe
          lower bound since each requires at least one primitive operator in the best case.
    - Efficiency: everything is bitwise, with precomputed indices and masks. No string searches over
      the state during evaluation (only lookups by id).

    Admissibility considerations:
    - Counting scheduled primitive operators in the current TN is admissible: they must all be
      executed to finish.
    - For abstract tasks, we use conservative per-task lower bounds. The only cross-task sharing
      concern is AchieveHandEmpty: we aggregate it as a single unit (0/1) for the whole TN to avoid
      overcounting.
    - Goals: we include goal-based LB as 0/1 (0 if all goal facts currently true, else 1). Using 0/1
      avoids any risk of overestimating due to multi-fact add effects.

    This heuristic is designed to be very fast, scale to large instances, and provide non-trivial
    guidance on the Barman benchmarks.
    """

    def __init__(self, model, barman_gpt5_heuristic_2: bool = True):
        super().__init__(model)
        self.enabled = barman_gpt5_heuristic_2

        # Fast fact access: name -> Fact, and per-predicate indexes
        self._name_to_fact: Dict[str, object] = {}
        self._contains_index: Dict[Tuple[str, str], int] = {}  # (container, beverage) -> fact global_id
        self._clean_index: Dict[str, int] = {}  # container -> fact global_id
        self._ontable_index: Dict[str, int] = {}  # container -> fact global_id
        self._holding_index: Dict[Tuple[str, str], int] = {}  # (hand, container) -> fact global_id
        self._handempty_index: Dict[str, int] = {}  # hand -> fact global_id

        # Agg masks
        self._any_hand_empty_mask: int = 0

        # Known shakers set for quick checks (to detect if any shaker already contains a cocktail)
        self._shakers: Set[str] = set()

        # Precompute indices and masks
        self._precompute_fact_indices()

    # ------------------------------- Public API -------------------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        return self.compute(state_bitwise, task_network)

    def compute(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        if not self.enabled:
            return 0

        # Lower bound from already scheduled primitive operators
        op_lb = 0

        # Sum of LBs for abstract tasks (with special aggregation for AchieveHandEmpty)
        abstract_lb = 0
        handempty_task_present = False

        # Iterate over the (possibly id-based) TN components and evaluate
        for tn_item in task_network:
            comp, gid = self._get_component_and_id(tn_item)
            if comp is None:
                continue

            if self._is_operator_id(gid):
                op_lb += 1  # every scheduled operator must be executed
                continue

            if self._is_abstract_task_id(gid):
                head, args = self._parse_component_name(getattr(comp, 'name', ''))
                if not head:
                    # Unknown/malformed name; safest is to add 0 to keep admissibility
                    continue

                if head == 'AchieveContainsShotCocktail' and len(args) == 2:
                    abstract_lb += self._lb_AchieveContainsShotCocktail(state_bitwise, args[0], args[1])
                elif head == 'AchieveContainsShakerCocktail' and len(args) == 2:
                    abstract_lb += self._lb_AchieveContainsShakerCocktail(state_bitwise, args[0], args[1])
                elif head == 'DoPourShakerToShot' and len(args) == 3:
                    abstract_lb += self._lb_DoPourShakerToShot(state_bitwise, args[0], args[1], args[2])
                elif head == 'AchieveContainsShakerIngredient' and len(args) == 2:
                    abstract_lb += self._lb_AchieveContainsShakerIngredient(state_bitwise, args[0], args[1])
                elif head == 'AchieveContainsShotIngredient' and len(args) == 2:
                    abstract_lb += self._lb_AchieveContainsShotIngredient(state_bitwise, args[0], args[1])
                elif head == 'AchieveCleanShot' and len(args) == 1:
                    abstract_lb += self._lb_AchieveCleanShot(state_bitwise, args[0])
                elif head == 'AchieveCleanShaker' and len(args) == 1:
                    abstract_lb += self._lb_AchieveCleanShaker(state_bitwise, args[0])
                elif head == 'AchieveHolding' and len(args) == 2:
                    abstract_lb += self._lb_AchieveHolding(state_bitwise, args[0], args[1])
                elif head == 'AchieveOnTable' and len(args) == 1:
                    abstract_lb += self._lb_AchieveOnTable(state_bitwise, args[0])
                elif head == 'AchieveHandEmpty':
                    # aggregate later (0 if any hand empty, else 1 if at least one such task exists)
                    handempty_task_present = True
                else:
                    # Unknown abstract task: add 0 (stay admissible)
                    pass
            else:
                # Safety: if it's neither operator nor abstract (shouldn't happen)
                pass

        # Aggregate AchieveHandEmpty tasks cost once
        if handempty_task_present:
            abstract_lb += (0 if self._any_hand_empty_true(state_bitwise) else 1)

        # Goal-based very conservative lower bound: 1 if any goal fact missing; else 0
        goal_lb = 1 if (self.model.goals & (~state_bitwise)) != 0 else 0

        # Final admissible estimate
        h_val = op_lb + abstract_lb
        if goal_lb > h_val:
            h_val = goal_lb
        return h_val

    # ---------------------------- Lower bound pieces --------------------------
    def _lb_AchieveContainsShotCocktail(self, state: int, shot: str, cocktail: str) -> int:
        # If already satisfied: 0
        if self._is_true(state, self._contains_fact_gid(shot, cocktail)):
            return 0
        # If some shaker already contains the cocktail: 1 (just pour to the shot)
        for s in self._shakers:
            if self._is_true(state, self._contains_fact_gid(s, cocktail)):
                return 1
        # Otherwise: at least 2 (need to create cocktail in shaker via shake, then pour to shot)
        return 2

    def _lb_AchieveContainsShakerCocktail(self, state: int, shaker: str, cocktail: str) -> int:
        # 0 if already in shaker, else at least 1 (e.g., one shake if both ingredients are there)
        return 0 if self._is_true(state, self._contains_fact_gid(shaker, cocktail)) else 1

    def _lb_DoPourShakerToShot(self, state: int, shaker: str, shot: str, cocktail: str) -> int:
        # If shot already has cocktail: 0; else one pour action is minimally needed
        return 0 if self._is_true(state, self._contains_fact_gid(shot, cocktail)) else 1

    def _lb_AchieveContainsShakerIngredient(self, state: int, shaker: str, ingr: str) -> int:
        # 0 if already in shaker; else at least 1 (one pour from some shot)
        return 0 if self._is_true(state, self._contains_fact_gid(shaker, ingr)) else 1

    def _lb_AchieveContainsShotIngredient(self, state: int, shot: str, ingr: str) -> int:
        # 0 if already in shot; else at least 1 (one fill-shot action)
        return 0 if self._is_true(state, self._contains_fact_gid(shot, ingr)) else 1

    def _lb_AchieveCleanShot(self, state: int, shot: str) -> int:
        # 0 if already clean; else at least 1 (clean-shot eventually)
        gid = self._clean_index.get(shot)
        return 0 if self._is_true(state, gid) else 1

    def _lb_AchieveCleanShaker(self, state: int, shaker: str) -> int:
        # 0 if already clean; else at least 1 (clean-shaker eventually)
        gid = self._clean_index.get(shaker)
        return 0 if self._is_true(state, gid) else 1

    def _lb_AchieveHolding(self, state: int, hand: str, container: str) -> int:
        # 0 if already holding; else at least 1 (grasp eventually)
        gid = self._holding_index.get((hand, container))
        return 0 if self._is_true(state, gid) else 1

    def _lb_AchieveOnTable(self, state: int, container: str) -> int:
        # 0 if already on table; else at least 1 (drop eventually)
        gid = self._ontable_index.get(container)
        return 0 if self._is_true(state, gid) else 1

    # --------------------------- Helpers and parsing --------------------------
    def _precompute_fact_indices(self):
        # Map: name -> fact; also build per-predicate indices (positive facts only)
        # Detect shakers using shakerLevel/shakerEmptyLevel facts
        for fact in self.model.facts:
            name = getattr(fact, 'name', '')
            self._name_to_fact[name] = fact

        for fact in self.model.facts:
            name = getattr(fact, 'name', '')
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue

            # Only index positive variants
            if not name.startswith('+'):
                continue

            if pred == 'contains' and len(args) == 2:
                self._contains_index[(args[0], args[1])] = fact.global_id
            elif pred == 'clean' and len(args) == 1:
                self._clean_index[args[0]] = fact.global_id
            elif pred == 'ontable' and len(args) == 1:
                self._ontable_index[args[0]] = fact.global_id
            elif pred == 'holding' and len(args) == 2:
                self._holding_index[(args[0], args[1])] = fact.global_id
            elif pred == 'handEmpty' and len(args) == 1:
                self._handempty_index[args[0]] = fact.global_id
                self._any_hand_empty_mask |= (1 << fact.global_id)
            elif pred == 'shakerLevel' and len(args) == 2:
                # container (arg0) is a shaker for sure
                self._shakers.add(args[0])
            elif pred == 'shakerEmptyLevel' and len(args) == 2:
                self._shakers.add(args[0])

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            pred = s[:bracket_idx]
            args_str = s[bracket_idx + 1 : -1]
            args = args_str.split(',') if args_str else []
            return pred, args
        return None, []

    @staticmethod
    def _parse_component_name(name: str) -> Tuple[Optional[str], List[str]]:
        # Operators/Tasks usually follow: head[arg1,arg2,...]
        # No leading +/- here.
        if not name:
            return None, []
        s = name.strip()
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            head = s[:bracket_idx]
            args_str = s[bracket_idx + 1 : -1]
            args = args_str.split(',') if args_str else []
            return head, args
        return s, []  # sometimes no args (robustness)

    def _contains_fact_gid(self, container: str, beverage: str) -> Optional[int]:
        return self._contains_index.get((container, beverage))

    @staticmethod
    def _is_true(state_bitwise: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) != 0

    def _any_hand_empty_true(self, state_bitwise: int) -> bool:
        return (state_bitwise & self._any_hand_empty_mask) != 0

    def _get_component_and_id(self, item: Union[int, object]) -> Tuple[Optional[object], Optional[int]]:
        # Accepts either a global_id (int) or the actual component (Operator/AbstractTask)
        if isinstance(item, int):
            try:
                comp = self.model.get_component(item)
                return comp, item
            except Exception:
                return None, None
        else:
            gid = getattr(item, 'global_id', None)
            if gid is None:
                return None, None
            # Ensure we use the canonical object from the model (for type/range checks)
            try:
                comp = self.model.get_component(gid)
            except Exception:
                comp = item
            return comp, gid

    def _is_operator_id(self, gid: int) -> bool:
        return self.model.iop_init <= gid <= self.model.iop_end

    def _is_abstract_task_id(self, gid: int) -> bool:
        return self.model.iabt_init <= gid <= self.model.iabt_end
