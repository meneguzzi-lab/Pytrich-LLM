from typing import Dict, Tuple, List, Set, Optional

# Robust import of the framework base class
try:
    from Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic
    except Exception:
        class Heuristic:  # Fallback stub (for linting/testing outside framework)
            def __init__(self, name: str = ""):
                self.name = name
            def __call__(self, model, state, task_network):
                return 0


class BarmanGPT5Heuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Key design ideas (lower bounds, domain-specific, HTN-aware):
    - We reason on remaining abstract tasks whenever possible, estimating the minimum
      number of primitive actions still required (admissible lower bound).
    - We explicitly model the cocktail-making pipeline in this domain:
        * To prepare a cocktail in a shaker from scratch (2 ingredients):
          - For each ingredient: grasp a clean shot (1), fill from dispenser (1), pour to shaker (1) => 3 each => 6 total
          - Grasp shaker (1), shake (1) => +2
          - Total make-in-shaker (not counting cleaning shaker, not counting final pour to shot): 8 actions
        * Cleaning a shaker before switching cocktails:
          - If it already contains a cocktail and is shaked: empty-shaker (1) + clean-shaker (1) => 2
          - If empty but not clean: clean-shaker (1) => 1
          - If already clean: 0
        * Pouring a ready cocktail from shaker to shot:
          - Prepare the target shot to be clean+empty (0..2 lower bound depending on state)
          - Grasp the shaker if not held (0..1 lower bound)
          - Pour-shaker-to-shot (1)
    - We group AchieveContainsShotCocktail tasks by cocktail to avoid double counting the
      cocktail-making cost when the same cocktail is requested for multiple shots.
    - We also provide conservative lower bounds for mid-level tasks if they appear in the
      current task network (e.g., AchieveCleanShot, AchieveHolding, DoPourShakerToShot).

    Implementation details:
    - We index all grounded facts once (bit-IDs) and use fast bitwise membership checks.
    - We parse task names to extract ground arguments safely; if parsing fails, we fall back to coarse bounds.
    - We avoid scanning explicit state strings inside the heuristic loop to remain fast on large instances.

    Admissibility notes:
    - All counted contributions represent necessary minimal actions on any valid completion,
      or we choose even smaller conservative lower bounds (e.g., assuming available fresh shots),
      so h(n) never exceeds the true remaining cost.
    - We carefully avoid double-counting making cost for identical cocktails by grouping.

    Usage:
    - Instantiate in Pytrich with parameter name barman_gpt5_heuristic_3 (lowercase as required).
    """

    def __init__(self, barman_gpt5_heuristic_3: bool = False):
        super().__init__(name="BarmanGPT5Heuristic3")
        # Index/caches (built lazily per model)
        self._indexed = False
        self._indexed_model_id = None

        # Fact indices
        self.contains_fact: Dict[Tuple[str, str], int] = {}
        self.contains_by_container: Dict[str, List[int]] = {}
        self.clean_fact: Dict[str, int] = {}
        self.empty_fact: Dict[str, int] = {}
        self.used_by_container: Dict[str, List[int]] = {}
        self.holding_fact: Dict[Tuple[str, str], int] = {}
        self.holding_by_container: Dict[str, List[int]] = {}
        self.handempty_fact: Dict[str, int] = {}
        self.ontable_fact: Dict[str, int] = {}
        self.shaked_fact: Dict[str, int] = {}
        self.unshaked_fact: Dict[str, int] = {}

        # Object sets
        self.shakers: Set[str] = set()
        self.hands: Set[str] = set()
        self.shots: Set[str] = set()

    # ------------------------ Utility parsing helpers ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, [a.strip() for a in args]
        return None, []

    def _parse_task(self, name: str) -> Tuple[str, List[str]]:
        # Expected grounded string like: TaskName(arg1,arg2,...) but we make it robust.
        try:
            s = name.strip()
            l = s.find("(")
            r = s.rfind(")")
            if l != -1 and r != -1 and r > l:
                base = s[:l].strip()
                args = s[l + 1:r].split(",")
                args = [a.strip() for a in args if a.strip()]
                return base, args
            # Fallback: try whitespace-separated (unlikely in this framework)
            parts = s.split()
            return (parts[0], parts[1:]) if parts else (s, [])
        except Exception:
            return name, []

    # ------------------------ Model indexing ------------------------
    def _ensure_index(self, model):
        if self._indexed and self._indexed_model_id == id(model):
            return

        # Reset indices
        self.contains_fact.clear()
        self.contains_by_container.clear()
        self.clean_fact.clear()
        self.empty_fact.clear()
        self.used_by_container.clear()
        self.holding_fact.clear()
        self.holding_by_container.clear()
        self.handempty_fact.clear()
        self.ontable_fact.clear()
        self.shaked_fact.clear()
        self.unshaked_fact.clear()
        self.shakers.clear()
        self.hands.clear()
        self.shots.clear()

        # Collect via a single pass over facts
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue

            if pred == "contains" and len(args) == 2:
                cont, bev = args
                self.contains_fact[(cont, bev)] = f.global_id
                self.contains_by_container.setdefault(cont, []).append(f.global_id)
                # We'll detect shots later as containers in contains but not known shakers

            elif pred == "clean" and len(args) == 1:
                self.clean_fact[args[0]] = f.global_id

            elif pred == "empty" and len(args) == 1:
                self.empty_fact[args[0]] = f.global_id

            elif pred == "used" and len(args) == 2:
                cont = args[0]
                self.used_by_container.setdefault(cont, []).append(f.global_id)

            elif pred == "holding" and len(args) == 2:
                hand, cont = args
                self.holding_fact[(hand, cont)] = f.global_id
                self.holding_by_container.setdefault(cont, []).append(f.global_id)

            elif pred == "handEmpty" and len(args) == 1:
                self.handempty_fact[args[0]] = f.global_id
                self.hands.add(args[0])

            elif pred == "ontable" and len(args) == 1:
                self.ontable_fact[args[0]] = f.global_id

            elif pred == "shaked" and len(args) == 1:
                self.shaked_fact[args[0]] = f.global_id

            elif pred == "unshaked" and len(args) == 1:
                self.unshaked_fact[args[0]] = f.global_id

            elif pred == "shakerLevel" and len(args) == 2:
                self.shakers.add(args[0])

            elif pred == "shakerEmptyLevel" and len(args) == 2:
                self.shakers.add(args[0])

        # Deduce shots: any container appearing in contains but not a shaker
        for cont in self.contains_by_container.keys():
            if cont not in self.shakers:
                self.shots.add(cont)

        self._indexed = True
        self._indexed_model_id = id(model)

    # ------------------------ Bitwise helpers ------------------------
    @staticmethod
    def _is_true_bit(state: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state >> fact_id) & 1) == 1

    def _is_true_fact(self, state: int, fact_map: Dict, key) -> bool:
        fid = fact_map.get(key, None)
        return self._is_true_bit(state, fid)

    def _any_true_ids(self, state: int, ids: List[int]) -> bool:
        # Iterate bitwise quickly
        for fid in ids:
            if ((state >> fid) & 1) == 1:
                return True
        return False

    # ------------------------ Domain-specific helpers ------------------------
    def _shot_prepare_clean_empty_cost(self, state: int, shot: str) -> int:
        """Lower bound actions to make shot clean and empty.
        - If clean and empty: 0
        - If contains something: at least empty-shot (1) + clean-shot (1) => 2
        - Else if empty but not clean and used: at least clean-shot (1)
        - Else 0 as safe lower bound (unusual/unreachable combos)
        """
        clean = self._is_true_fact(state, self.clean_fact, shot)
        empty = self._is_true_fact(state, self.empty_fact, shot)
        contains_any = self._any_true_ids(state, self.contains_by_container.get(shot, []))
        used_any = self._any_true_ids(state, self.used_by_container.get(shot, []))

        if clean and empty:
            return 0
        if contains_any:
            # Need to empty then clean (used will likely be true afterward or already true)
            return 2
        if empty and (not clean) and used_any:
            return 1
        # Default conservative LB
        return 0

    def _exists_ready_shaker_for_cocktail(self, state: int, cocktail: str) -> List[str]:
        """Return list of shakers that currently contain the cocktail and are shaked."""
        ready = []
        for sh in self.shakers:
            if self._is_true_fact(state, self.shaked_fact, sh) and \
               self._is_true_fact(state, self.contains_fact, (sh, cocktail)):
                ready.append(sh)
        return ready

    def _any_hand_holding_container(self, state: int, container: str) -> bool:
        return self._any_true_ids(state, self.holding_by_container.get(container, []))

    def _min_shaker_cleaning_cost(self, state: int) -> int:
        """Lower bound: cost to get some shaker to a clean state suitable for mixing new cocktail.
        0 if some clean shaker exists; 1 if some is empty but dirty (clean-shaker);
        2 if only available shaker contains cocktail and is shaked (empty+clean)."""
        if not self.shakers:
            return 0
        best = None
        for sh in self.shakers:
            is_clean = self._is_true_fact(state, self.clean_fact, sh)
            is_empty = self._is_true_fact(state, self.empty_fact, sh)
            contains_any = self._any_true_ids(state, self.contains_by_container.get(sh, []))
            is_shaked = self._is_true_fact(state, self.shaked_fact, sh)

            if is_clean:
                cost = 0
            else:
                if is_empty:
                    cost = 1  # must do clean-shaker
                elif contains_any and is_shaked:
                    cost = 2  # empty-shaker + clean-shaker
                else:
                    # Uncommon states (e.g., dirty/unshaked/contains-ingredients), use safe LB 1
                    cost = 1
            best = cost if best is None else min(best, cost)
        return best if best is not None else 0

    # ------------------------ Core heuristic computation ------------------------
    def __call__(self, model, state: int, task_network: List) -> int:
        self._ensure_index(model)

        # If pure-goal problem (not the case here), we could extract goal facts via model.state_explicit_repr(model.goals)
        # But in HDDL HTN Barman, the guidance comes from the remaining task network.

        # Buckets for abstract tasks
        groups_shot_cocktail: Dict[str, List[str]] = {}  # cocktail -> list of shots
        shaker_cocktail_tasks: List[Tuple[str, str]] = []  # (shaker, cocktail)
        pour_tasks: List[Tuple[str, str, str]] = []  # (shaker, shot, cocktail)
        clean_shot_tasks: List[str] = []
        holding_tasks: List[Tuple[str, str]] = []  # (hand, container)
        handempty_tasks: List[str] = []  # hands
        ontable_tasks: List[str] = []  # containers
        shot_ing_tasks: List[Tuple[str, str]] = []  # (shot, ingredient)
        shaker_ing_tasks: List[Tuple[str, str]] = []  # (shaker, ingredient)

        h_val = 0

        # Count primitive operators as exact remaining cost contribution (each costs 1)
        for t in task_network:
            # Identify Operator vs AbstractTask by duck-typing on attributes
            # Operators have 'pos_precons' and 'apply'; AbstractTask has 'decompositions'
            is_operator = hasattr(t, 'apply') and hasattr(t, 'pos_precons')
            name = getattr(t, 'name', '')
            if is_operator:
                # Remaining primitive actions will cost at least their count
                h_val += 1
                continue

            # Abstract task: parse its name and bucketize
            base, args = self._parse_task(name)
            base = base or ""

            if base.startswith("AchieveContainsShotCocktail") and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                # Defer satisfaction test until group evaluation
                groups_shot_cocktail.setdefault(cocktail, []).append(shot)

            elif base.startswith("AchieveContainsShakerCocktail") and len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                shaker_cocktail_tasks.append((shaker, cocktail))

            elif base.startswith("DoPourShakerToShot") and len(args) >= 3:
                shaker, shot, cocktail = args[0], args[1], args[2]
                pour_tasks.append((shaker, shot, cocktail))

            elif base.startswith("AchieveCleanShot") and len(args) >= 1:
                clean_shot_tasks.append(args[0])

            elif base.startswith("AchieveHolding") and len(args) >= 2:
                hand, cont = args[0], args[1]
                holding_tasks.append((hand, cont))

            elif base.startswith("AchieveHandEmpty") and len(args) >= 1:
                handempty_tasks.append(args[0])

            elif base.startswith("AchieveOnTable") and len(args) >= 1:
                ontable_tasks.append(args[0])

            elif base.startswith("AchieveContainsShotIngredient") and len(args) >= 2:
                shot, ing = args[0], args[1]
                shot_ing_tasks.append((shot, ing))

            elif base.startswith("AchieveContainsShakerIngredient") and len(args) >= 2:
                shaker, ing = args[0], args[1]
                shaker_ing_tasks.append((shaker, ing))

            else:
                # Unknown/other tasks: contribute 0 to keep admissible
                pass

        # Evaluate simple maintenance tasks
        for hand in handempty_tasks:
            if not self._is_true_fact(state, self.handempty_fact, hand):
                # Need at least one drop to free the hand
                h_val += 1

        for cont in ontable_tasks:
            if not self._is_true_fact(state, self.ontable_fact, cont):
                # Need at least a drop to place it on the table
                h_val += 1

        for hand, cont in holding_tasks:
            if not self._is_true_bit(state, self.holding_fact.get((hand, cont))):
                # Need at least one grasp
                h_val += 1

        for shot in clean_shot_tasks:
            if not self._is_true_fact(state, self.clean_fact, shot):
                # Lower bound to clean shot (consider whether it contains something)
                if self._any_true_ids(state, self.contains_by_container.get(shot, [])):
                    h_val += 2  # empty + clean
                elif self._any_true_ids(state, self.used_by_container.get(shot, [])):
                    h_val += 1  # clean only
                else:
                    # Default LB
                    h_val += 0

        # Evaluate mid-level ingredient tasks (conservative LBs)
        for shot, ing in shot_ing_tasks:
            # If already satisfied -> 0, else at least grasp+fill plus any needed prep of the shot
            if not self._is_true_fact(state, self.contains_fact, (shot, ing)):
                prep = self._shot_prepare_clean_empty_cost(state, shot)
                # At least 2 actions: grasp + fill
                h_val += prep + 2

        for shaker, ing in shaker_ing_tasks:
            # If already in shaker -> 0, else at least one pouring action from some shot
            if not self._is_true_fact(state, self.contains_fact, (shaker, ing)):
                # Very conservative: assume a prepared shot with ing exists and just one pour
                h_val += 1

        # Evaluate AchieveContainsShakerCocktail tasks that are not covered by grouped shot-cocktail goals
        covered_cocktails = set(groups_shot_cocktail.keys())
        for shaker, cocktail in shaker_cocktail_tasks:
            if cocktail in covered_cocktails:
                # Will be handled in the group below
                continue
            if self._is_true_fact(state, self.contains_fact, (shaker, cocktail)):
                continue
            # Need to make the cocktail in shaker: cleaning LB + 8 (2 ingredients -> 6 + grasp + shake -> 2)
            # Cleaning LB specific to this shaker (use min est across states)
            is_clean = self._is_true_fact(state, self.clean_fact, shaker)
            is_empty = self._is_true_fact(state, self.empty_fact, shaker)
            contains_any = self._any_true_ids(state, self.contains_by_container.get(shaker, []))
            is_shaked = self._is_true_fact(state, self.shaked_fact, shaker)
            if is_clean:
                clean_cost = 0
            else:
                if is_empty:
                    clean_cost = 1
                elif contains_any and is_shaked:
                    clean_cost = 2
                else:
                    clean_cost = 1
            h_val += clean_cost + 8

        # Evaluate explicit pour tasks
        if pour_tasks:
            # Group by shaker to charge grasp once per shaker if needed
            pours_by_shaker: Dict[str, List[Tuple[str, str]]] = {}
            for shaker, shot, cocktail in pour_tasks:
                # If already satisfied -> 0
                if self._is_true_fact(state, self.contains_fact, (shot, cocktail)):
                    continue
                pours_by_shaker.setdefault(shaker, []).append((shot, cocktail))

            for shaker, lst in pours_by_shaker.items():
                if not lst:
                    continue
                # One-time holding cost if needed
                hold_cost = 0 if self._any_hand_holding_container(state, shaker) else 1
                # However, if no pour is actually possible because shaker does not contain the cocktail,
                # we still keep this as a LB (we might need to make it, but that will be accounted elsewhere)
                # Charge hold once per shaker
                if hold_cost > 0:
                    h_val += hold_cost
                for shot, cocktail in lst:
                    # Shot prep + one pour
                    prep = self._shot_prepare_clean_empty_cost(state, shot)
                    h_val += prep + 1

        # Evaluate grouped AchieveContainsShotCocktail tasks per cocktail
        for cocktail, shots in groups_shot_cocktail.items():
            # Filter out already satisfied
            rem_shots = [s for s in shots if not self._is_true_fact(state, self.contains_fact, (s, cocktail))]
            if not rem_shots:
                continue

            # If any shaker is already ready with this cocktail -> only pours remain
            ready_shakers = self._exists_ready_shaker_for_cocktail(state, cocktail)
            if ready_shakers:
                # One-time hold cost for any ready shaker (LB): if none held now, at least 1 grasp
                # Choose best-case among ready shakers: if any is already held, 0; else 1
                any_held = any(self._any_hand_holding_container(state, sh) for sh in ready_shakers)
                if not any_held:
                    h_val += 1
                # Each remaining shot: prep + pour
                for shot in rem_shots:
                    prep = self._shot_prepare_clean_empty_cost(state, shot)
                    h_val += prep + 1
            else:
                # Need to make the cocktail in some shaker, then pour to each shot
                clean_cost = self._min_shaker_cleaning_cost(state)
                make_cost = 8  # 2 ingredients (6) + grasp shaker (1) + shake (1)
                h_val += clean_cost + make_cost
                for shot in rem_shots:
                    prep = self._shot_prepare_clean_empty_cost(state, shot)
                    h_val += prep + 1  # pour

        # Heuristic value is a non-negative integer lower bound
        return max(0, int(h_val))

