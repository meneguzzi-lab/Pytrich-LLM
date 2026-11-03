from typing import Dict, List, Tuple, Set, Optional

# Best-effort import for the base Heuristic class in different Pytrich setups
try:
    from Heuristics.heuristic import Heuristic  # common in Pytrich
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic  # alternative fallback
    except Exception:
        # Safety fallback to allow static loading; runtime system should provide the real Heuristic
        class Heuristic:  # type: ignore
            def __init__(self, model):
                self.model = model


class BarmanGPT5Heuristic7(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design goals and rationale:
    - Admissible lower bound tailored to Barman's structure: building a cocktail requires
      assembling two ingredients into a shaker, shaking, and pouring into a (clean, empty) shot.
    - Fast bitwise state checks (critical for large instances) through precomputed fact-id masks.
    - Uses both goals and the task network (HTN) to estimate remaining effort, but combines them with
      max to preserve admissibility and avoid double-counting.

    Heuristic structure h(n):
    - Goals-based lower bound (h_goals):
        Let each goal be a fact contains[shot, cocktail]. For each missing goal g:
          base = 1 (you will need at least one pour-shaker-to-shot for that specific shot)
          extra(g) = minimal additional steps beyond the pour to make this pour feasible, estimated from the state:
             * shot preconditions (cheap LB):
                 +1 if the target shot is not empty (must empty before pouring)
                 +1 if the target shot is not clean AND it is known-used (must clean); otherwise 0 to stay admissible
             * shaker/cocktail availability:
                 0 if a shaker already contains the cocktail (since extra only counts beyond the pour)
                 1 if some shaker already has both ingredients (just shake)
                 (min over shakers) otherwise: sum per missing ingredient:
                     0 if already in the shaker
                     1 if available in any non-shaker container (e.g., a shot with that ingredient exists) -> just pour to shaker
                     2 otherwise (need to fill a shot with the ingredient, then pour to shaker)
                   + 1 (for the shake)
        Then combine across all missing goals to stay admissible and capture shared work:
            h_goals = (#missing_goals) + max_g extra(g)
        Rationale: every missing goal requires at least one distinct pour, thus the sum of 1's is safe.
        Shared prep work (like creating/holding ingredients) can help at most one 'hardest' cocktail; taking
        the max extra across goals preserves admissibility and provides strong guidance.

    - Task-network lower bound (h_tn):
        Count 1 per top-level AchieveContainsShotCocktail task still in the current task network.
        This gives a decomposition-aware LB that doesn't overcount (each such task needs a distinct pour).

    Final heuristic: h(n) = max(h_goals, h_tn)

    Efficiency:
    - Precomputes fact-id indices and bitmasks for contains / used / clean / empty / shaked.
    - Uses only bitwise membership checks at evaluation time. Scales well to largest instances.

    Notes:
    - Parsing grounded fact names follows the critical format: +predicate[arg1,arg2].
    - No heavy string operations on each node; all heuristic-critical work is O(1) masks and few dictionary lookups.
    - The heuristic is designed to be robust even if the domain is extended to multiple shakers.
    """

    def __init__(self, model, barman_gpt5_heuristic_7: bool = True):
        super().__init__(model)
        self.model = model

        # Parsed indices
        self.shakers: Set[str] = set()
        self.contains_entries: List[Tuple[str, str, int]] = []  # (container, beverage, fact_id)
        self.contains_by_pair: Dict[Tuple[str, str], int] = {}
        self.contains_mask_by_beverage_nonshaker: Dict[str, int] = {}
        self.contains_mask_by_beverage_shaker: Dict[str, int] = {}

        self.clean_id: Dict[str, int] = {}
        self.empty_id: Dict[str, int] = {}
        self.shaked_id: Dict[str, int] = {}
        self.used_mask_by_container: Dict[str, int] = {}

        self.cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}  # cocktail -> (ing1, ing2)

        # Goals: list of (shot, cocktail, fact_id)
        self.goal_contains_pairs: List[Tuple[str, str, int]] = []

        self._build_indices()
        self._parse_goals()

    # ------------------- Heuristic interface -------------------
    def __call__(self, state_bitwise: int, task_network: List) -> int:
        h_goals = self._goals_lower_bound(state_bitwise)
        h_tn = self._tasknet_lower_bound(task_network)
        return max(h_goals, h_tn)

    # ------------------- Core LB computations -------------------
    def _goals_lower_bound(self, state_bitwise: int) -> int:
        missing = 0
        max_extra = 0

        for shot, cocktail, fact_id in self.goal_contains_pairs:
            # If goal already satisfied in state, contribute 0
            if self._is_true(state_bitwise, fact_id):
                continue

            missing += 1

            # 1) Shot-side minimal requirements (beyond pour)
            extra_shot = 0
            # Need target shot to be empty (if not empty, at least one empty-shot action)
            eid = self.empty_id.get(shot)
            if eid is not None and not self._is_true(state_bitwise, eid):
                extra_shot += 1
            # Need target shot to be clean. If we know the shot is dirty AND used, at least one clean-shot action.
            cid = self.clean_id.get(shot)
            if cid is not None and not self._is_true(state_bitwise, cid):
                # only add a certain lower bound if we know the shot is 'used' with some beverage
                used_mask = self.used_mask_by_container.get(shot, 0)
                if used_mask and (state_bitwise & used_mask) != 0:
                    extra_shot += 1
                # else: leave as 0 to remain strictly admissible

            # 2) Shaker-side minimal requirements (beyond the final pour)
            #    Try: (a) already has cocktail, (b) has both ingredients, (c) needs ingredient add + shake.
            extra_shaker = self._extra_shaker_cost(state_bitwise, cocktail)

            extra = extra_shot + extra_shaker
            if extra > max_extra:
                max_extra = extra

        if missing == 0:
            return 0
        # every missing goal needs at least one pour; and we add the maximum extra across goals to stay admissible
        return missing + max_extra

    def _tasknet_lower_bound(self, task_network: List) -> int:
        if not task_network:
            return 0
        lb = 0
        for t in task_network:
            # Defensive: tasks have attribute 'name' in Pytrich
            name = getattr(t, 'name', '')
            if not name:
                continue
            if 'AchieveContainsShotCocktail' in name:
                # each such task will inevitably require at least one pour action
                lb += 1
        return lb

    # Compute extra shaker prep beyond the final pour for a single (missing) goal cocktail
    def _extra_shaker_cost(self, state_bitwise: int, cocktail: str) -> int:
        # If any shaker already contains the cocktail -> no extra beyond pour
        mask_shakers_have_cocktail = self.contains_mask_by_beverage_shaker.get(cocktail, 0)
        if mask_shakers_have_cocktail and (state_bitwise & mask_shakers_have_cocktail) != 0:
            return 0  # extra beyond pour is zero (we only need the pour)

        # try if some shaker already has both ingredients: need just shake as extra (1)
        ing1, ing2 = self.cocktail_parts.get(cocktail, (None, None))
        if ing1 is not None and ing2 is not None and self.shakers:
            for s in self.shakers:
                c1_id = self.contains_by_pair.get((s, ing1))
                c2_id = self.contains_by_pair.get((s, ing2))
                if c1_id is not None and c2_id is not None:
                    if self._is_true(state_bitwise, c1_id) and self._is_true(state_bitwise, c2_id):
                        return 1  # only need to shake (pour is already counted in baseline)

        # otherwise, we need to add ingredients to a shaker, then shake
        best_extra = None
        if not self.shakers:
            # Extremely defensive: if no shaker known, just say we need at least a shake (can't do better)
            return 1

        for s in self.shakers:
            req_sum = 0
            for ing in (ing1, ing2):
                if ing is None:
                    continue  # robust to missing static info (shouldn't happen in provided instances)
                # If already in shaker s -> 0
                in_s_id = self.contains_by_pair.get((s, ing))
                if in_s_id is not None and self._is_true(state_bitwise, in_s_id):
                    continue

                # If available somewhere else (any non-shaker container contains it) -> 1
                nonshaker_mask = self.contains_mask_by_beverage_nonshaker.get(ing, 0)
                if nonshaker_mask and (state_bitwise & nonshaker_mask) != 0:
                    req_sum += 1
                else:
                    # Otherwise need to fill a shot with the ingredient, then pour to shaker -> 2
                    req_sum += 2
            # Add 1 for the shake
            req_sum += 1

            if best_extra is None or req_sum < best_extra:
                best_extra = req_sum

        return best_extra if best_extra is not None else 1

    # ------------------- Indices and parsing -------------------
    def _build_indices(self):
        # First pass: collect entries and identify shakers
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'shakerLevel' or pred == 'shakerEmptyLevel':
                if len(args) >= 1:
                    self.shakers.add(args[0])

        # Second pass: build maps
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'contains' and len(args) == 2:
                c, b = args[0], args[1]
                self.contains_entries.append((c, b, fact.global_id))
                self.contains_by_pair[(c, b)] = fact.global_id

            elif pred == 'clean' and len(args) == 1:
                self.clean_id[args[0]] = fact.global_id

            elif pred == 'empty' and len(args) == 1:
                self.empty_id[args[0]] = fact.global_id

            elif pred == 'shaked' and len(args) == 1:
                self.shaked_id[args[0]] = fact.global_id

            elif pred == 'used' and len(args) == 2:
                c = args[0]
                self.used_mask_by_container[c] = self.used_mask_by_container.get(c, 0) | (1 << fact.global_id)

            elif pred == 'cocktailPart1' and len(args) == 2:
                ckt, ing = args[0], args[1]
                i1, i2 = self.cocktail_parts.get(ckt, (None, None))
                self.cocktail_parts[ckt] = (ing, i2)

            elif pred == 'cocktailPart2' and len(args) == 2:
                ckt, ing = args[0], args[1]
                i1, i2 = self.cocktail_parts.get(ckt, (None, None))
                self.cocktail_parts[ckt] = (i1, ing)

        # Build beverage->mask by shaker/non-shaker
        for c, b, fid in self.contains_entries:
            if c in self.shakers:
                self.contains_mask_by_beverage_shaker[b] = self.contains_mask_by_beverage_shaker.get(b, 0) | (1 << fid)
            else:
                self.contains_mask_by_beverage_nonshaker[b] = self.contains_mask_by_beverage_nonshaker.get(b, 0) | (1 << fid)

    def _parse_goals(self):
        # model.goals is a bitwise integer; obtain explicit representation and map back to ids
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'contains' and len(args) == 2:
                c, b = args[0], args[1]
                fid = self.contains_by_pair.get((c, b))
                if fid is not None:
                    self.goal_contains_pairs.append((c, b, fid))

    # ------------------- Utilities -------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name in the format: +predicate[arg1,arg2] or -predicate[arg1,arg2].
        Returns (predicate, [args...]) or (None, []).
        """
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1 : -1]
            args = args_str.split(',') if args_str else []
            return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1
