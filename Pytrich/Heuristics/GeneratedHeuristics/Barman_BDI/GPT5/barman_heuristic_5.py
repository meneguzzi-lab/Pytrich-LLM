from typing import Dict, Tuple, List, Set, Optional

# Pytrich base class (provided by the framework)
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT5Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design rationale (concise):
    - The core top-level tasks are AchieveContainsShotCocktail(shot, cocktail).
      Making a cocktail requires (in the best case, ignoring hand/holding preconditions):
        * 2x (fill-shot + pour-shot-to-shaker) + 1x shake + 1x pour-shaker-to-shot.
        * This gives a tight per-cocktail lower bound of 6 actions if the shaker
          starts clean and empty for that cocktail.
      However, multiple shots of the same cocktail can share a single prepared
      shaker content and need only 1 action each (pour-shaker-to-shot) after the
      initial preparation. Moving between different cocktail groups requires at
      least empty+clean (2) before preparing the next cocktail (additional 5).
    - We exploit this structure to compute an admissible grouped lower bound:
        H_group = total_pours + initial_prep_cost_min + 7 * (G - 1)
        where:
          * total_pours = number of remaining AchieveContainsShotCocktail tasks
                           (each needs at least one pour to shot), minus those
                           already satisfied in the current state.
          * G = number of distinct cocktails still required by remaining
                AchieveContainsShotCocktail tasks (after filtering already
                satisfied ones).
          * initial_prep_cost_min is a state-aware, minimal preparation cost for
            the first cocktail group given the current shaker contents:
              - 0 if the shaker already contains that cocktail and is shaked
              - 1 if the shaker contains both required ingredients and is unshaked
              - 3 if the shaker contains exactly one of the required ingredients
                and is unshaked (fill+pour other ingredient + shake)
              - 5 if shaker is empty and clean (make from scratch)
              - 6 if shaker is empty and NOT clean (clean + make from scratch)
              - 7 otherwise (must empty+clean + make from scratch)
      This remains admissible even with the peculiar level/next relations in the
      provided instances (which permit repeated pours), because we only ever
      count hard preconditions and primitive actions that must happen at least
      once; we never assume optimistic reuse unless enabled by the current state.

    - For mixed task networks during refinement, we also compute a per-task
      lower bound (H_tasks) for individual abstract tasks using domain knowledge
      and the current state (facts bitset). The final heuristic is:

          h(state, tn) = max(H_group, H_tasks)

      Taking max of two lower bounds preserves admissibility and provides
      stronger guidance.

    Efficiency:
    - All relevant fact indices (global ids) are precomputed once in __init__.
    - State membership is checked via bitwise operations only (no string search).
    - Task parsing is robust to different name formats and cached minimally.
    """

    def __init__(self, model, barman_gpt5_heuristic_5: bool = True):
        super().__init__(model)
        self.model = model
        self.barman_gpt5_heuristic_5 = barman_gpt5_heuristic_5

        # Static caches/indexes over model facts
        self.contains_facts: Dict[Tuple[str, str], int] = {}  # (container, beverage) -> fact.global_id
        self.clean_facts: Dict[str, int] = {}                # container -> fact.global_id
        self.empty_facts: Dict[str, int] = {}                # container -> fact.global_id
        self.shaked_facts: Dict[str, int] = {}               # shaker -> fact.global_id
        self.unshaked_facts: Dict[str, int] = {}             # shaker -> fact.global_id
        self.cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}  # cocktail -> (part1, part2)

        # Identify shakers from domain facts (via shakerLevel or shakerEmptyLevel predicates)
        self.shakers: Set[str] = set()

        # Cocktail/ingredient universes (deduced from cocktailPart1/2)
        self.cocktails: Set[str] = set()
        self.ingredients: Set[str] = set()

        # Build indices from all facts in the model
        self._index_facts()

        # Choose a primary shaker (the domain instances provide a single shaker)
        self.primary_shaker: Optional[str] = next(iter(self.shakers), None)

    # ---------- Public API ----------

    def evaluate(self, state_bitwise: int, task_network: List) -> int:
        if task_network is None or len(task_network) == 0:
            return 0

        # 1) Grouped bound for top-level cocktail tasks
        h_group = self._grouped_cocktail_lower_bound(state_bitwise, task_network)

        # 2) Per-task lower bound for the current (possibly mixed) task network
        h_tasks = self._per_task_lower_bound(state_bitwise, task_network)

        # Final heuristic: take the stronger (still admissible)
        return max(h_group, h_tasks)

    # ---------- Internal: Fact and Task Parsing / Indexing ----------

    def _index_facts(self):
        # Scan all facts, parse their predicate/args from the canonical name
        for fact in self.model.facts:
            pred, args = self._parse_fact_name_safe(fact.name)
            if pred is None:
                continue

            if pred == 'contains' and len(args) == 2:
                container, beverage = args
                self.contains_facts[(container, beverage)] = fact.global_id
            elif pred == 'clean' and len(args) == 1:
                self.clean_facts[args[0]] = fact.global_id
            elif pred == 'empty' and len(args) == 1:
                self.empty_facts[args[0]] = fact.global_id
            elif pred == 'shaked' and len(args) == 1:
                self.shaked_facts[args[0]] = fact.global_id
            elif pred == 'unshaked' and len(args) == 1:
                self.unshaked_facts[args[0]] = fact.global_id
            elif pred == 'shakerLevel' and len(args) == 2:
                # args[0] is a shaker id
                self.shakers.add(args[0])
            elif pred == 'shakerEmptyLevel' and len(args) == 2:
                self.shakers.add(args[0])
            elif pred == 'cocktailPart1' and len(args) == 2:
                c, i = args
                self.cocktails.add(c)
                self.ingredients.add(i)
                p1, p2 = self.cocktail_parts.get(c, (None, None))
                self.cocktail_parts[c] = (i, p2)
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, i = args
                self.cocktails.add(c)
                self.ingredients.add(i)
                p1, p2 = self.cocktail_parts.get(c, (None, None))
                self.cocktail_parts[c] = (p1, i)

    def _parse_fact_name_safe(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names. Model stores names like '+pred[arg1,arg2]'.
        Be forgiving if format slightly differs.
        """
        name = name.strip()
        # Expected: +predicate[arg1,arg2]
        if name.startswith('+') or name.startswith('-'):
            sign = name[0]
            rest = name[1:]
        else:
            sign = '+'
            rest = name
        bidx = rest.find('[')
        if bidx > 0 and rest.endswith(']'):
            pred = rest[:bidx]
            args_str = rest[bidx + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args
        # Try parentheses fallback
        pidx = rest.find('(')
        if pidx > 0 and rest.endswith(')'):
            pred = rest[:pidx]
            args_str = rest[pidx + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args
        # Try space-separated fallback
        parts = rest.split()
        if len(parts) >= 1:
            pred = parts[0]
            args = parts[1:]
            return pred, args
        return None, []

    def _parse_task_name_safe(self, name: str) -> Tuple[str, List[str]]:
        """
        Robust parsing for task/operator names with potential formats:
         - Name[args]
         - Name(args)
         - Name arg1 arg2
        Returns (base_name, args_list) where args_list may be empty on failure.
        """
        s = name.strip()
        # Prefer bracket form
        b = s.find('[')
        if b > 0 and s.endswith(']'):
            base = s[:b]
            args_str = s[b + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return base, args
        # Parentheses form
        p = s.find('(')
        if p > 0 and s.endswith(')'):
            base = s[:p]
            args_str = s[p + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return base, args
        # Space-separated
        parts = s.split()
        if len(parts) >= 1:
            base = parts[0]
            args = parts[1:]
            return base, args
        return s, []

    # ---------- Internal: Bitwise state access ----------

    @staticmethod
    def _is_true(state_bitwise: int, fact_gid: Optional[int]) -> bool:
        if fact_gid is None:
            return False
        return (state_bitwise >> fact_gid) & 1 == 1

    def _gid(self, pred: str, args: Tuple[str, ...]) -> Optional[int]:
        # Limited set: contains, clean, empty, shaked, unshaked via pre-indexed maps
        if pred == 'contains' and len(args) == 2:
            return self.contains_facts.get((args[0], args[1]))
        if pred == 'clean' and len(args) == 1:
            return self.clean_facts.get(args[0])
        if pred == 'empty' and len(args) == 1:
            return self.empty_facts.get(args[0])
        if pred == 'shaked' and len(args) == 1:
            return self.shaked_facts.get(args[0])
        if pred == 'unshaked' and len(args) == 1:
            return self.unshaked_facts.get(args[0])
        return None

    # ---------- Internal: Grouped bound for AchieveContainsShotCocktail ----------

    def _grouped_cocktail_lower_bound(self, state_bitwise: int, task_network: List) -> int:
        # Collect pending top-level cocktail requirements from current TN
        pending: List[Tuple[str, str]] = []  # (shot, cocktail)
        for t in task_network:
            name = getattr(t, 'name', '')
            base, args = self._parse_task_name_safe(name)
            if base.endswith('AchieveContainsShotCocktail') or base == 'AchieveContainsShotCocktail':
                # Expect (shot, cocktail)
                if len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    # If already satisfied in current state, skip (zero cost)
                    gid = self._gid('contains', (shot, cocktail))
                    if not self._is_true(state_bitwise, gid):
                        pending.append((shot, cocktail))

        if not pending:
            return 0

        # Count per cocktail
        counts_by_cocktail: Dict[str, int] = {}
        for _, c in pending:
            counts_by_cocktail[c] = counts_by_cocktail.get(c, 0) + 1

        total_pours = sum(counts_by_cocktail.values())  # at least 1 pour per remaining shot
        G = len(counts_by_cocktail)

        # If we have no shaker info, return the trivial (but admissible) lower bound: total pours only
        if not self.primary_shaker:
            return total_pours

        s = self.primary_shaker

        # Current shaker status
        is_empty = self._is_true(state_bitwise, self.empty_facts.get(s))
        is_clean = self._is_true(state_bitwise, self.clean_facts.get(s))
        is_shaked = self._is_true(state_bitwise, self.shaked_facts.get(s))
        is_unshaked = self._is_true(state_bitwise, self.unshaked_facts.get(s))

        # Beverages currently in shaker
        shaker_contents: Set[str] = set()
        # We can iterate over beverages that show up in 'contains' facts indexed for this container
        for (cont, bev), gid in self.contains_facts.items():
            if cont == s and self._is_true(state_bitwise, gid):
                shaker_contents.add(bev)

        # Compute the minimal initial preparation cost among the target cocktails
        initial_prep_candidates: List[int] = []

        # Helper to test how many ingredients of cocktail c are already inside the shaker
        def ingr_match_count(c_name: str) -> int:
            p1, p2 = self.cocktail_parts.get(c_name, (None, None))
            count = 0
            if p1 and p1 in shaker_contents:
                count += 1
            if p2 and p2 in shaker_contents:
                count += 1
            return count

        # Case 1: already contains a needed cocktail and is shaked -> 0
        has_matching_cocktail = any((c in shaker_contents) and is_shaked for c in counts_by_cocktail.keys())
        if has_matching_cocktail:
            initial_prep_candidates.append(0)

        # Case 2: has both required ingredients for some needed cocktail and is unshaked -> 1 (shake)
        for c in counts_by_cocktail.keys():
            if is_unshaked and ingr_match_count(c) >= 2:
                initial_prep_candidates.append(1)
                break

        # Case 3: has exactly one required ingredient for some needed cocktail and is unshaked -> 3 (fill+pour other + shake)
        for c in counts_by_cocktail.keys():
            if is_unshaked and ingr_match_count(c) == 1:
                initial_prep_candidates.append(3)
                break

        # Case 4: empty & clean -> 5; empty & not clean -> 6; otherwise (not empty and no help) -> 7
        if is_empty:
            initial_prep_candidates.append(5 if is_clean else 6)
        else:
            # Not empty and no helpful match -> 7 (empty+clean+make)
            initial_prep_candidates.append(7)

        initial_prep_cost_min = min(initial_prep_candidates) if initial_prep_candidates else 0

        # For additional distinct cocktail groups beyond the first, we need at least (empty+clean + (make from scratch)) = 7 each
        group_switch_cost = 7 * max(G - 1, 0)

        return total_pours + initial_prep_cost_min + group_switch_cost

    # ---------- Internal: Per-task lower bounds ----------

    def _per_task_lower_bound(self, state_bitwise: int, task_network: List) -> int:
        total = 0
        for t in task_network:
            name = getattr(t, 'name', '')
            base, args = self._parse_task_name_safe(name)

            # Only consider abstract tasks (primitives can be treated as 0; they are to be executed, not refined)
            # We attempt to detect known abstract tasks by name.
            if base.endswith('AchieveContainsShotCocktail') or base == 'AchieveContainsShotCocktail':
                # LB: at least one pour from shaker to shot if not already true
                if len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    if not self._is_true(state_bitwise, self._gid('contains', (shot, cocktail))):
                        total += 1

            elif base.endswith('AchieveContainsShakerCocktail') or base == 'AchieveContainsShakerCocktail':
                # LB depends on current shaker status
                if len(args) >= 2:
                    shaker, cocktail = args[0], args[1]
                    if self._is_true(state_bitwise, self._gid('contains', (shaker, cocktail))) and \
                       self._is_true(state_bitwise, self._gid('shaked', (shaker,))):
                        # Already has cocktail and is shaked: 0 to achieve this abstract task
                        pass
                    else:
                        # Evaluate a minimal prep from current state for this cocktail (like initial prep above)
                        is_empty = self._is_true(state_bitwise, self._gid('empty', (shaker,)))
                        is_clean = self._is_true(state_bitwise, self._gid('clean', (shaker,)))
                        is_shaked = self._is_true(state_bitwise, self._gid('shaked', (shaker,)))
                        is_unshaked = self._is_true(state_bitwise, self._gid('unshaked', (shaker,)))

                        # Gather current shaker contents
                        shaker_bevs = set()
                        for (cont, bev), gid in self.contains_facts.items():
                            if cont == shaker and self._is_true(state_bitwise, gid):
                                shaker_bevs.add(bev)

                        # If already shaked with different cocktail or irrelevant content, need empty+clean first
                        if is_shaked and len(shaker_bevs) > 0 and (cocktail not in shaker_bevs):
                            # 7 = empty+clean + (fill+pour + fill+pour + shake)
                            total += 7
                        else:
                            # Try ingredient-based shortcuts
                            p1, p2 = self.cocktail_parts.get(cocktail, (None, None))
                            if is_unshaked and p1 and p2 and (p1 in shaker_bevs) and (p2 in shaker_bevs):
                                total += 1  # just shake
                            elif is_unshaked and p1 and p2 and (
                                 (p1 in shaker_bevs) ^ (p2 in shaker_bevs)):
                                total += 3  # add missing ingredient (fill+pour) + shake
                            elif is_empty and is_clean:
                                total += 5  # make from scratch
                            elif is_empty and not is_clean:
                                total += 6  # clean + make from scratch
                            else:
                                # Not empty and not directly helpful -> 7
                                total += 7

            elif base.endswith('AchieveContainsShakerIngredient') or base == 'AchieveContainsShakerIngredient':
                # LB: 0 if already in shaker; else 2 normally; 3 if shaker is shaked and must be emptied first
                if len(args) >= 2:
                    shaker, ingr = args[0], args[1]
                    if self._is_true(state_bitwise, self._gid('contains', (shaker, ingr))):
                        pass
                    else:
                        if self._is_true(state_bitwise, self._gid('shaked', (shaker,))):
                            total += 3  # empty + (fill+pour)
                        else:
                            total += 2  # (fill+pour)

            elif base.endswith('AchieveContainsShotIngredient') or base == 'AchieveContainsShotIngredient':
                # LB: 0 if shot already has ingredient; else 1 (fill-shot)
                if len(args) >= 2:
                    shot, ingr = args[0], args[1]
                    if not self._is_true(state_bitwise, self._gid('contains', (shot, ingr))):
                        total += 1

            elif base.endswith('AchieveCleanShot') or base == 'AchieveCleanShot':
                if len(args) >= 1:
                    shot = args[0]
                    if not self._is_true(state_bitwise, self._gid('clean', (shot,))):
                        # If not clean:
                        #  - If contains something: empty-shot + clean-shot => 2
                        #  - Else if empty and used: clean-shot => 1
                        #  - Else fall back to 1 as a safe lower bound
                        has_any = False
                        for (cont, bev), gid in self.contains_facts.items():
                            if cont == shot and self._is_true(state_bitwise, gid):
                                has_any = True
                                break
                        if has_any:
                            total += 2
                        else:
                            total += 1

            elif base.endswith('AchieveCleanShaker') or base == 'AchieveCleanShaker':
                if len(args) >= 1:
                    shaker = args[0]
                    if not self._is_true(state_bitwise, self._gid('clean', (shaker,))):
                        if self._is_true(state_bitwise, self._gid('empty', (shaker,))):
                            total += 1  # clean-shaker
                        else:
                            # Full shaker with cocktail must be emptied then cleaned => 2
                            total += 2

            elif base.endswith('AchieveHolding') or base == 'AchieveHolding':
                # LB: 0 if already holding; else at least 1 (grasp)
                if len(args) >= 2:
                    hand, cont = args[0], args[1]
                    # We don't have a direct holding fact index (predicate: holding ?hand ?container)
                    # Try to find it via the contains_facts index is not applicable; we need to scan facts once.
                    holding_gid = self._find_fact_gid_cached('holding', (hand, cont))
                    if holding_gid is None or not self._is_true(state_bitwise, holding_gid):
                        total += 1

            elif base.endswith('AchieveOnTable') or base == 'AchieveOnTable':
                if len(args) >= 1:
                    cont = args[0]
                    ontable_gid = self._find_fact_gid_cached('ontable', (cont,))
                    if ontable_gid is None or not self._is_true(state_bitwise, ontable_gid):
                        total += 1

            elif base.endswith('AchieveHandEmpty') or base == 'AchieveHandEmpty':
                if len(args) >= 1:
                    hand = args[0]
                    hand_empty_gid = self._find_fact_gid_cached('handEmpty', (hand,))
                    if hand_empty_gid is None or not self._is_true(state_bitwise, hand_empty_gid):
                        total += 1

            elif base.endswith('DoPourShakerToShot') or base == 'DoPourShakerToShot':
                # LB: at least 1 (pour-shaker-to-shot)
                total += 1

            else:
                # Unrecognized tasks: default to 0 to remain admissible
                pass

        return total

    # Cache for less frequent predicate lookups not pre-indexed
    _extra_fact_cache: Dict[Tuple[str, Tuple[str, ...]], Optional[int]] = {}

    def _find_fact_gid_cached(self, pred: str, args: Tuple[str, ...]) -> Optional[int]:
        key = (pred, args)
        if key in self._extra_fact_cache:
            return self._extra_fact_cache[key]
        gid = None
        # One-time scan of all facts to locate this predicate/args
        for f in self.model.facts:
            p, a = self._parse_fact_name_safe(f.name)
            if p == pred and tuple(a) == args:
                gid = f.global_id
                break
        self._extra_fact_cache[key] = gid
        return gid


# Alias for potential external configuration naming consistency
barman_gpt5_heuristic_5 = BarmanGPT5Heuristic5
