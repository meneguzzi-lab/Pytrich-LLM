from typing import Dict, List, Tuple, Set, Union, Optional

# Attempt to import Heuristic base class from Pytrich
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimal stub to avoid import errors when inspecting the file standalone
    class Heuristic:
        def __init__(self, model):
            self.model = model


class BarmanGPT5Heuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    High-level idea
    - Goals are tasks AchieveContainsShotCocktail(shot, cocktail). To achieve one from scratch you need at least:
        2x (fill-shot + pour-shot-to-shaker) + 1x shake + 1x pour-shaker-to-shot = 6 primitive actions.
      This gives a safe admissible lower bound baseline of 6 per cocktail when nothing is yet prepared.
    - We leverage shaker contents synergy:
        * If shaker already contains the exact cocktail (and is shaked), each required shot of that cocktail can
          be achieved by a single pour-shaker-to-shot action (LB 1 each, plus any shot cleaning/emptying needed).
        * If shaker contains both required ingredients for a cocktail (and is unshaked), we need 1 shake and
          then 1 pour per shot (LB 2 per shot set, plus shot fixes).
        * If shaker contains exactly one of the required ingredients (and is unshaked), we need one more ingredient
          transfer (fill + pour) + shake (3), then 1 pour per shot.
        * Otherwise, to create a new cocktail from scratch: 2x (fill + pour) + shake = 5 to get it in the shaker,
          plus 1 pour per shot.
    - Multiple goals and grouping by cocktail:
        We group all pending AchieveContainsShotCocktail tasks by cocktail. Making a cocktail once (cost above)
        enables pouring it into multiple shots cheaply (1 per shot). This grouping yields a tighter admissible bound.
        Between different cocktails, in this domain one shaker must be emptied and cleaned to switch cocktails.
        That induces at least 2 primitive actions (empty-shaker + clean-shaker) per switch, i.e., 2*(G-1) if there
        are G distinct cocktails still to produce. We add this as a general lower bound across groups.
    - Shot cleaning/emptying lower bound:
        For any pour-shaker-to-shot, the shot must be empty and clean. For each pending target shot we add:
            +1 if it's not empty, +1 if it's not clean.
        This is a safe LB and usually 0 early in instances (shots start clean+empty), but adds guidance later.

    Robust parsing and efficient bitwise checks
    - Facts are represented as bitwise flags. We pre-index frequently needed '+' facts (contains, clean, empty,
      shaked, unshaked, cocktailPart1/2) by their grounded arguments for O(1) bitwise checks.
    - Tasks may appear in varied grounded name formats. We parse task/operator names robustly with support for
      '[...]', '(...)', or whitespace-separated forms. We look for these pending items in the task network:
        * AbstractTask: AchieveContainsShotCocktail(shot, cocktail)
        * AbstractTask: DoPourShakerToShot(shaker, shot, cocktail)
        * Operator: pour-shaker-to-shot(cocktail, shot, ...)

    Admissibility
    - All contributions are necessary-action lower bounds given current state relaxation; we never assume extra
      prerequisites (e.g., grasping) are strictly required in the lower bound. Grouping and transition penalties
      reflect the single-shaker constraint and known method structure without double-counting creation costs.

    Performance
    - All state membership checks are bitwise.
    - All indices are precomputed in __init__.
    """

    def __init__(self, model, barman_gpt5_heuristic_15: bool = True):
        super().__init__(model)
        self.barman_gpt5_heuristic_15 = barman_gpt5_heuristic_15
        # Precomputed indices for fast bitwise access
        self.contains_pos: Dict[Tuple[str, str], int] = {}
        self.clean_pos: Dict[str, int] = {}
        self.empty_pos: Dict[str, int] = {}
        self.shaked_pos: Dict[str, int] = {}
        self.unshaked_pos: Dict[str, int] = {}
        self.cocktail_parts: Dict[str, Tuple[str, str]] = {}
        self._shakers: Set[str] = set()

        self._build_fact_indices()

    # ---------------------------- Core Heuristic API ----------------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        # Extract pending (shot, cocktail) pairs from task network
        pending = self._extract_pending_shot_cocktail_pairs(task_network, state_bitwise)
        if not pending:
            return 0

        # Filter out pairs already satisfied in the state: contains[shot, cocktail]
        pending_filtered: Dict[str, Set[str]] = {}
        for shot, cocktail in pending:
            c_fact = self.contains_pos.get((shot, cocktail))
            if c_fact is not None and self._is_true(state_bitwise, c_fact):
                continue
            pending_filtered.setdefault(cocktail, set()).add(shot)

        if not pending_filtered:
            return 0

        # Compute per-shot minimal cleaning/emptying lower bound
        shot_fix_costs: Dict[str, int] = {}
        for shots in pending_filtered.values():
            for s in shots:
                if s not in shot_fix_costs:
                    shot_fix_costs[s] = self._shot_pour_fix_cost(state_bitwise, s)

        # Base cost sum across all groups (one-time creation per cocktail + pours per shot + shot fixes)
        # Default creation cost (from scratch): 5 (2 fills + 2 pours + 1 shake)
        base_sum = 0
        for cocktail, shots in pending_filtered.items():
            k = len(shots)
            group_shot_fix = sum(shot_fix_costs[s] for s in shots)
            base_sum += 5 + k + group_shot_fix

        # Compute best synergy reduction for exactly one cocktail group based on current shaker contents
        best_reduction = 0
        for cocktail, shots in pending_filtered.items():
            reduction = self._group_creation_reduction_from_current_state(state_bitwise, cocktail)
            if reduction > best_reduction:
                best_reduction = reduction

        # Transition overhead between distinct cocktails: at least 2 per switch (empty + clean)
        group_count = len(pending_filtered)
        transitions_overhead = 0
        if group_count >= 2:
            transitions_overhead = 2 * (group_count - 1)

        h_value = base_sum - best_reduction + transitions_overhead
        # Heuristic must be non-negative integer
        return int(max(0, h_value))

    # ---------------------------- Index Building ----------------------------
    def _build_fact_indices(self):
        for fact in self.model.facts:
            pred, args, is_pos = self._parse_fact_name(fact.name)
            if not is_pos:
                # We only index positive literals; bitwise state uses those for truth checks
                continue
            if pred == 'contains' and len(args) == 2:
                container, beverage = args[0], args[1]
                self.contains_pos[(container, beverage)] = fact.global_id
            elif pred == 'clean' and len(args) == 1:
                self.clean_pos[args[0]] = fact.global_id
            elif pred == 'empty' and len(args) == 1:
                self.empty_pos[args[0]] = fact.global_id
            elif pred == 'shaked' and len(args) == 1:
                self.shaked_pos[args[0]] = fact.global_id
                self._shakers.add(args[0])
            elif pred == 'unshaked' and len(args) == 1:
                self.unshaked_pos[args[0]] = fact.global_id
                self._shakers.add(args[0])
            elif pred == 'shakerLevel' and len(args) == 2:
                # Helps enumerate shakers robustly
                self._shakers.add(args[0])
            elif pred == 'shakerEmptyLevel' and len(args) == 2:
                self._shakers.add(args[0])
            elif pred == 'cocktailPart1' and len(args) == 2:
                c, ing = args[0], args[1]
                if c not in self.cocktail_parts:
                    self.cocktail_parts[c] = (ing, None)
                else:
                    # ensure tuple fill
                    prev = self.cocktail_parts[c]
                    self.cocktail_parts[c] = (ing, prev[1])
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, ing = args[0], args[1]
                if c not in self.cocktail_parts:
                    self.cocktail_parts[c] = (None, ing)
                else:
                    prev = self.cocktail_parts[c]
                    self.cocktail_parts[c] = (prev[0], ing)

        # Normalize cocktail parts (ensure both present; ignore incomplete pairs gracefully)
        for c, (a, b) in list(self.cocktail_parts.items()):
            if a is None or b is None:
                # keep as-is; reduction checks will handle missing safely
                pass

    # ---------------------------- Parsing helpers ----------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str], bool]:
        """
        Returns: (predicate, [args...], is_positive_literal)
        Facts appear as '+pred[a,b]' or '-pred[a,b]'. We only index '+' facts for bitwise truth checks.
        """
        name = name.strip()
        is_pos = True
        if name.startswith('+'):
            name = name[1:]
            is_pos = True
        elif name.startswith('-'):
            name = name[1:]
            is_pos = False
        # Format: pred[arg1,arg2]
        bracket_idx = name.find('[')
        if bracket_idx > 0 and name.endswith(']'):
            pred = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args, is_pos
        # Fallbacks (shouldn't happen for facts, but keep robust)
        return None, [], is_pos

    def _parse_task_or_operator_name(self, name: str) -> Tuple[str, List[str]]:
        """
        Robust name parsing for grounded tasks/operators. Supports:
          - 'TaskName[arg1,arg2]'
          - 'TaskName(arg1,arg2)'
          - 'TaskName arg1 arg2'
        Returns (predicate, [args...]) where predicate is the symbol name.
        """
        s = name.strip()
        # Bracket form
        lb = s.find('[')
        if lb > 0 and s.endswith(']'):
            pred = s[:lb].strip()
            args_str = s[lb + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred, args
        # Parenthesis form
        lp = s.find('(')
        if lp > 0 and s.endswith(')'):
            pred = s[:lp].strip()
            args_str = s[lp + 1:-1]
            # allow comma or whitespace separated
            if ',' in args_str:
                args = [a.strip() for a in args_str.split(',') if a.strip()]
            else:
                args = [a for a in args_str.split() if a]
            return pred, args
        # Whitespace-separated
        parts = s.split()
        if not parts:
            return s, []
        return parts[0], parts[1:]

    # ---------------------------- Task-network processing ----------------------------
    def _extract_pending_shot_cocktail_pairs(self, task_network: List[Union[int, object]], state_bitwise: int) -> List[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()

        for item in task_network:
            comp = self.model.get_component(item) if isinstance(item, int) else item
            # AbstractTask has 'decompositions' attribute; Operator has 'pos_precons'
            name = getattr(comp, 'name', '')
            pred, args = self._parse_task_or_operator_name(name)
            if not pred:
                continue

            # Abstract task cases
            if hasattr(comp, 'decompositions'):
                if pred == 'AchieveContainsShotCocktail' and len(args) >= 2:
                    shot, cocktail = args[0], args[1]
                    pairs.add((shot, cocktail))
                elif pred == 'DoPourShakerToShot' and len(args) >= 3:
                    # DoPourShakerToShot(shaker, shot, cocktail)
                    shot, cocktail = args[1], args[2]
                    pairs.add((shot, cocktail))
            else:
                # Operator cases
                if pred == 'pour-shaker-to-shot' and len(args) >= 2:
                    # Operator signature: (cocktail, shot, hand, shaker, level, level)
                    cocktail, shot = args[0], args[1]
                    pairs.add((shot, cocktail))

        return list(pairs)

    # ---------------------------- State queries ----------------------------
    def _is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _shot_pour_fix_cost(self, state_bitwise: int, shot: str) -> int:
        # At least 1 action if not empty, and at least 1 if not clean
        c = 0
        id_empty = self.empty_pos.get(shot)
        if id_empty is not None and not self._is_true(state_bitwise, id_empty):
            c += 1
        id_clean = self.clean_pos.get(shot)
        if id_clean is not None and not self._is_true(state_bitwise, id_clean):
            c += 1
        return c

    def _group_creation_reduction_from_current_state(self, state_bitwise: int, cocktail: str) -> int:
        """
        Default creation cost for a cocktail group is 5 (2 fills + 2 pours + 1 shake).
        This function returns how much we can reduce that 5 based on current shaker contents:
            - If any shaker contains the cocktail and is shaked: reduction = 5 (creation cost -> 0)
            - Else if any shaker contains both required ingredients and is unshaked: reduction = 4 (5 -> 1)
            - Else if any shaker contains exactly one required ingredient and is unshaked: reduction = 2 (5 -> 3)
            - Else: reduction = 0
        """
        parts = self.cocktail_parts.get(cocktail, (None, None))
        ing1, ing2 = parts[0], parts[1]

        # If we don't know parts, no reduction possible
        if ing1 is None or ing2 is None:
            # But cocktail already in shaker is still a reduction path
            # Check only that first
            for sh in self._shakers:
                shaked_id = self.shaked_pos.get(sh)
                cont_id = self.contains_pos.get((sh, cocktail))
                if self._is_true(state_bitwise, shaked_id) and self._is_true(state_bitwise, cont_id):
                    return 5
            return 0

        # Check all shakers for best possible reduction
        best = 0
        for sh in self._shakers:
            # 1) Exact cocktail already in shaker and shaked
            if self._is_true(state_bitwise, self.shaked_pos.get(sh)) and \
               self._is_true(state_bitwise, self.contains_pos.get((sh, cocktail))):
                best = max(best, 5)
                continue  # cannot get better than 5

            # 2) Ingredients present in unshaked shaker
            if not self._is_true(state_bitwise, self.unshaked_pos.get(sh)):
                continue
            has1 = self._is_true(state_bitwise, self.contains_pos.get((sh, ing1)))
            has2 = self._is_true(state_bitwise, self.contains_pos.get((sh, ing2)))
            if has1 and has2:
                best = max(best, 4)
            elif has1 or has2:
                best = max(best, 2)

        return best


# Alias required name for external discovery (optional convenience)
__all__ = ["BarmanGPT5Heuristic15"]
