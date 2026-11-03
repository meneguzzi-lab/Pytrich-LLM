import re
from typing import Dict, Tuple, List, Optional, Set

# Fallback-safe import of Heuristic base class
try:
    from Pytrich.Heuristics.heuristic import Heuristic  # Typical Pytrich location
except Exception:
    try:
        from .heuristic import Heuristic
    except Exception:
        class Heuristic(object):  # Minimal stub to avoid import errors in static checks
            def __init__(self, model):
                self.model = model


class BarmanGPT5Heuristic4(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Core idea (admissible lower bound):
    - Each goal "AchieveContainsShotCocktail(shot, cocktail)" requires (at least):
      - The receiving shot must be empty and clean (otherwise: one empty-shot and/or one clean-shot action).
      - The shaker must contain the cocktail and be shaked, or it must be prepared with the cocktail:
        * If shaker already contains the cocktail and is shaked: 0 actions to prepare the shaker.
        * Else: (possibly) empty the shaker if it is currently shaked with another cocktail (1 action),
                 add the missing ingredients (each ingredient: at least one fill-shot + one pour-shot-to-... = 2 actions),
                 then shake once (1 action).
      - Finally, pour the shaker to the shot (1 action).

    - We ignore grasp/drop/holding bookkeeping in the lower bound (admissible under-estimation) but
      we do count the primitive domain actions that are strictly necessary in every valid plan: fill-shot,
      pour-shot-to-..., shake, empty-shot, clean-shot, empty-shaker, pour-shaker-to-shot.

    - For multiple goals (multiple shots/cocktails), we compute the sum of the per-goal lower bounds from the
      current state. This remains admissible because it measures each goal from the same (shared) state
      and ignores possible additional costs that appear only after executing other goals (under-estimation).

    Implementation details:
    - Uses bitwise checks for state membership (fast) via fact.global_id.
    - Parses grounded facts of the form +predicate[arg1,arg2].
    - Extracts goal (shot, cocktail) pairs from the initial_task_network by parsing task names containing
      "AchieveContainsShotCocktail". If this fails, the heuristic may return 0 (safe fallback but less informative).
    - Precomputes cocktail -> (ing1, ing2) and indexes the relevant facts for constant-time checks.

    Expected interface:
    - __init__(model, barman_gpt5_heuristic_4: bool=True)
    - __call__(state_bitwise: int, task_network: List) -> int
    """

    def __init__(self, model, barman_gpt5_heuristic_4: bool = True):
        super().__init__(model)
        self.enabled = barman_gpt5_heuristic_4
        self.model = model

        # Precomputed indices for quick lookup
        self._plus_fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._facts_by_pred: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}

        # Domain-specific maps
        self._cocktail_to_ings: Dict[str, Tuple[str, str]] = {}
        self._shakers: Set[str] = set()
        self._shots: Set[str] = set()

        # From initial TN, extract all (shot, cocktail) goals
        self._goals_sc: List[Tuple[str, str]] = []

        # Build indices from facts
        self._index_facts()
        # Collect (shot, cocktail) pairs from the initial task network
        self._extract_goals_from_initial_tn()

    # ------------------------ Public API ------------------------
    def __call__(self, state_bitwise: int, task_network) -> int:
        if not self.enabled:
            return 0
        try:
            return self._estimate_total(state_bitwise)
        except Exception:
            # In case of any unexpected parsing/runtime issue, remain admissible by returning 0
            return 0

    # ------------------------ Heuristic Core ------------------------
    def _estimate_total(self, state_bitwise: int) -> int:
        # Sum the per-goal lower bounds from the current state
        h_total = 0
        for shot, cocktail in self._goals_sc:
            # If already achieved, no cost
            if self._is_true(state_bitwise, 'contains', (shot, cocktail)):
                continue
            h_total += self._estimate_one_goal(state_bitwise, shot, cocktail)
        return max(0, h_total)

    def _estimate_one_goal(self, state_bitwise: int, shot: str, cocktail: str) -> int:
        # If the cocktail parts are unknown (shouldn't happen), return a minimal safe bound: 1 for final pour
        if cocktail not in self._cocktail_to_ings:
            return 1

        ing1, ing2 = self._cocktail_to_ings[cocktail]
        shaker = self._pick_any_shaker()
        if shaker is None:
            # No shaker found -> cannot achieve; return 0 to keep admissible (planner will handle dead-end)
            return 0

        cost = 0
        # 1) Ensure the receiving shot is ready: must be empty and clean for the final pour
        #    Each missing property requires at least one action (empty-shot or clean-shot).
        if not self._is_true(state_bitwise, 'empty', (shot,)):
            cost += 1  # empty-shot is necessary
        if not self._is_true(state_bitwise, 'clean', (shot,)):
            cost += 1  # clean-shot is necessary

        # 2) Ensure the shaker provides the required cocktail (and is shaked)
        if self._is_true(state_bitwise, 'contains', (shaker, cocktail)) and \
           self._is_true(state_bitwise, 'shaked', (shaker,)):
            shaker_prep = 0
        else:
            shaker_prep = 0
            # If shaker currently shaked with something else (i.e., shaked and not already containing this cocktail),
            # we must at least empty it to allow adding ingredients again
            if self._is_true(state_bitwise, 'shaked', (shaker,)) and \
               not self._is_true(state_bitwise, 'contains', (shaker, cocktail)):
                shaker_prep += 1  # empty-shaker
                have_ing = 0  # after emptying, nothing remains
            else:
                have_ing = 0
                if self._is_true(state_bitwise, 'contains', (shaker, ing1)):
                    have_ing += 1
                if self._is_true(state_bitwise, 'contains', (shaker, ing2)):
                    have_ing += 1

            missing = max(0, 2 - have_ing)
            # Each missing ingredient minimally requires: fill-shot (1) + pour-shot-to-... (1) => 2 actions
            shaker_prep += 2 * missing
            # Finally, to obtain the cocktail we need to shake once (unless we already contained the cocktail)
            if not self._is_true(state_bitwise, 'contains', (shaker, cocktail)):
                shaker_prep += 1  # shake

        cost += shaker_prep

        # 3) Final pour from shaker to shot: at least 1 action
        cost += 1
        return cost

    # ------------------------ Fact utilities ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith('+') or name.startswith('-'):
            sign = name[0]
            body = name[1:]
            bracket_idx = body.find('[')
            if bracket_idx > 0 and body.endswith(']'):
                predicate = body[:bracket_idx]
                args_str = body[bracket_idx + 1:-1]
                args = tuple(a.strip() for a in args_str.split(',')) if args_str else tuple()
                return sign, predicate, args
        return None, None, None

    def _index_facts(self):
        # Build indices for positive facts only
        for fact in getattr(self.model, 'facts', []):
            sign, pred, args = self._parse_fact_name(getattr(fact, 'name', ''))
            if pred is None:
                continue
            if sign != '+':
                continue  # only index positive facts for truth checks

            key = (pred, args)
            self._plus_fact_by_key[key] = fact.global_id
            self._facts_by_pred.setdefault(pred, []).append((args, fact.global_id))

            # Build cocktail parts map
            if pred == 'cocktailPart1' and len(args) == 2:
                c, i1 = args
                cdata = list(self._cocktail_to_ings.get(c, (None, None)))
                cdata[0] = i1
                self._cocktail_to_ings[c] = (cdata[0], cdata[1])
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, i2 = args
                cdata = list(self._cocktail_to_ings.get(c, (None, None)))
                cdata[1] = i2
                self._cocktail_to_ings[c] = (cdata[0], cdata[1])

            # Collect shakers (from shaker-level predicates or any shaker-unique predicate we see)
            if pred in ('shakerLevel', 'shakerEmptyLevel', 'shaked', 'unshaked') and len(args) >= 1:
                self._shakers.add(args[0])

            # Try to detect shot objects via 'used' or 'clean/empty' when second arg is beverage
            if pred in ('contains', 'used', 'clean', 'empty', 'ontable') and len(args) >= 1:
                # We cannot be 100% sure, but in Barman, shots appear in contains/used/clean/empty and are not 'shaker1'
                if args[0] not in self._shakers:
                    # This may add both shots and other containers (e.g., other cups); still fine
                    self._shots.add(args[0])

        # Clean cocktail map: ensure tuples are fully populated
        for c, ingpair in list(self._cocktail_to_ings.items()):
            i1, i2 = ingpair
            if i1 is None or i2 is None:
                # If partially missing, remove to avoid wrong estimates
                self._cocktail_to_ings.pop(c, None)

    def _is_true(self, state_bitwise: int, pred: str, args: Tuple[str, ...]) -> bool:
        gid = self._plus_fact_by_key.get((pred, args))
        if gid is None:
            return False
        return (state_bitwise >> gid) & 1 == 1

    def _pick_any_shaker(self) -> Optional[str]:
        # Prefer deterministic pick for stability
        if not self._shakers:
            # Attempt to infer from any 'shakerLevel' facts if not seen
            for args, _gid in self._facts_by_pred.get('shakerLevel', []):
                if len(args) >= 1:
                    self._shakers.add(args[0])
        if not self._shakers:
            return None
        return sorted(self._shakers)[0]

    # ------------------------ Task parsing utilities ------------------------
    def _extract_goals_from_initial_tn(self):
        self._goals_sc = []
        visited = set()
        # The initial task network is a list of Operators or AbstractTasks
        initial_tn = getattr(self.model, 'initial_tn', [])
        for t in initial_tn:
            self._collect_goal_pairs_from_tasklike(t, self._goals_sc, visited)
        # Deduplicate while preserving order
        seen_pairs = set()
        unique_pairs = []
        for sc in self._goals_sc:
            if sc not in seen_pairs:
                unique_pairs.append(sc)
                seen_pairs.add(sc)
        self._goals_sc = unique_pairs

    def _collect_goal_pairs_from_tasklike(self, tasklike, acc: List[Tuple[str, str]], visited: set):
        if tasklike is None:
            return
        # Protect against cycles on exotic structures
        if id(tasklike) in visited:
            return
        visited.add(id(tasklike))

        name = getattr(tasklike, 'name', '')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        if not isinstance(name, str):
            name = str(name)

        pairs = self._parse_achieve_contains_shot_cocktail_from_name(name)
        for p in pairs:
            acc.append(p)

        # If this is a decomposition, descend into its task network
        tnet = getattr(tasklike, 'task_network', None)
        if tnet is not None and isinstance(tnet, (list, tuple)):
            for sub in tnet:
                self._collect_goal_pairs_from_tasklike(sub, acc, visited)

    def _parse_achieve_contains_shot_cocktail_from_name(self, name: str) -> List[Tuple[str, str]]:
        res: List[Tuple[str, str]] = []
        if 'AchieveContainsShotCocktail' not in name:
            return res

        # Try patterns with parentheses first: AchieveContainsShotCocktail(shotX,cocktailY)
        m = re.search(r'AchieveContainsShotCocktail\s*\(\s*([^,\s]+)\s*,\s*([^)\s]+)\s*\)', name)
        if m:
            shot = m.group(1).strip()
            cocktail = m.group(2).strip()
            res.append((shot, cocktail))
            return res

        # Try whitespace-separated: AchieveContainsShotCocktail shotX cocktailY
        m2 = re.search(r'AchieveContainsShotCocktail\s+([^,\s]+)\s+([^,\s]+)', name)
        if m2:
            shot = m2.group(1).strip()
            cocktail = m2.group(2).strip()
            res.append((shot, cocktail))
            return res

        # As a fallback, try bracketed style if ever used: AchieveContainsShotCocktail[shotX,cocktailY]
        m3 = re.search(r'AchieveContainsShotCocktail\s*\[\s*([^,\s]+)\s*,\s*([^]\s]+)\s*\]', name)
        if m3:
            shot = m3.group(1).strip()
            cocktail = m3.group(2).strip()
            res.append((shot, cocktail))
            return res

        return res


# Backwards-compatible alias (if the planner tries to import by variable name)
BarmanGPT5Heuristic4Class = BarmanGPT5Heuristic4
