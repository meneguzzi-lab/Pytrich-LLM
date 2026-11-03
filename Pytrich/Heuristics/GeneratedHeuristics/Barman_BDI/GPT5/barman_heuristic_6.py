from typing import Dict, Tuple, Set, Optional, List

# Pytrich base heuristic import: try common locations
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback path if project uses a package structure
    try:
        from Pytrich.Heuristics.Heuristic import Heuristic
    except Exception:
        # Last resort: define a minimal base to avoid import error during static checks
        class Heuristic:
            def __init__(self, model):
                self.model = model


class BarmanGPT5Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Design goals
    - Safe (admissible) lower bound by exploiting domain structure:
      We only add clear, unavoidable minimum costs for high-level cocktail-making tasks.
    - Hierarchy-aware: prefer estimating at the highest present abstraction layer in the
      current task network to avoid double-counting across overlapping subtasks.
    - Efficient: pre-index facts once; during evaluate(), use bitwise checks only.

    Key domain insights used
    - To get a shot containing a cocktail (AchieveContainsShotCocktail s c):
      At minimum, you must pour the (already made) cocktail from a shaker to that shot (1 op).
      If the cocktail is not yet in any shaker, it must be created via shaking two ingredients.
      Creating a cocktail in a shaker minimally requires, in the best case:
        * If some shaker already contains the cocktail: 0 ops.
        * If some shaker contains both required ingredients (unshaked): 1 op (shake).
        * Otherwise, for each missing ingredient, at least one transfer from a shot to the shaker
          (1 op) if some shot already has it, or 2 ops (fill-shot + pour-to-shaker) if not. Then 1 shake op.
      We use the minimum over all shakers (admissible lower bound even if a specific shaker is bound later).

    - Primitive operators already in the task network must be executed; we add their exact count to the
      estimate (this preserves admissibility and improves guidance).

    Admissibility/consistency safeguards
    - We compute estimates only at one abstraction layer per state (preferring AchieveContainsShotCocktail
      > AchieveContainsShakerCocktail > ingredient-level tasks). This avoids double counting shared work.
    - We only add costs that are minimal and strictly necessary given current state (using bitwise checks).
    - When we cannot reliably parse the parameters for a task, we default to 0 contribution (never overestimate).

    Implementation notes
    - Grounded fact names are in the "+pred[a,b]" form. We parse them once to build indexes.
    - We collect the set of cocktails from initial-state facts cocktailPart1/2.
    - We detect shakers and shots by scanning fact predicates (shakerLevel/shaked => shakers; containers
      minus shakers => shots).

    Parameters
    - barman_gpt5_heuristic_6 (bool): enable/disable the heuristic (default: True)
    """

    def __init__(self, model, barman_gpt5_heuristic_6: bool = True):
        super().__init__(model)
        self.enabled = barman_gpt5_heuristic_6

        # Fact indexes: predicate -> { args_tuple: Fact }
        self._pred_index: Dict[str, Dict[Tuple[str, ...], object]] = {}

        # Object role sets (derived from facts)
        self.shakers: Set[str] = set()
        self.shots: Set[str] = set()
        self.hands: Set[str] = set()
        self.containers: Set[str] = set()

        # Static mapping: cocktail -> {ingredient1, ingredient2}
        self.cocktail_parts: Dict[str, Set[str]] = {}
        self.cocktails: Set[str] = set()
        self.ingredients: Set[str] = set()

        self._build_fact_indexes()
        self._extract_roles_and_statics()

    # -------------------- Core Heuristic API --------------------
    def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
        if not self.enabled or task_network is None or len(task_network) == 0:
            return 0

        # Gather categories to be hierarchy-aware
        primitive_ops = 0
        shot_cocktail_tasks: List[object] = []  # AchieveContainsShotCocktail
        shaker_cocktail_tasks: List[object] = []  # AchieveContainsShakerCocktail
        other_tasks: List[object] = []

        for item in task_network:
            # Duck-typing: Operator has 'add_effects' and 'pos_precons'
            if hasattr(item, 'add_effects') and hasattr(item, 'pos_precons'):
                primitive_ops += 1
            else:
                # AbstractTask-like: must have 'name'
                name = getattr(item, 'name', '')
                if 'AchieveContainsShotCocktail' in name:
                    shot_cocktail_tasks.append(item)
                elif 'AchieveContainsShakerCocktail' in name:
                    shaker_cocktail_tasks.append(item)
                else:
                    other_tasks.append(item)

        h = 0
        # Always include exact cost of primitive operators already in TN
        h += primitive_ops

        # Prefer estimating at the highest abstraction layer present
        if len(shot_cocktail_tasks) > 0:
            # For each AchieveContainsShotCocktail(s, c):
            #   - If contains(s, c) already holds -> 0
            #   - Else: 1 (pour) + LB(make shaker have c)
            for t in shot_cocktail_tasks:
                name = getattr(t, 'name', '')
                shot = self._find_any_token(name, self.shots)
                cocktail = self._find_any_token(name, self.cocktails)

                # If we can verify goal already holds, add 0; else add conservative LB
                if shot and cocktail and self._is_true(state_bitwise, 'contains', (shot, cocktail)):
                    continue

                if cocktail:
                    h += 1  # At least one pour from shaker to the target shot (if not already satisfied)
                    h += self._lb_make_cocktail_any_shaker(state_bitwise, cocktail)
                else:
                    # Unknown parameters -> stay admissible by adding nothing
                    # (we cannot guarantee it's not already satisfied)
                    pass
            return h

        if len(shaker_cocktail_tasks) > 0:
            # For each AchieveContainsShakerCocktail(x, c): LB to make shaker contain cocktail c
            for t in shaker_cocktail_tasks:
                name = getattr(t, 'name', '')
                cocktail = self._find_any_token(name, self.cocktails)
                if not cocktail:
                    continue  # cannot parse -> 0
                shaker = self._find_any_token(name, self.shakers)
                if shaker:
                    h += self._lb_make_cocktail_for_shaker(state_bitwise, shaker, cocktail)
                else:
                    h += self._lb_make_cocktail_any_shaker(state_bitwise, cocktail)
            return h

        # Fallback (no high-level cocktail tasks present):
        # We keep it conservative to preserve admissibility.
        # Optionally add tiny guidance for ingredient-level tasks if parsable.
        for t in other_tasks:
            name = getattr(t, 'name', '')

            if 'AchieveContainsShakerIngredient' in name:
                shaker = self._find_any_token(name, self.shakers)
                ing = self._find_any_token(name, self.ingredients)
                if shaker and ing:
                    if self._is_true(state_bitwise, 'contains', (shaker, ing)):
                        continue
                    # If any shot already has the ingredient -> 1 (pour to shaker), else 2 (fill+pour)
                    h += 1 if self._any_shot_contains(state_bitwise, ing) else 2
                # else unknown params -> 0

            elif 'AchieveContainsShotIngredient' in name:
                shot = self._find_any_token(name, self.shots)
                ing = self._find_any_token(name, self.ingredients)
                if shot and ing:
                    if self._is_true(state_bitwise, 'contains', (shot, ing)):
                        continue
                    # Minimal: one fill-shot action
                    h += 1
                # else unknown -> 0

            elif 'DoPourShakerToShot' in name:
                # Minimal: one pour-shot action if not already satisfied (and if we can parse params)
                shot = self._find_any_token(name, self.shots)
                cocktail = self._find_any_token(name, self.cocktails)
                if shot and cocktail:
                    if self._is_true(state_bitwise, 'contains', (shot, cocktail)):
                        continue
                    h += 1
                # else unknown -> 0

            # For AchieveCleanShot/AchieveCleanShaker/AchieveHolding/AchieveHandEmpty/AchieveOnTable,
            # we do not add anything here to avoid double-counting with operators and to stay conservative.

        return h

    # -------------------- Lower-bound helpers --------------------
    def _lb_make_cocktail_any_shaker(self, state_bitwise: int, cocktail: str) -> int:
        if len(self.shakers) == 0:
            return 0
        best = None
        for sh in self.shakers:
            cost = self._lb_make_cocktail_for_shaker(state_bitwise, sh, cocktail)
            if best is None or cost < best:
                best = cost
        return best if best is not None else 0

    def _lb_make_cocktail_for_shaker(self, state_bitwise: int, shaker: str, cocktail: str) -> int:
        # If shaker already has the cocktail
        if self._is_true(state_bitwise, 'contains', (shaker, cocktail)):
            return 0

        parts = self.cocktail_parts.get(cocktail)
        if not parts or len(parts) == 0:
            # Unknown recipe -> cannot assert any needed action safely
            return 0
        # Ensure exactly two ingredients if available, else consider all listed as required
        req_ings = list(parts)

        # Check if shaker already contains required ingredients
        have_flags = [self._is_true(state_bitwise, 'contains', (shaker, ing)) for ing in req_ings]
        if all(have_flags):
            # Only shake is needed to transform ingredients into cocktail
            return 1

        missing_cost = 0
        for ing, have in zip(req_ings, have_flags):
            if have:
                continue
            # Best case: some shot currently contains the needed ingredient -> 1 (pour to shaker)
            # Else: must create that shot with ingredient and then pour -> 2 ops (fill + pour)
            missing_cost += 1 if self._any_shot_contains(state_bitwise, ing) else 2

        # After both ingredients are (best-case) in the shaker, we still need to shake once
        missing_cost += 1
        return missing_cost

    def _any_shot_contains(self, state_bitwise: int, ingredient: str) -> bool:
        for shot in self.shots:
            if self._is_true(state_bitwise, 'contains', (shot, ingredient)):
                return True
        return False

    # -------------------- Parsing and indexing utilities --------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name = name[1:]
        bracket_idx = name.find('[')
        if bracket_idx > 0 and name.endswith(']'):
            predicate = name[:bracket_idx]
            args_str = name[bracket_idx + 1:-1]
            args = args_str.split(',') if args_str else []
            return predicate, args
        return None, []

    def _index_fact(self, predicate: str, args: Tuple[str, ...], fact_obj: object):
        bucket = self._pred_index.get(predicate)
        if bucket is None:
            bucket = {}
            self._pred_index[predicate] = bucket
        bucket[args] = fact_obj

    def _get_fact_obj(self, predicate: str, args: Tuple[str, ...]):
        bucket = self._pred_index.get(predicate)
        if not bucket:
            return None
        return bucket.get(args)

    def _get_fact_id(self, predicate: str, args: Tuple[str, ...]) -> Optional[int]:
        fact_obj = self._get_fact_obj(predicate, args)
        return getattr(fact_obj, 'global_id', None) if fact_obj is not None else None

    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        fid = self._get_fact_id(predicate, args)
        if fid is None:
            return False
        return (state_bitwise >> fid) & 1 == 1

    def _find_any_token(self, text: str, candidates: Set[str]) -> Optional[str]:
        # Return the first candidate that appears as a token in the text (with simple boundary checks)
        if not text or not candidates:
            return None
        for cand in candidates:
            start = 0
            while True:
                idx = text.find(cand, start)
                if idx == -1:
                    break
                left_ok = (idx == 0) or (not text[idx - 1].isalnum() and text[idx - 1] != '_')
                ridx = idx + len(cand)
                right_ok = (ridx == len(text)) or (not text[ridx].isalnum() and text[ridx] != '_')
                if left_ok and right_ok:
                    return cand
                start = idx + 1
        return None

    def _build_fact_indexes(self):
        # Index all grounded facts by predicate and arguments
        for f in getattr(self.model, 'facts', []):
            name = getattr(f, 'name', '')
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            self._index_fact(pred, tuple(args), f)

    def _extract_roles_and_statics(self):
        # Parse initial state to build cocktail recipes and object-role sets
        # 1) Cocktails and recipes from initial true facts
        try:
            init_facts = self.model.state_explicit_repr(self.model.initial_state)
        except Exception:
            init_facts = []

        for name in init_facts:
            pred, args = self._parse_fact_name(name)
            if pred == 'cocktailPart1' and len(args) == 2:
                c, ing = args[0], args[1]
                self.cocktails.add(c)
                self.ingredients.add(ing)
                self.cocktail_parts.setdefault(c, set()).add(ing)
            elif pred == 'cocktailPart2' and len(args) == 2:
                c, ing = args[0], args[1]
                self.cocktails.add(c)
                self.ingredients.add(ing)
                self.cocktail_parts.setdefault(c, set()).add(ing)
            elif pred == 'handEmpty' and len(args) == 1:
                self.hands.add(args[0])
            elif pred in ('ontable', 'clean', 'empty') and len(args) == 1:
                self.containers.add(args[0])
            elif pred in ('shakerLevel', 'shakerEmptyLevel', 'shaked', 'unshaked') and len(args) >= 1:
                self.shakers.add(args[0])

        # If shakers were not found via init facts, search all facts as fallback
        if len(self.shakers) == 0:
            for (pred, bucket) in self._pred_index.items():
                if pred in ('shakerLevel', 'shakerEmptyLevel', 'shaked', 'unshaked'):
                    for args in bucket.keys():
                        if len(args) >= 1:
                            self.shakers.add(args[0])

        # Collect containers more broadly (ontable/clean/empty in all facts)
        if len(self.containers) == 0:
            for pred in ('ontable', 'clean', 'empty'):
                bucket = self._pred_index.get(pred)
                if bucket:
                    for args in bucket.keys():
                        if len(args) == 1:
                            self.containers.add(args[0])

        # Shots = containers - shakers (by name). In this domain, this separation is safe.
        self.shots = set(x for x in self.containers if x not in self.shakers)

        # Ingredients set may be empty if not found via init; fallback: scan contains facts with shots
        if len(self.ingredients) == 0:
            contains_bucket = self._pred_index.get('contains', {})
            for (a1, a2) in contains_bucket.keys():
                # Heuristic fallback: if the beverage appears with any shot in contains, treat it as ingredient candidate
                if a1 in self.shots:
                    self.ingredients.add(a2)

        # Cocktails set may also be empty if not found via init; fallback: scan contains facts with shaker
        if len(self.cocktails) == 0:
            contains_bucket = self._pred_index.get('contains', {})
            for (a1, a2) in contains_bucket.keys():
                # Use the heuristic: beverages that appear in shaker contains coupled with shaked are cocktails
                if a1 in self.shakers:
                    self.cocktails.add(a2)

        # Ensure cocktail_parts exist for known cocktails (may be empty if not fully parsable)
        for c in list(self.cocktails):
            self.cocktail_parts.setdefault(c, set())
