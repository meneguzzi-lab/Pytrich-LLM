from typing import Dict, List, Set, Tuple, Union

# Attempt to import the planner's Heuristic base. Provide a minimal fallback if unavailable.
try:
    from Heuristics.Heuristic import Heuristic  # Typical Pytrich path
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic  # Alternate path
    except Exception:
        class Heuristic:  # Fallback to allow static checks
            def __init__(self, model):
                self.model = model
            def __call__(self, state, task_network):
                return 0


class BarmanGPT5Heuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    High-level idea (admissible lower bound):
    - To get a shot to contain a cocktail, at least one pour-shaker-to-shot action is required per (shot, cocktail) pair not yet satisfied in the state.
    - Preparing a cocktail in a shaker requires shaking after having both ingredients in the shaker. In the best case, if both ingredients are already in a shaker (unshaken), only one shake is needed. Otherwise, each missing ingredient requires at least:
        * one pour-shot-to-(clean/used)-shaker action (to transfer from a shot), and
        * if no shot currently contains that ingredient, at least one fill-shot action to produce it.
      Thus, for a cocktail c with two ingredients (i1, i2), a lower bound for preparing c in some shaker s is:
        cost_s(c) = (missing_i1_cost + missing_i2_cost) + 1 (shake)
        where missing_ik_cost = 0 if contains(s, ik) already holds;
                                else 1 (pour-to-shaker) + (0 if any shot already contains ik else 1 (fill-shot))
      We then take the minimum over all shakers s. If any shaker already contains c, the cost is 0.

    Overall heuristic value:
        h = (#pending distinct pours) + sum_{distinct cocktails still needed} min_s cost_s(c)

    Notes:
    - We compute the set of pending (shot, cocktail) pairs from the current task network by detecting
      AbstractTasks 'AchieveContainsShotCocktail' and 'DoPourShakerToShot'. We also detect primitive
      operator 'pour-shaker-to-shot' if present in the network. Already-satisfied pairs (contains(shot, cocktail) true in state) are ignored.
    - We compute the set of cocktails that will need to be prepared from these pairs, plus any explicitly
      present 'AchieveContainsShakerCocktail' tasks that are not yet satisfied in the state.
    - We purposefully ignore costs for hand constraints (grasp/drop), cleanliness of shots/shakers, etc.,
      because accounting for them without overestimation is tricky; ignoring them keeps the heuristic admissible.
    - We exploit domain structure (two ingredients per cocktail, reusable shaker cocktail for many pours)
      while ensuring the estimate never exceeds the optimal remaining cost (admissibility).

    Implementation details:
    - Uses the model's bitwise state to test fact truth quickly via precomputed global_id masks.
    - Parses grounded component names using the [+/-] or bracket/parenthesis style described by the framework.
    - Precomputes indexes for quick lookup: contains facts, shakers, shots, ingredient availability masks,
      and cocktail->ingredients mapping from static cocktailPart1/2 facts.

    This heuristic scales to large instances by relying on bitwise checks and precomputed masks.
    """

    def __init__(self, model, barman_gpt5_heuristic_0: bool = True):
        super().__init__(model)
        self.barman_gpt5_heuristic_0 = barman_gpt5_heuristic_0

        # Precomputed indices
        self.contains_id: Dict[Tuple[str, str], int] = {}
        self.shakers: Set[str] = set()
        self.containers: Set[str] = set()
        self.shots: Set[str] = set()
        self.cocktail_parts: Dict[str, Tuple[str, str]] = {}  # cocktail -> (ing1, ing2)
        self.cocktails: Set[str] = set()
        # For quick check: any shot contains ingredient i => bitmask of contains[shot, i] facts
        self.ingredient_in_any_shot_mask: Dict[str, int] = {}

        self._precompute_domain_indexes()

    # ------------- Core Heuristic API -------------
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]]):
        return self.h(state_bitwise, task_network)

    def h(self, state_bitwise: int, task_network: List[Union[object, object]]):
        # Collect pending (shot, cocktail) pairs and cocktails to be prepared
        pour_pairs, shaker_cocktail_needs = self._collect_demands(state_bitwise, task_network)

        # Count pours still required (one per distinct unsatisfied shot-cocktail pair)
        pour_count = len(pour_pairs)

        # Cocktails required to be available in some shaker (deduplicate by cocktail)
        cocktails_required: Set[str] = set([c for (_, c) in pour_pairs]) | shaker_cocktail_needs

        # Sum minimal prep costs per distinct cocktail not yet in any shaker
        prep_cost = 0
        for c in cocktails_required:
            prep_cost += self._prep_cost_for_cocktail(state_bitwise, c)

        h_val = pour_count + prep_cost
        if h_val < 0:
            h_val = 0
        return h_val

    # ------------- Domain Analysis and Precomputation -------------
    def _precompute_domain_indexes(self):
        # Collect shakers from shakerLevel/shakerEmptyLevel/shaked/unshaked facts
        for f in self.model.facts:
            pred, args = self._parse_component_name(f.name)
            if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and len(args) >= 1:
                self.shakers.add(args[0])

        # Parse cocktail parts from static facts
        for f in self.model.facts:
            pred, args = self._parse_component_name(f.name)
            if pred == "cocktailPart1" and len(args) == 2:
                c, i1 = args[0], args[1]
                if c not in self.cocktail_parts:
                    self.cocktail_parts[c] = (i1, None)
                else:
                    self.cocktail_parts[c] = (i1, self.cocktail_parts[c][1])
                self.cocktails.add(c)
            elif pred == "cocktailPart2" and len(args) == 2:
                c, i2 = args[0], args[1]
                if c not in self.cocktail_parts:
                    self.cocktail_parts[c] = (None, i2)
                else:
                    self.cocktail_parts[c] = (self.cocktail_parts[c][0], i2)
                self.cocktails.add(c)

        # Build contains mapping and container sets
        for f in self.model.facts:
            pred, args = self._parse_component_name(f.name)
            if pred == "contains" and len(args) == 2:
                cont, bev = args[0], args[1]
                self.contains_id[(cont, bev)] = f.global_id
                self.containers.add(cont)

        # Deduce shots as containers that are not recognized as shakers
        self.shots = {c for c in self.containers if c not in self.shakers}

        # For each ingredient that appears in cocktail parts, build mask of any shot containing it
        # Note: We derive ingredients from cocktail_parts values.
        ingredients: Set[str] = set()
        for c, (i1, i2) in self.cocktail_parts.items():
            if i1 is not None:
                ingredients.add(i1)
            if i2 is not None:
                ingredients.add(i2)

        for ing in ingredients:
            mask = 0
            for shot in self.shots:
                fact_id = self.contains_id.get((shot, ing))
                if fact_id is not None:
                    mask |= (1 << fact_id)
            self.ingredient_in_any_shot_mask[ing] = mask

    # ------------- Demand Extraction from Task Network -------------
    def _collect_demands(self, state_bitwise: int, task_network: List[Union[object, object]]):
        """
        Extracts:
        - pour_pairs: set of (shot, cocktail) still to be achieved (not already true in the state)
        - shaker_cocktail_needs: set of cocktails that need to be achieved in a shaker (from tasks), and not already true
        """
        pour_pairs: Set[Tuple[str, str]] = set()
        shaker_cocktail_needs: Set[str] = set()

        for comp in task_network:
            try:
                name = comp.name
            except Exception:
                # Skip unknown components defensively
                continue

            pred, args = self._parse_component_name(name)
            if not pred:
                continue

            # High-level goal: AchieveContainsShotCocktail[shot,cocktail]
            if pred == "AchieveContainsShotCocktail" and len(args) >= 2:
                shot, cocktail = args[0], args[1]
                if not self._state_has_contains(state_bitwise, shot, cocktail):
                    pour_pairs.add((shot, cocktail))

            # Task representing the pouring step: DoPourShakerToShot[shaker,shot,cocktail]
            elif pred == "DoPourShakerToShot" and len(args) >= 3:
                shot, cocktail = args[1], args[2]
                if not self._state_has_contains(state_bitwise, shot, cocktail):
                    pour_pairs.add((shot, cocktail))

            # Primitive operator already placed in the TN: pour-shaker-to-shot[cocktail,shot,hand,shaker,level,level]
            elif pred == "pour-shaker-to-shot" and len(args) >= 2:
                # operator signature: (cocktail, shot, hand, shaker, level, level)
                cocktail, shot = args[0], args[1]
                if not self._state_has_contains(state_bitwise, shot, cocktail):
                    pour_pairs.add((shot, cocktail))

            # Shaker preparation requirement: AchieveContainsShakerCocktail[shaker,cocktail]
            elif pred == "AchieveContainsShakerCocktail" and len(args) >= 2:
                shaker, cocktail = args[0], args[1]
                if not self._state_has_contains(state_bitwise, shaker, cocktail):
                    shaker_cocktail_needs.add(cocktail)

        return pour_pairs, shaker_cocktail_needs

    # ------------- Cost Computations -------------
    def _prep_cost_for_cocktail(self, state_bitwise: int, cocktail: str) -> int:
        """
        Minimal lower bound number of actions to make cocktail available in some shaker.
        Returns 0 if some shaker already contains the cocktail.
        Otherwise returns min over shakers of: for each missing ingredient, (1 pour) + (0 or 1 fill), plus (1 shake).
        If cocktail parts are unknown or no shakers exist, returns 0 for admissibility.
        """
        # Already available in some shaker?
        for s in self.shakers:
            if self._state_has_contains(state_bitwise, s, cocktail):
                return 0

        parts = self.cocktail_parts.get(cocktail)
        if not parts or parts[0] is None or parts[1] is None:
            # Unknown structure; return 0 to avoid any overestimation
            return 0
        ing1, ing2 = parts[0], parts[1]

        if not self.shakers:
            # No shakers (degenerate); return 0 to keep admissible
            return 0

        best = None
        for s in self.shakers:
            # For each ingredient, check if already present in shaker
            cost = 0
            # Ingredient 1
            if not self._state_has_contains(state_bitwise, s, ing1):
                pour_cost = 1
                has_shot_i1 = (self.ingredient_in_any_shot_mask.get(ing1, 0) & state_bitwise) != 0
                fill_cost = 0 if has_shot_i1 else 1
                cost += pour_cost + fill_cost
            # Ingredient 2
            if not self._state_has_contains(state_bitwise, s, ing2):
                pour_cost = 1
                has_shot_i2 = (self.ingredient_in_any_shot_mask.get(ing2, 0) & state_bitwise) != 0
                fill_cost = 0 if has_shot_i2 else 1
                cost += pour_cost + fill_cost
            # One shake to convert ingredients into the cocktail
            cost += 1

            if best is None or cost < best:
                best = cost

        return best if best is not None else 0

    # ------------- Utilities -------------
    def _state_has_contains(self, state_bitwise: int, container: str, beverage: str) -> bool:
        fact_id = self.contains_id.get((container, beverage))
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    @staticmethod
    def _parse_component_name(name: str) -> Tuple[str, List[str]]:
        """
        Generic grounded name parser supporting both [args] and (args) formats.
        Examples:
          'AchieveContainsShotCocktail[shot2,cocktail1]' -> ('AchieveContainsShotCocktail', ['shot2','cocktail1'])
          'pour-shaker-to-shot[cocktail1,shot2,left,shaker1,level1,level1]' -> ('pour-shaker-to-shot', [...])
          '+contains[shot2,cocktail1]' -> ('contains', ['shot2','cocktail1'])
        """
        s = name.strip()
        # Drop +/- prefix if present
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        # Find opening bracket or parenthesis
        b_idx = s.find('[')
        p_idx = s.find('(')
        if b_idx == -1 and p_idx == -1:
            return s, []
        use_paren = (p_idx != -1) and (b_idx == -1 or p_idx < b_idx)
        if use_paren:
            l_idx = p_idx
            r_char = ')'
        else:
            l_idx = b_idx
            r_char = ']'
        pred = s[:l_idx]
        r_idx = s.rfind(r_char)
        if r_idx == -1:
            r_idx = len(s)
        inner = s[l_idx + 1:r_idx]
        args = [a.strip() for a in inner.split(',') if a.strip()]
        return pred, args


# Alias required by some loaders
barman_gpt5_heuristic_0 = BarmanGPT5Heuristic0
