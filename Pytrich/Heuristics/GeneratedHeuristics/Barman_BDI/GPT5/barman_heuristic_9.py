from typing import Dict, Tuple, List, Set, Optional

# Robust import for the base Heuristic in Pytrich
try:
    from .heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic
    except Exception:
        # Minimal fallback to allow static checks; real framework provides Heuristic
        class Heuristic:
            def __init__(self, model=None):
                self.model = model
            def __call__(self, state, task_network):
                return 0


class BarmanGPT5Heuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Barman HTN domain.

    Rationale (lower-bound, admissible):
    - Goal structure: each goal is (contains shot_i cocktail_j). We extract these from model.goals.
    - Serving a shot from a shaker that already contains that cocktail (and is shaked):
        * Needs at least 1 pour action per goal.
        * If the target shot is not clean/empty, additional actions are required (lower-bounded by 1 or 2).
          - If the shot currently contains any beverage -> at least 2 actions: empty-shot + clean-shot.
          - Else add 1 for each missing property (not empty, not clean) since pour requires both.
    - Producing a cocktail in a shaker from scratch (if no shaker currently contains it):
        * At least 5 actions: 2 fill-shot + 2 pour-shot-to-shaker + 1 shake. We ignore grasps/drops to keep it a safe LB.
    - Switching between different cocktails in a single shaker requires cleaning cycles:
        * A minimal inter-cocktail switch costs at least 2 actions: empty-shaker + clean-shaker.
        * If we need to produce m distinct cocktails not already present in any shaker, then:
          - If there exists a clean&empty shaker, at least (m-1) switches are needed.
          - If the shaker currently contains some cocktail (whether needed or not) or is non-empty, at least m switches are needed.
    - We sum per-goal serving costs (they cannot be shared) and add production + switching LBs across cocktails.

    Implementation details:
    - Uses fast bitwise membership checks (state >> fact.global_id) & 1.
    - Parses goal facts via model.state_explicit_repr(model.goals) and a robust fact-name parser.
    - Pre-indexes facts to avoid string scans at runtime: (predicate, args) -> Fact.
    - Derives cocktails from static facts (cocktailPart1/2) and ingredients from dispenses/parts.
    - Works on both small and large instances; only linear scans over relevant indexed structures.

    Admissibility note:
    - We never count grasp/drop costs (which are necessary in real plans) and we only add cleaning/switching costs
      when they are provably necessary, hence we remain a lower bound on the true cost.
    """

    def __init__(self, model, barman_gpt5_heuristic_9: bool = True):
        super().__init__(model)
        self.name = "BarmanGPT5Heuristic9"

        # Core indices
        self.fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}

        # Domain object sets
        self.cocktail_names: Set[str] = set()
        self.ingredient_names: Set[str] = set()
        self.beverage_names: Set[str] = set()  # ingredients U cocktails
        self.shaker_names: Set[str] = set()

        # Quick-access maps for common checks
        self.contains_facts_by_container: Dict[str, Dict[str, object]] = {}
        self.clean_fact_by_container: Dict[str, object] = {}
        self.empty_fact_by_container: Dict[str, object] = {}
        self.shaked_fact_by_shaker: Dict[str, object] = {}

        # Build indices once
        self._build_indices()

    # ----------------------------
    # Indexing & Parsing Utilities
    # ----------------------------

    @staticmethod
    def _parse_fact_name(name: str):
        """
        Parse grounded fact names of the form '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Returns (predicate, [args]). Returns (None, []) if format does not match.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _add_fact_index(self, fact_obj):
        pred, args = self._parse_fact_name(fact_obj.name)
        if pred is None:
            return
        key = (pred, tuple(args))
        self.fact_by_key[key] = fact_obj

        # Populate helpful per-container maps
        if pred == "contains" and len(args) == 2:
            cont, bev = args[0], args[1]
            if cont not in self.contains_facts_by_container:
                self.contains_facts_by_container[cont] = {}
            self.contains_facts_by_container[cont][bev] = fact_obj
        elif pred == "clean" and len(args) == 1:
            self.clean_fact_by_container[args[0]] = fact_obj
        elif pred == "empty" and len(args) == 1:
            self.empty_fact_by_container[args[0]] = fact_obj
        elif pred == "shaked" and len(args) == 1:
            self.shaked_fact_by_shaker[args[0]] = fact_obj

        # Collect domain object sets from static relations
        if pred == "cocktailPart1" and len(args) == 2:
            self.cocktail_names.add(args[0])
            self.ingredient_names.add(args[1])
        if pred == "cocktailPart2" and len(args) == 2:
            self.cocktail_names.add(args[0])
            self.ingredient_names.add(args[1])
        if pred == "dispenses" and len(args) == 2:
            self.ingredient_names.add(args[1])
        if pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked") and len(args) >= 1:
            self.shaker_names.add(args[0])

    def _build_indices(self):
        # Index all facts by parsed (predicate, args)
        for fact in self.model.facts:
            self._add_fact_index(fact)
        # Beverages are ingredients U cocktails (containers can contain both types)
        self.beverage_names = set(self.ingredient_names) | set(self.cocktail_names)

    # ----------------------------
    # Bitwise State Checks
    # ----------------------------

    def _is_true(self, state_bitwise: int, predicate: str, args: Tuple[str, ...]) -> bool:
        fact = self.fact_by_key.get((predicate, tuple(args)))
        if not fact:
            return False
        return ((state_bitwise >> fact.global_id) & 1) == 1

    def _container_contains_any_beverage(self, state_bitwise: int, container: str) -> bool:
        mapping = self.contains_facts_by_container.get(container)
        if not mapping:
            return False
        for fact in mapping.values():
            if ((state_bitwise >> fact.global_id) & 1) == 1:
                return True
        return False

    def _shaker_contains_cocktail(self, state_bitwise: int, shaker: str, cocktail: str) -> bool:
        # Requires contains(shaker, cocktail). Since cocktails are created by shake, this implies shaked too.
        return self._is_true(state_bitwise, "contains", (shaker, cocktail))

    def _any_shaker_clean_and_empty(self, state_bitwise: int) -> bool:
        for sh in self.shaker_names:
            if self._is_true(state_bitwise, "clean", (sh,)) and self._is_true(state_bitwise, "empty", (sh,)):
                return True
        return False

    def _any_shaker_contains_any_cocktail(self, state_bitwise: int) -> bool:
        for sh in self.shaker_names:
            mapping = self.contains_facts_by_container.get(sh, {})
            # Only check cocktails
            for c in self.cocktail_names:
                fact = mapping.get(c)
                if fact and ((state_bitwise >> fact.global_id) & 1) == 1:
                    return True
        return False

    def _any_shaker_not_empty(self, state_bitwise: int) -> bool:
        for sh in self.shaker_names:
            if not self._is_true(state_bitwise, "empty", (sh,)):
                return True
        return False

    # ----------------------------
    # Heuristic Computation
    # ----------------------------

    def __call__(self, state_bitwise: int, task_network: List) -> int:
        # 1) Extract goal facts and filter for contains(shot, cocktail)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        pending_pairs: List[Tuple[str, str]] = []  # (shot, cocktail)

        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred != "contains" or len(args) != 2:
                continue
            shot, cocktail = args[0], args[1]
            # Only consider contains where the beverage is one of the domain cocktails
            if cocktail not in self.cocktail_names:
                continue
            # If goal already satisfied in the current state, skip
            if self._is_true(state_bitwise, "contains", (shot, cocktail)):
                continue
            pending_pairs.append((shot, cocktail))

        if not pending_pairs:
            return 0

        # 2) Group pending goals by cocktail
        goals_by_cocktail: Dict[str, List[str]] = {}
        for shot, cocktail in pending_pairs:
            goals_by_cocktail.setdefault(cocktail, []).append(shot)

        # 3) Per-goal lower bound: pour + minimal shot-preparation requirements
        #    - pour to shot: at least 1 action
        #    - if shot contains any beverage: at least 2 (empty + clean)
        #    - else +1 if not empty, +1 if not clean
        serving_cost = 0
        for cocktail, shots in goals_by_cocktail.items():
            for shot in shots:
                # Minimal preparations for the target shot
                if self._container_contains_any_beverage(state_bitwise, shot):
                    prep_lb = 2  # must empty then clean at least
                else:
                    need_empty = not self._is_true(state_bitwise, "empty", (shot,))
                    need_clean = not self._is_true(state_bitwise, "clean", (shot,))
                    prep_lb = (1 if need_empty else 0) + (1 if need_clean else 0)
                serving_cost += 1 + prep_lb  # 1 for the pour-shaker-to-shot action

        # 4) Production lower bound for cocktails not currently available in any shaker
        #    Each new cocktail produced requires at least 5 primitive actions (2 fills + 2 pours to shaker + 1 shake)
        m_prod = 0
        for cocktail, shots in goals_by_cocktail.items():
            has_in_some_shaker = False
            for sh in self.shaker_names:
                if self._shaker_contains_cocktail(state_bitwise, sh, cocktail):
                    has_in_some_shaker = True
                    break
            if not has_in_some_shaker:
                m_prod += 1
        production_cost = 5 * m_prod

        # 5) Switching/cleaning lower bound between different cocktails to be produced
        #    - If clean&empty shaker exists -> at least (m_prod - 1) switches
        #    - Else if some shaker contains a cocktail already or is non-empty -> at least m_prod switches
        #    Each switch (to new production) costs at least 2 actions: empty-shaker + clean-shaker
        if m_prod <= 0:
            switch_cleans = 0
        else:
            if self._any_shaker_clean_and_empty(state_bitwise):
                switch_cleans = max(0, m_prod - 1)
            elif self._any_shaker_contains_any_cocktail(state_bitwise) or self._any_shaker_not_empty(state_bitwise):
                switch_cleans = m_prod
            else:
                # Default conservative case (treat as neutral start)
                switch_cleans = max(0, m_prod - 1)
        switching_cost = 2 * switch_cleans

        # Total heuristic is sum of independent per-goal serving LBs plus global production/switching LBs
        h_val = serving_cost + production_cost + switching_cost
        if h_val < 0:
            h_val = 0
        return int(h_val)


# Alias to comply with possible factory/registration mechanisms
def barman_gpt5_heuristic_9(model):
    return BarmanGPT5Heuristic9(model)
