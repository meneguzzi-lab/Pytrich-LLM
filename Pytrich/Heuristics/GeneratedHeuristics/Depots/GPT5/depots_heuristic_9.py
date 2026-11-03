# Domain-dependent heuristic for HDDL Depots domain in Pytrich
# Class name: DepotsGPT5Heuristic9 (inherits from Heuristic)
# __init__ parameter name: depots_gpt5_heuristic_9 (lowercase)
#
# Rationale (short):
# - The Depots domain revolves around putting crates onto target surfaces, possibly across places, using hoists and trucks.
# - Achieving a goal on(c, s) always requires a final Drop action. Prior steps minimally include making c "lifting" (via Lift or Unload), and possibly transporting c (Load/Drive/Unload) if c and s are at different places. Clearing preconditions (clear c, clear s) add at least one action each if not satisfied.
# - We compute a safe, informative lower bound per goal by case analysis on the crate's current modality: lifting, in truck, or at some place. We add one-time minimal costs for required actions consistent with the domain (e.g., transport requires at least Load+Drive+Unload if c is not already in a truck; or Drive+Unload if it is).
# - The heuristic strictly uses fast bitwise checks for state membership.
# - We also add a basic lower bound on remaining primitive operators already present in the task network (each must be executed), without trying to parse abstract task parameters (to keep robust). The final estimate is the max of the goal-based bound and the operator-count bound, preserving admissibility.
#
# Notes:
# - We only use positive facts (names start with '+') for state checks.
# - We avoid heavy string-based state enumeration inside the heuristic; all checks are bitwise on fact.global_id.
# - The per-goal estimate is a sum across each distinct goal on-fact; this is admissible because each goal requires at least one final Drop specific to that goal's crate.
# - Clearing penalties (for clear c and clear s) are added at most once per goal and reflect minimal necessity: if not clear, at least one action is needed somewhere to establish it.

from typing import Dict, Tuple, List, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Depots HDDL domain.

    Estimation strategy (admissible lower bound):
    - For each goal on(c, s) not yet satisfied in the current state:
        base = 1 (must perform a Drop to achieve on(c, s))
        + cost_to_make_lifting(c) depending on whether c is already lifting, in a truck, or at some place.
        + minimal transport cost if c and s are at different places (depends on modality):
          * If lifting and places differ: Load + Drive + Unload = 3
          * If in a truck and places differ: Drive = 1
          * If at a place and places differ: Lift + Load + Drive + Unload = 4
        + 1 if clear(s) is false (must clear target surface at least once)
        + 1 if clear(c) is false and c is at a place (must clear c at least once before Lift)
    - If s place is unknown, we omit transport costs (keeps lower bound valid).
    - If c place is unknown, we conservatively use the minimal necessary actions known for the modality.

    Additional bound from task network:
    - Count remaining primitive operators present in the current task_network; each will require at least one step.
    - Final heuristic value is max(goal_based_bound, operator_count_bound). Using max keeps admissibility.

    Implementation details:
    - Facts are parsed using the provided +predicate[arg1,arg2] scheme.
    - All state checks are bitwise for speed.
    - We build indexing dictionaries for quick access to relevant facts: on, at, in, lifting, clear.
    """

    def __init__(self, model, depots_gpt5_heuristic_9: bool = True):
        super().__init__()
        self.model = model

        # Predicate -> {(arg_tuple): Fact}
        self._pred_index: Dict[str, Dict[Tuple[str, ...], object]] = {}

        # Quick typed collections inferred from fact names
        self._places: Set[str] = set()
        self._trucks: Set[str] = set()
        self._hoists: Set[str] = set()
        self._crates: Set[str] = set()
        self._pallets: Set[str] = set()

        # Build indices
        self._build_fact_indices()

        # Cache goal facts as parsed pairs for fast iteration
        self._goal_pairs: List[Tuple[str, str]] = self._extract_goal_on_pairs()

    # ------------------------ Public Heuristic API ------------------------
    def __call__(self, state_bitwise: int, task_network: List[object]) -> int:
        return self.h(state_bitwise, task_network)

    def compute(self, state_bitwise: int, task_network: List[object]) -> int:
        return self.h(state_bitwise, task_network)

    def h(self, state_bitwise: int, task_network: List[object]) -> int:
        # Goal-based bound
        g_cost = self._goals_lower_bound(state_bitwise)
        # Remaining primitive operators present in the TN: at least one step each
        tn_op_lb = self._task_network_primitive_lb(task_network)
        # Take max to preserve admissibility while adding guidance when operators are already scheduled
        return max(g_cost, tn_op_lb)

    # ------------------------ Index Building ------------------------
    def _build_fact_indices(self):
        # Build predicate -> facts dict for quick lookup
        pred_index: Dict[str, Dict[Tuple[str, ...], object]] = {}

        facts = self.model.facts
        for fact in facts:
            name = fact.name
            if not name or not isinstance(name, str):
                continue
            if not (name.startswith("+") or name.startswith("-")):
                # Unexpected formatting; skip
                continue
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            if name.startswith("+"):
                pred_index.setdefault(pred, {})[tuple(args)] = fact
                # Some quick type inference for objects
                if pred == 'at' and len(args) == 2:
                    obj, place = args[0], args[1]
                    if self._is_place(place):
                        self._places.add(place)
                    # Objects from 'at' are locatable: trucks, hoists, pallets, crates
                    if obj.startswith('truck'):
                        self._trucks.add(obj)
                    elif obj.startswith('hoist'):
                        self._hoists.add(obj)
                    elif obj.startswith('pallet'):
                        self._pallets.add(obj)
                    elif obj.startswith('crate'):
                        self._crates.add(obj)
                elif pred == 'in' and len(args) == 2:
                    c, t = args[0], args[1]
                    if c.startswith('crate'):
                        self._crates.add(c)
                    if t.startswith('truck'):
                        self._trucks.add(t)
                elif pred == 'lifting' and len(args) == 2:
                    h, c = args[0], args[1]
                    if h.startswith('hoist'):
                        self._hoists.add(h)
                    if c.startswith('crate'):
                        self._crates.add(c)
                elif pred == 'clear' and len(args) == 1:
                    x = args[0]
                    if x.startswith('pallet'):
                        self._pallets.add(x)
                    elif x.startswith('crate'):
                        self._crates.add(x)
                elif pred == 'on' and len(args) == 2:
                    x, y = args[0], args[1]
                    if x.startswith('crate'):
                        self._crates.add(x)
                    if y.startswith('pallet'):
                        self._pallets.add(y)
                    elif y.startswith('crate'):
                        self._crates.add(y)

        self._pred_index = pred_index

        # If places were not discovered sufficiently from 'at' facts, we can try all second arguments seen in 'at'
        if not self._places and 'at' in self._pred_index:
            for (_obj, pl) in self._pred_index['at'].keys():
                if self._is_place(pl):
                    self._places.add(pl)

    def _extract_goal_on_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                pairs.append((args[0], args[1]))
        return pairs

    # ------------------------ Core Goal-based Lower Bound ------------------------
    def _goals_lower_bound(self, state_bitwise: int) -> int:
        total = 0
        for c, s in self._goal_pairs:
            # If already satisfied, contributes 0
            if self._is_true(state_bitwise, 'on', (c, s)):
                continue
            total += self._estimate_on_cost(state_bitwise, c, s)
        return total

    def _estimate_on_cost(self, state_bitwise: int, crate: str, surf: str) -> int:
        # Always need at least one Drop to set on(crate, surf)
        cost = 1  # Drop

        # Target surface clearing if currently not clear
        if self._should_count_clear_target(state_bitwise, surf):
            cost += 1

        # Determine modalities and places
        # Check if crate is currently being lifted
        lifting_h, lifting_p = self._get_lifting_info(state_bitwise, crate)
        if lifting_h is not None:
            sp = self._get_place_of_surface(state_bitwise, surf)
            if sp is None:
                # Unknown target place: we cannot count transport; keep only Drop + optional clear(s)
                return cost
            # If crate is lifting at the same place as surf: only need Drop
            if lifting_p == sp:
                return cost
            # Else: need to transport lifted crate using Load + Drive + Unload, then Drop
            # (We already counted Drop above)
            cost += 3  # Load + Drive + Unload
            return cost

        # Not lifting: check IN truck
        in_t, t_place = self._get_in_truck_info(state_bitwise, crate)
        if in_t is not None:
            sp = self._get_place_of_surface(state_bitwise, surf)
            if sp is None:
                # Unknown target place: Unload + Drop at minimum
                return cost + 1  # Unload
            if t_place == sp:
                # Just Unload + Drop (Drop already counted)
                return cost + 1  # Unload
            else:
                # Need Drive + Unload + Drop (Drop already counted)
                return cost + 2  # Drive + Unload

        # Not lifting, not in truck: check AT place
        cp = self._get_place_of_object(state_bitwise, crate)
        sp = self._get_place_of_surface(state_bitwise, surf)

        # If crate at a place
        if cp is not None:
            # If crate itself is not clear, at least one action is needed to clear it before lifting
            if not self._is_true(state_bitwise, 'clear', (crate,)):
                cost += 1
            if sp is None:
                # Without known target place, minimal: Lift + Drop
                return cost + 1  # Lift
            if cp == sp:
                # Same place: Lift + Drop (Drop already in cost)
                return cost + 1  # Lift
            else:
                # Different place: Lift + Load + Drive + Unload + Drop (Drop already in cost)
                return cost + 4  # Lift + Load + Drive + Unload

        # Fallback: if we cannot determine any modality/location, at least Drop must occur, but we also
        # need to make the crate lifting somehow (at least 1). Keep minimal bound by adding 0 or 1?
        # We conservatively add 0 here because Drop already counted and we lack definite modality info.
        return cost

    # ------------------------ Task-network primitive lower bound ------------------------
    def _task_network_primitive_lb(self, task_network: List[object]) -> int:
        # Each primitive operator in the network must execute at least once
        if not task_network:
            return 0
        count_ops = 0
        # Operators are recognized by having attributes: global_id, name, and methods applicable/apply, etc.
        # The provided Operator class should be the one used in the model.
        for item in task_network:
            # Heuristic-safe detection: check class name or attribute presence
            if hasattr(item, 'apply') and hasattr(item, 'get_precons') and hasattr(item, 'name'):
                # Treat as Operator
                count_ops += 1
        return count_ops

    # ------------------------ State Query Helpers ------------------------
    def _is_true(self, state_bitwise: int, pred: str, args: Tuple[str, ...]) -> bool:
        fact = self._pred_index.get(pred, {}).get(tuple(args))
        if fact is None:
            return False
        return ((state_bitwise >> fact.global_id) & 1) == 1

    def _get_place_of_object(self, state_bitwise: int, obj: str) -> Optional[str]:
        """Return place p if +at[obj,p] is true, else None."""
        at_map = self._pred_index.get('at', {})
        # Iterate over known places (bounded, small)
        for p in self._places:
            fact = at_map.get((obj, p))
            if fact is not None and ((state_bitwise >> fact.global_id) & 1):
                return p
        return None

    def _get_place_of_surface(self, state_bitwise: int, surf: str) -> Optional[str]:
        # Same as object place; surf can be pallet or crate
        return self._get_place_of_object(state_bitwise, surf)

    def _get_in_truck_info(self, state_bitwise: int, crate: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (truck, place_of_truck) if +in[crate,truck] holds, else (None, None)."""
        in_map = self._pred_index.get('in', {})
        at_map = self._pred_index.get('at', {})
        for t in self._trucks:
            f = in_map.get((crate, t))
            if f is not None and ((state_bitwise >> f.global_id) & 1):
                # Find truck place
                for p in self._places:
                    ft = at_map.get((t, p))
                    if ft is not None and ((state_bitwise >> ft.global_id) & 1):
                        return t, p
                return t, None
        return None, None

    def _get_lifting_info(self, state_bitwise: int, crate: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (hoist, place_of_hoist) if +lifting[hoist,crate] holds, else (None, None)."""
        lifting_map = self._pred_index.get('lifting', {})
        at_map = self._pred_index.get('at', {})
        for h in self._hoists:
            f = lifting_map.get((h, crate))
            if f is not None and ((state_bitwise >> f.global_id) & 1):
                # Find hoist place
                for p in self._places:
                    fh = at_map.get((h, p))
                    if fh is not None and ((state_bitwise >> fh.global_id) & 1):
                        return h, p
                return h, None
        return None, None

    def _should_count_clear_target(self, state_bitwise: int, surf: str) -> bool:
        # Count 1 if clear(surf) is currently false.
        cf = self._pred_index.get('clear', {}).get((surf,))
        if cf is None:
            # If fact not present in model's positive fact list, be conservative and do not add
            return False
        return ((state_bitwise >> cf.global_id) & 1) == 0

    # ------------------------ Utilities ------------------------
    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_place(x: str) -> bool:
        return x.startswith('depot') or x.startswith('distributor')
