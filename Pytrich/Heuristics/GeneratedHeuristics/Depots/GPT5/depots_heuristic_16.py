from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

from Pytrich.Heuristics.heuristic import Heuristic
# Type hints for Pytrich components
try:
    from Model import Model
    from Operator import Operator
    from AbstractTask import AbstractTask
except Exception:
    # Minimal stubs for type hints
    class Model: ...
    class Operator: ...
    class AbstractTask: ...


class DepotsGPT5Heuristic16(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depots domain.

    Key idea (admissible lower bound):
    - Each goal is an 'on(crate, surface)' target. The only primitive action that establishes 'on' is Drop.
      Therefore, each unsatisfied goal requires at least one Drop. Additional required non-drive
      actions depend on the crate's transport status relative to the target surface's place:
        * If crate already lifted at the target place:    1 (Drop)
        * If crate in truck (ignoring Drive cost):        2 (Unload, Drop)
        * If crate and target at same place:              2 (Lift, Drop)
        * If crate and target at different places:        4 (Lift, Load, Unload, Drop)  [Drive ignored]
      We intentionally ignore Drive costs to remain admissible while still providing guidance.
      Load/Unload/Lift/Drop are per-crate essentials that cannot be shared between different goals,
      so summing them over unsatisfied goals remains admissible.

    - We also consider the current task network minimally by adding the costs of already-grounded
      primitive operators pending in the task network (excluding 'nop'). This is a lower bound on
      the cost-to-finish the network. To remain safely admissible when combined with goal-based
      estimation, we return the maximum of (goal-based LB) and (primitive-operator LB).

    Implementation details:
    - Strictly uses bitwise checks for state membership for speed and scalability (see project notes).
    - Pre-indexes facts by predicate and object to enable O(k) lookup per relevant object instead of
      scanning all facts.
    - Robustly handles cases where object's place is unknown (e.g., lifted but missing 'at') by
      optimistic assumptions to preserve admissibility.

    Expected benefits:
    - Stronger guidance than a trivial heuristic by leveraging Depots structure (stacking and
      necessary hoist/truck manipulations), while staying efficient for very large instances.
    """

    def __init__(self, model: Model, depots_gpt5_heuristic_16: bool = True):
        super().__init__(model)
        self.name = "DepotsGPT5Heuristic16"

        # Indices for fast state queries
        # at_index[obj] -> List[(place, fact_id)]
        self.at_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # in_index[crate] -> List[(truck, fact_id)]  (crate in truck)
        self.in_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # on_index[crate] -> List[(surface, fact_id)]
        self.on_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # lifting_index[crate] -> List[(hoist, fact_id)]  (hoist lifting crate)
        self.lifting_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # Direct mapping for on(c, s) -> fact_id for quick goal check
        self.on_fact_id: Dict[Tuple[str, str], int] = {}

        # Build indices over all facts in the model
        self._build_indices()

        # Pre-extract goal pairs (crate, surface) and fact ids for faster evaluation
        self.goal_pairs: List[Tuple[str, str, int]] = []
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                fid = self.on_fact_id.get((c, s), None)
                if fid is not None:
                    self.goal_pairs.append((c, s, fid))

    # --------------- Pytrich-required method ---------------
    def h(self, state_bitwise: int, task_network: List[Union[Operator, AbstractTask]]) -> int:
        # 1) Goal-based lower bound (sum over unsatisfied on-goals)
        goal_lb = 0
        for c, s, fid in self.goal_pairs:
            if self._is_true(state_bitwise, fid):
                continue  # goal already satisfied
            goal_lb += self._on_goal_lower_bound(state_bitwise, c, s)

        # 2) Task-network lower bound from already-grounded primitive operators (excluding nop)
        tn_lb = 0
        for task in task_network or []:
            # Only count primitive operators; they must be executed -> admissible LB
            if hasattr(task, 'pos_precons') and hasattr(task, 'add_effects'):
                # Likely an Operator instance
                name_l = str(getattr(task, 'name', '')).lower()
                if 'nop' in name_l:
                    continue  # nop contributes no guidance; ignoring stays admissible
                cost = int(getattr(task, 'cost', 1))
                tn_lb += max(0, cost)

        # Use max to stay safely admissible (we must both complete TN and reach goals)
        return max(goal_lb, tn_lb)

    # --------------- Internal helpers ---------------
    def _build_indices(self):
        for fact in self.model.facts:
            try:
                fname = fact.name
            except Exception:
                continue
            pred, args = self._parse_fact_name(fname)
            if not pred:
                continue
            if pred == 'at' and len(args) == 2:
                obj, place = args[0], args[1]
                self.at_index[obj].append((place, fact.global_id))
            elif pred == 'in' and len(args) == 2:
                crate, truck = args[0], args[1]
                self.in_index[crate].append((truck, fact.global_id))
            elif pred == 'on' and len(args) == 2:
                crate, surface = args[0], args[1]
                self.on_index[crate].append((surface, fact.global_id))
                self.on_fact_id[(crate, surface)] = fact.global_id
            elif pred == 'lifting' and len(args) == 2:
                hoist, crate = args[0], args[1]
                self.lifting_index[crate].append((hoist, fact.global_id))
            # clear/available are not needed for the LB but could be added if desired

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name of the form +predicate[arg1,arg2] or -predicate[arg1,arg2].
        Returns (predicate, [args]) or (None, []).
        """
        if not name:
            return None, []
        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
            bracket_idx = s.find("[")
            if bracket_idx > 0 and s.endswith("]"):
                predicate = s[:bracket_idx]
                args_str = s[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _get_place_of(self, state_bitwise: int, obj: str) -> Optional[str]:
        """Return the unique place where obj is currently located, or None if unknown."""
        for place, fid in self.at_index.get(obj, []):
            if self._is_true(state_bitwise, fid):
                return place
        return None

    def _get_truck_of_crate(self, state_bitwise: int, crate: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (truck, place_of_truck) if crate is in some truck, else (None, None)."""
        for truck, fid in self.in_index.get(crate, []):
            if self._is_true(state_bitwise, fid):
                return truck, self._get_place_of(state_bitwise, truck)
        return None, None

    def _get_hoist_lifting_crate(self, state_bitwise: int, crate: str) -> Tuple[Optional[str], Optional[str]]:
        """Return (hoist, place_of_hoist) if some hoist is lifting the crate, else (None, None)."""
        for hoist, fid in self.lifting_index.get(crate, []):
            if self._is_true(state_bitwise, fid):
                return hoist, self._get_place_of(state_bitwise, hoist)
        return None, None

    def _get_place_of_crate(self, state_bitwise: int, crate: str) -> Optional[str]:
        """
        Best-effort to get crate's place:
        - Prefer direct at(crate, place) if present
        - Else if lifted: place is hoist's place
        - Else if in truck: place is truck's place
        - Else if on(crate, s): use at(s, place)
        - Else: unknown
        """
        p = self._get_place_of(state_bitwise, crate)
        if p is not None:
            return p
        # lifted?
        _, p = self._get_hoist_lifting_crate(state_bitwise, crate)
        if p is not None:
            return p
        # in truck?
        _, p = self._get_truck_of_crate(state_bitwise, crate)
        if p is not None:
            return p
        # on other surface?
        for surface, fid in self.on_index.get(crate, []):
            if self._is_true(state_bitwise, fid):
                return self._get_place_of(state_bitwise, surface)
        return None

    def _on_goal_lower_bound(self, state_bitwise: int, crate: str, surface: str) -> int:
        """
        Admissible LB for achieving on(crate, surface):
        - Already achieved:                               0
        - Crate lifted at target place:                   1  (Drop)
        - Crate in truck (Drive ignored):                 2  (Unload, Drop)
        - Same place (neither lifted nor in truck):       2  (Lift, Drop)
        - Different places (Drive ignored):               4  (Lift, Load, Unload, Drop)
        - Unknown info:                                   1  (at least one Drop is needed)
        """
        # If already true, cost 0 (caller ensures unsatisfied, but keep for safety)
        fid = self.on_fact_id.get((crate, surface))
        if fid is not None and self._is_true(state_bitwise, fid):
            return 0

        # Places
        target_place = self._get_place_of(state_bitwise, surface)

        # If lifted
        hoist, hoist_place = self._get_hoist_lifting_crate(state_bitwise, crate)
        if hoist is not None:
            if target_place is not None and hoist_place == target_place:
                return 1  # Drop only
            else:
                # Load + Unload + Drop (ignoring Drive)
                return 3

        # If in a truck
        truck, truck_place = self._get_truck_of_crate(state_bitwise, crate)
        if truck is not None:
            # Unload + Drop (Drive ignored)
            return 2

        # Otherwise, on some surface or simply at some place
        crate_place = self._get_place_of_crate(state_bitwise, crate)
        if crate_place is None or target_place is None:
            # Minimal optimistic LB: at least one Drop is needed
            return 1

        if crate_place == target_place:
            # Lift + Drop
            return 2
        else:
            # Cross-place transfer (Drive ignored): Lift + Load + Unload + Drop
            return 4
