from typing import Dict, Tuple, List, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT5Heuristic13(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depot domain.

    Core idea (admissible, consistent, and efficient):
      - We exploit per-crate uniqueness of two necessary actions to place a crate onto a target surface:
          1) exactly one Drop per crate to achieve on[c, s2]
          2) exactly one pickup (either Lift or Unload) to get the crate into lifting state prior to Drop
        Summing these per unsatisfied goal crates yields a safe lower bound (these actions cannot be shared).

      - We further compute a per-goal state-dependent refinement cost capturing unavoidable extras beyond the
        base pickup+drop for the single hardest goal (taking max preserves admissibility):
          - Transport mismatch: if the crate's current place differs from the target surface place, you need at least
            Load+Drive+Unload (=3) if the crate is at/lifting; if it is already in a truck, you need at least one Drive (=1).
          - Target surface clearance: if the target surface is not clear, at least one Lift is required to clear it (=1).
        We take the maximum of these per-goal estimates to avoid overcounting shared actions (e.g., a Drive can serve
        multiple crates). Combining with the additive unique base via max keeps the heuristic admissible.

      - Final heuristic value:
            h(state) = max( U, max_i C_i )
        where U is the sum over all unsatisfied goals of (1 Drop + 1 pickup if not already lifting), and C_i is the
        per-goal refined estimate for crate i that includes pickup+drop plus the minimal additional transport and
        clearance efforts for that specific goal.

    Hierarchy awareness:
      - The domain tasks decompose to primitive actions Drop, Lift, Load, Unload, Drive. Our counts reflect the
        minimal primitive actions necessary to refine abstract tasks like do_put_on into executable steps while
        remaining admissible (lower bounds only). This captures the essence of method structures without incurring
        high overhead.

    Efficiency:
      - All state checks use fast bitwise membership via Fact.global_id and the given model state bitset.
      - We build compact indices once in __init__ to answer queries (locations, lifting, in, clear) in O(1)/O(k) time.

    Notes:
      - Goals are assumed to be '+on[c,s]' facts (as in provided instances). The heuristic gracefully handles any
        subset and ignores unrelated goals if present.
      - No slow calls to model.state_explicit_repr(state) inside evaluate (only at init for goals extraction).
    """

    def __init__(self, model, depots_gpt5_heuristic_13: Optional[dict] = None):
        super().__init__(model)
        self.model = model

        # Indices for fast fact lookup and state queries
        self.facts_by_pred_args: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self.on_index: Dict[Tuple[str, str], object] = {}
        self.at_index: Dict[Tuple[str, str], object] = {}
        self.at_index_by_obj: Dict[str, List[Tuple[str, object]]] = {}
        self.in_index_by_crate: Dict[str, List[Tuple[str, object]]] = {}
        self.lifting_index_by_crate: Dict[str, List[Tuple[str, object]]] = {}
        self.clear_index: Dict[str, object] = {}

        # For quick access to all places where an object might be (object -> list of candidate place facts)
        # We infer places from 'at' facts' second argument.
        self.places: Set[str] = set()

        # Build indices over all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self.facts_by_pred_args[key] = fact

            if pred == 'on' and len(args) == 2:
                c, s = args
                self.on_index[(c, s)] = fact
            elif pred == 'at' and len(args) == 2:
                x, p = args
                self.at_index[(x, p)] = fact
                self.places.add(p)
                self.at_index_by_obj.setdefault(x, []).append((p, fact))
            elif pred == 'in' and len(args) == 2:
                c, t = args
                self.in_index_by_crate.setdefault(c, []).append((t, fact))
            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self.lifting_index_by_crate.setdefault(c, []).append((h, fact))
            elif pred == 'clear' and len(args) == 1:
                s = args[0]
                self.clear_index[s] = fact

        # Precompute goal on-facts as (crate, surface, fact)
        self.goal_on_facts: List[Tuple[str, str, object]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                c, s = args
                f = self.on_index.get((c, s))
                if f is not None:
                    self.goal_on_facts.append((c, s, f))

    # ------------------------------
    # Heuristic interface
    # ------------------------------
    def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
        # Base unique lower bound U: sum over unsatisfied goals of (1 Drop + 1 pickup if not lifting now)
        U = 0
        # Per-goal refined bound (pickup+drop + transport + clearance); we take max over all goals
        per_goal_max = 0

        for c, s2, goal_fact in self.goal_on_facts:
            if self._is_true(state_bitwise, goal_fact):
                continue  # already satisfied

            # Determine if crate is already being lifted by any hoist
            is_lifting_now = self._crate_lifting_now(state_bitwise, c)
            pickup_needed = 0 if is_lifting_now else 1
            drop_needed = 1  # Always needed for placing on target surface

            # Add to unique lower bound
            U += (drop_needed + pickup_needed)

            # Refined per-goal bound additions (beyond pickup+drop already counted locally here):
            transport_extra = self._transport_extra_cost(state_bitwise, c, s2)
            clear_extra = self._clear_extra_cost(state_bitwise, s2)

            Ci = pickup_needed + drop_needed + transport_extra + clear_extra
            if Ci > per_goal_max:
                per_goal_max = Ci

        # Final heuristic: admissible combination
        return max(U, per_goal_max)

    # ------------------------------
    # Helpers
    # ------------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_obj) -> bool:
        if fact_obj is None:
            return False
        return ((state_bitwise >> fact_obj.global_id) & 1) == 1

    def _crate_lifting_now(self, state_bitwise: int, crate: str) -> bool:
        for h, lf in self.lifting_index_by_crate.get(crate, []):
            if self._is_true(state_bitwise, lf):
                return True
        return False

    def _get_object_place(self, state_bitwise: int, obj: str) -> Optional[str]:
        """Return current place p where at[obj, p] holds, or None if unknown in this state."""
        for p, f in self.at_index_by_obj.get(obj, []):
            if self._is_true(state_bitwise, f):
                return p
        return None

    def _get_crate_status_and_place(self, state_bitwise: int, crate: str) -> Tuple[str, Optional[str]]:
        """Return (status, place) for crate in { 'lifting', 'in', 'at', 'unknown' } with the associated place.
        - lifting: place is the place of the lifting hoist
        - in:      place is the place of the truck holding the crate
        - at:      place is the place where the crate is
        - unknown: place None if undetermined
        """
        # Check lifting first
        for h, lf in self.lifting_index_by_crate.get(crate, []):
            if self._is_true(state_bitwise, lf):
                p = self._get_object_place(state_bitwise, h)
                return 'lifting', p
        # Check in truck
        for t, inf in self.in_index_by_crate.get(crate, []):
            if self._is_true(state_bitwise, inf):
                p = self._get_object_place(state_bitwise, t)
                return 'in', p
        # Check at place
        p = self._get_object_place(state_bitwise, crate)
        if p is not None:
            return 'at', p
        return 'unknown', None

    def _transport_extra_cost(self, state_bitwise: int, crate: str, target_surface: str) -> int:
        """Minimal transport extra actions beyond exactly one pickup and one drop for achieving on[crate, target_surface].
        Rules (safe lower bounds):
          - If crate place == target place: 0
          - Else if crate status is 'in' a truck: at least 1 (Drive)
          - Else (crate 'at' or 'lifting'): at least 3 (Load + Drive + Unload)
          - If we cannot determine places, return 0 (to preserve admissibility)
        """
        status, cur_place = self._get_crate_status_and_place(state_bitwise, crate)
        target_place = self._get_object_place(state_bitwise, target_surface)
        if cur_place is None or target_place is None:
            return 0
        if cur_place == target_place:
            return 0
        if status == 'in':
            return 1  # Drive at least once
        # status in {'at', 'lifting', 'unknown-but-not-in'} -> need Load+Drive+Unload minimally
        return 3

    def _clear_extra_cost(self, state_bitwise: int, surface: str) -> int:
        """If target surface is not clear now, at least one Lift is required to clear it before Drop.
        Return 1 if not clear, else 0. If 'clear' fact unknown, return 0.
        """
        f_clear = self.clear_index.get(surface)
        if f_clear is None:
            return 0
        return 0 if self._is_true(state_bitwise, f_clear) else 1
