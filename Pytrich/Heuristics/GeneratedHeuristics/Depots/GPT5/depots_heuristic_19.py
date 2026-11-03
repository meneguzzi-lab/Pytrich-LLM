from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Depot (HDDL) domain.

    Key domain insights leveraged (lower-bound reasoning):
    - The only primitive action that makes on(c, s) true is Drop. Therefore, for every goal on(c, s)
      that is not yet satisfied, at least one Drop is required.
    - To apply Drop, the crate must be in a lifting state (lifting(h, c)). Achieving lifting for c
      requires at least one prior action (either Lift or Unload) unless it is already lifting.
      Hence, per unsatisfied goal on(c, s):
        * If lifting(?, c) already holds: lower bound is at least 1 (just Drop).
        * Otherwise: lower bound is at least 2 (one to create lifting, one Drop).
    - Inter-place movement: If a crate and its target surface are at different places (and the place
      of the surface is known), then at least one Drive must eventually occur in the overall plan
      (no action other than Drive changes a truck's place, and crates cannot teleport across places).
      However, because a single Drive can serve multiple crates (e.g., packing several crates in one
      truck before driving), we add at most a single global +1 to the heuristic value if any such
      cross-place requirement exists in the current state. This preserves admissibility while adding
      guidance.

    Admissibility:
    - The estimate per unsatisfied on-goal is 1 (if already lifting) or 2 (otherwise), which is a
      valid lower bound because achieving on requires a Drop and (unless already lifting) at least
      one more action to establish lifting.
    - The global +1 for Drive is added only once if ANY remaining goal demonstrably requires an
      inter-place move, since that is a minimum across all goals combined.

    Efficiency considerations:
    - All fact lookups are implemented with bit-index checks over pre-indexed fact IDs (no string set
      membership over state_explicit_repr during search), which is critical for large instances.

    Implementation details:
    - We parse and index facts for the relevant predicates: on, at, in, lifting, clear (clear is
      parsed but not required for this lower bound). This allows constant-time checks per fact ID
      using bitwise operations.
    - Goals are assumed to be on-facts in this domain; we extract and store them once.

    Expected interface:
    - __call__(state_bitwise: int, task_network: List[...]) -> int
      Returns the heuristic value h(state, task_network). The task network is unused in the lower bound
      (we remain conservative); using only the state and the domain structure already provides strong,
      scalable guidance for Depots.
    """

    def __init__(self, model, depots_gpt5_heuristic_19: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.depots_gpt5_heuristic_19_cfg = depots_gpt5_heuristic_19 or {}

        # Fast name->Fact lookup
        self._fact_by_name: Dict[str, object] = {}

        # Indexed fact IDs by predicate/arguments
        # on(c, s) -> fact_id
        self._on_id: Dict[Tuple[str, str], int] = {}
        # at(x, p): x can be crate, pallet, truck, hoist (locatable)
        # store as obj -> List[(place, fact_id)] for quick scan
        self._at_ids_by_obj: Dict[str, List[Tuple[str, int]] ] = {}
        # in(c, t): crate in truck
        self._in_ids_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        # lifting(h, c): store by crate -> List[(hoist, fact_id)]
        self._lifting_ids_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        # clear(x) -> fact_id (parsed but not used in LB)
        self._clear_id: Dict[str, int] = {}

        # Build indices from all facts
        for fact in self.model.facts:
            self._fact_by_name[fact.name] = fact
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                self._on_id[(args[0], args[1])] = fact.global_id
            elif pred == 'at' and len(args) == 2:
                obj, place = args[0], args[1]
                self._at_ids_by_obj.setdefault(obj, []).append((place, fact.global_id))
            elif pred == 'in' and len(args) == 2:
                c, t = args[0], args[1]
                self._in_ids_by_crate.setdefault(c, []).append((t, fact.global_id))
            elif pred == 'lifting' and len(args) == 2:
                h, c = args[0], args[1]
                self._lifting_ids_by_crate.setdefault(c, []).append((h, fact.global_id))
            elif pred == 'clear' and len(args) == 1:
                self._clear_id[args[0]] = fact.global_id

        # Parse and store goal on-facts (as (crate, surface, fact_id or None if not found))
        self._goal_ons: List[Tuple[str, str, Optional[int]]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                fact_id = self._on_id.get((c, s))
                self._goal_ons.append((c, s, fact_id))

    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.h(state_bitwise, task_network)

    def h(self, state_bitwise: int, task_network: List = None) -> int:
        """
        Heuristic estimate (admissible lower bound):
        - Sum over goals on(c, s): 1 if lifting(?, c) is true, else 2 if on(c, s) not yet true.
        - Additionally, add a single global +1 if there exists at least one unsatisfied goal
          demonstrably requiring a cross-place move (crate-place != target-place), given current state.
        """
        if state_bitwise is None:
            return 0

        base_sum = 0
        need_drive = False  # global (at most one) drive LB across all remaining goals

        for c, s, on_fid in self._goal_ons:
            # If this specific on-goal is already satisfied, skip
            if on_fid is not None and ((state_bitwise >> on_fid) & 1):
                continue

            # Determine if crate c is already being lifted
            lifting_info = self._get_lifting_hoist(state_bitwise, c)
            if lifting_info is not None:
                # Already lifting: at least one Drop needed
                base_sum += 1
                # If places are mismatched (when surface place is known), then at least one Drive will be required overall
                p_s = self._get_place_of(state_bitwise, s)
                if p_s is not None:
                    h = lifting_info
                    p_l = self._get_place_of(state_bitwise, h)
                    if p_l is not None and p_l != p_s:
                        need_drive = True
                continue

            # Not lifting: at least one action to establish lifting (Lift or Unload) + one Drop
            base_sum += 2

            # Check if we can soundly conclude at least one Drive is needed globally
            p_s = self._get_place_of(state_bitwise, s)
            if p_s is None:
                # If surface place is unknown (e.g., surface currently lifted or in truck),
                # we cannot prove a cross-place requirement now.
                continue

            # If crate is in a truck, compare truck place vs target place
            t = self._get_truck_of(state_bitwise, c)
            if t is not None:
                p_t = self._get_place_of(state_bitwise, t)
                if p_t is not None and p_t != p_s:
                    need_drive = True
                continue

            # Otherwise, if crate has a place, compare it to the surface place
            p_c = self._get_place_of(state_bitwise, c)
            if p_c is not None and p_c != p_s:
                need_drive = True

        # Single global drive LB at most +1
        return base_sum + (1 if need_drive else 0)

    # -----------------------------
    # Internal helpers
    # -----------------------------

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names as stored in the model:
        Expected formats: '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Returns (predicate, [args...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            core = name[1:]
            bracket_idx = core.find("[")
            if bracket_idx > 0 and core.endswith("]"):
                predicate = core[:bracket_idx]
                args_str = core[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _get_place_of(self, state_bitwise: int, obj: str) -> Optional[str]:
        """Return the place of a locatable object (crate, pallet, truck, hoist) in the current state, if known."""
        for place, fid in self._at_ids_by_obj.get(obj, []):
            if (state_bitwise >> fid) & 1:
                return place
        return None

    def _get_truck_of(self, state_bitwise: int, crate: str) -> Optional[str]:
        """Return the truck that currently contains crate, if any."""
        for t, fid in self._in_ids_by_crate.get(crate, []):
            if (state_bitwise >> fid) & 1:
                return t
        return None

    def _get_lifting_hoist(self, state_bitwise: int, crate: str) -> Optional[str]:
        """Return the hoist that is currently lifting crate, if any."""
        for h, fid in self._lifting_ids_by_crate.get(crate, []):
            if (state_bitwise >> fid) & 1:
                return h
        return None


# Optional convenient alias used by the framework in some configurations
HeuristicClass = DepotsGPT5Heuristic19
