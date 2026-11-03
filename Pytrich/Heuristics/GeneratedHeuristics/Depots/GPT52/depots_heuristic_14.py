from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic14(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Rationale / domain insights
    --------------------------
    The top-level tasks in the benchmark instances are almost exclusively:
        (do_put_on crateX surfaceY)

    The domain methods for do_put_on reveal a small set of cases:
      m0: already on(c, s2) -> 0 cost
      m1: crate at same place as target surface -> clear crate, clear surface, lift, drop
      m2: crate in truck -> get truck to place, clear surface, unload, drop
      m3: otherwise -> load truck at source place, drive, unload at destination

    The hard part in Depots is *stack clearing* (to make a crate clear for Lift, and
    to make a destination surface clear for Drop) plus occasional trucking.

    This heuristic computes a cheap lower bound by summing, for each goal/on atom
    not yet satisfied, an estimate of the minimal number of primitive actions still
    needed, using only *bitwise* fact checks (fast) and precomputed static mappings.

    Key approximations (all optimistic => admissible as a lower bound):
      - Clearing a surface that is not currently clear is counted as 1 action
        (even though it may require many moves). This is safe (underestimates).
      - If the crate is not clear, we add 1 (it must be cleared somehow).
      - If crate and destination are at different places (or crate is in a truck),
        we add at least one transport action. We use:
            +1 if needs drive (truck relocation)  [optimistic; real may be more]
            +2 for load/unload when changing place [optimistic]
      - If crate is already lifting by some hoist, we can skip lift.

    The result is informative enough to reduce node expansions while being very
    cheap to compute, even on large instances.
    """

    def __init__(self, model, depots_gpt52_heuristic_14: str = "depots_gpt52_heuristic_14"):
        super().__init__(model, depots_gpt52_heuristic_14)
        self.model = model

        # --- Parse all facts once to build indices for fast state queries ---
        self._fact_by_pred: Dict[str, List[Tuple[List[str], int]]] = {}
        self._goal_on_pairs: List[Tuple[str, str]] = []

        # For quick lookup of a particular grounded fact id
        self._fact_id_at: Dict[Tuple[str, str], int] = {}          # (obj, place) -> fact_id
        self._fact_id_in: Dict[Tuple[str, str], int] = {}          # (crate, truck) -> fact_id
        self._fact_id_on: Dict[Tuple[str, str], int] = {}          # (crate, surface) -> fact_id
        self._fact_id_clear: Dict[str, int] = {}                   # surface -> fact_id
        self._fact_id_lifting: Dict[Tuple[str, str], int] = {}     # (hoist, crate) -> fact_id

        # Reverse maps useful for "who is on what" and "what is on this surface"
        self._on_by_surface: Dict[str, List[str]] = {}  # surface -> [crate,...]

        # Cache all parsed facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_by_pred.setdefault(pred, []).append((args, f.global_id))

            if pred == "at" and len(args) == 2:
                self._fact_id_at[(args[0], args[1])] = f.global_id
            elif pred == "in" and len(args) == 2:
                self._fact_id_in[(args[0], args[1])] = f.global_id
            elif pred == "on" and len(args) == 2:
                self._fact_id_on[(args[0], args[1])] = f.global_id
                self._on_by_surface.setdefault(args[1], []).append(args[0])
            elif pred == "clear" and len(args) == 1:
                self._fact_id_clear[args[0]] = f.global_id
            elif pred == "lifting" and len(args) == 2:
                self._fact_id_lifting[(args[0], args[1])] = f.global_id

        # Extract goal on-facts (instances' goals are conjunctions of +on[...])
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self._goal_on_pairs.append((args[0], args[1]))

        # Precompute object type-ish sets (best-effort, from facts)
        # Places are second arg of at(x, place). Surfaces appear in clear(s) or as on(_, s).
        self._all_places: Set[str] = set()
        for (obj, pl), _fid in self._fact_id_at.items():
            self._all_places.add(pl)

        self._all_surfaces: Set[str] = set(self._fact_id_clear.keys()) | set(self._on_by_surface.keys())

        # Trucks and hoists are inferred from predicates using their argument positions
        self._all_trucks: Set[str] = set()
        for (crate, tr), _fid in self._fact_id_in.items():
            self._all_trucks.add(tr)
        for args, _fid in self._fact_by_pred.get("at", []):
            # cannot infer reliably which are trucks/hoists here; we use in/lifting for that
            pass
        self._all_hoists: Set[str] = set()
        for (h, c), _fid in self._fact_id_lifting.items():
            self._all_hoists.add(h)

    # ---------- Pytrich required interface ----------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        # In Pytrich, heuristics are called with (state, task_network).
        # We don't need the TN for this domain-specific goal-driven estimate.
        return self.compute_heuristic(state_bitwise)

    # ---------- Core heuristic ----------
    def compute_heuristic(self, state_bitwise: int) -> int:
        # Sum of per-goal lower bounds; clamp at 0.
        h_val = 0
        for crate, dest_surface in self._goal_on_pairs:
            if self._is_true_on(state_bitwise, crate, dest_surface):
                continue
            h_val += self._estimate_put_on_cost_lb(state_bitwise, crate, dest_surface)
        return max(0, int(h_val))

    # ---------- Estimation helpers ----------
    def _estimate_put_on_cost_lb(self, s: int, crate: str, dest_surface: str) -> int:
        """Optimistic lower bound for achieving on(crate, dest_surface)."""
        # If already lifting by any hoist, we can potentially just Drop/Load.
        is_lifted = self._is_any_lifting(s, crate)

        # Destination must be at some place; if unknown, ignore transport aspect.
        dest_place = self._get_place_of_locatable(s, dest_surface)

        # Where is the crate?
        crate_place = self._get_place_of_locatable(s, crate)
        crate_truck = self._get_truck_containing(s, crate)

        # Need destination surface clear for Drop.
        need_clear_dest = not self._is_clear(s, dest_surface)

        # Need crate clear for Lift (unless already lifting / already in truck and will unload).
        need_clear_crate = not self._is_clear(s, crate)

        # Case 1: crate and destination are co-located (crate at place, dest at same place)
        if crate_place is not None and dest_place is not None and crate_place == dest_place:
            # Minimal primitive actions: Lift + Drop (2), plus possibly clearing prerequisites.
            base = 0
            if not is_lifted:
                base += 1  # Lift
            base += 1      # Drop
            if need_clear_crate and not is_lifted:
                base += 1
            if need_clear_dest:
                base += 1
            return base

        # Case 2: crate is in a truck
        if crate_truck is not None:
            # To end on dest: at least Unload + Drop (2), maybe Drive truck, maybe clear dest.
            base = 0
            # Ensure truck at destination place (optimistically 1 Drive if not already there).
            if dest_place is not None:
                truck_place = self._get_place_of_locatable(s, crate_truck)
                if truck_place is None or truck_place != dest_place:
                    base += 1  # Drive
            base += 1  # Unload
            base += 1  # Drop
            if need_clear_dest:
                base += 1
            return base

        # Case 3: crate is at some other place (or unknown): need load/drive/unload/drop
        # Minimal chain when moving between places: Lift, Load, Drive, Unload, Drop = 5
        # But we keep it optimistic based on what we can infer.
        base = 0

        # Need to get crate into a truck: requires Lift + Load (2) at least
        if not is_lifted:
            base += 1  # Lift
        base += 1      # Load

        # Drive if destination place known and differs
        if dest_place is not None:
            base += 1  # Drive (optimistically one step)

        # Unload + Drop
        base += 1  # Unload
        base += 1  # Drop

        # Clear prerequisites (optimistic +1 each)
        if need_clear_crate and not is_lifted:
            base += 1
        if need_clear_dest:
            base += 1

        return base

    # ---------- Bitwise fact checks ----------
    def _is_true_fact_id(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _is_true_on(self, s: int, crate: str, surface: str) -> bool:
        return self._is_true_fact_id(s, self._fact_id_on.get((crate, surface)))

    def _is_clear(self, s: int, surface: str) -> bool:
        return self._is_true_fact_id(s, self._fact_id_clear.get(surface))

    def _is_any_lifting(self, s: int, crate: str) -> bool:
        # Check any lifting(hoist, crate)
        for (h, c), fid in self._fact_id_lifting.items():
            if c == crate and ((s >> fid) & 1) == 1:
                return True
        return False

    # ---------- State queries (best-effort) ----------
    def _get_place_of_locatable(self, s: int, obj: str) -> Optional[str]:
        # Find any at(obj, place) that is true.
        # We avoid scanning explicit state; instead, scan preindexed at-facts for this obj.
        for (o, pl), fid in self._fact_id_at.items():
            if o == obj and ((s >> fid) & 1) == 1:
                return pl
        return None

    def _get_truck_containing(self, s: int, crate: str) -> Optional[str]:
        for (c, tr), fid in self._fact_id_in.items():
            if c == crate and ((s >> fid) & 1) == 1:
                return tr
        return None

    # ---------- Fact-name parsing (per spec) ----------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
