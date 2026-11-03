from typing import Dict, Tuple, List, Optional, Set
from collections import defaultdict

# Best-effort imports for the base Heuristic class across possible layouts
from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Depots HDDL domain.

    Design goals:
    - Admissible: never overestimates the remaining cost (number of primitive actions).
    - Efficient: uses only bitwise checks and precomputed fact indices (no explicit state reconstruction per node).
    - Domain-aware: approximates the minimum actions needed to achieve each (on crate surface) goal using Depots mechanics.

    Key domain insights used:
    - To put a crate c on a target surface s2, at minimum you must perform a final Drop.
    - If the crate is already lifted at the same place as s2, only Drop is needed (cost 1).
    - If the crate is in a truck at the same place as s2, Unload + Drop are needed (cost 2).
    - If the crate and s2 are at the same place (and the crate is not lifted nor in a truck), Lift + Drop are needed (cost 2).
    - If they are at different places (and not already lifted/in-truck), the transfer will cost at least Drive + (become lifted/unload) + Drop (we use 3 as a safe optimistic lower bound).

    Notes on admissibility:
    - We intentionally ignore additional clear(...) requirements and hoist availability in the estimate. This keeps the
      estimate optimistic (i.e., a lower bound). Counting these could double-count across multiple goals and risk inadmissibility because
      some clear facts become true "for free" as side-effects of earlier necessary actions.
    - We also do not add cost contributions for abstract tasks in the task network beyond the explicit goal facts. In this domain, top-level
      goals are expressed as on(crate, surface) and the HTN task network typically refines toward those goals. Adding task-network costs risks
      double-counting with goal-based estimates.

    Implementation outline:
    - Precompute fast indices for facts: on, at, in, lifting, clear (global_id lookup by grounded arguments).
    - Precompute per-entity mappings to their potential at[entity, place] facts to quickly detect current place via bitwise checks.
    - Extract goal on-pairs from model.goals via model.state_explicit_repr.
    - For each unsatisfied goal on(c, s2), compute a per-goal lower bound using only bitwise membership checks.

    Complexity:
    - Per node: O(|goals| * |places_per_entity|). Places per entity is small (number of depots/distributors), so this scales well even for the largest instance.
    """

    def __init__(self, model, depots_gpt5_heuristic_11: Optional[dict] = None):
        super().__init__(model)
        self.depots_gpt5_heuristic_11 = depots_gpt5_heuristic_11 or {}

        # Fact indices
        self.on_ids: Dict[Tuple[str, str], int] = {}
        self.in_ids: Dict[Tuple[str, str], int] = {}
        self.lifting_ids: Dict[Tuple[str, str], int] = {}
        self.clear_ids: Dict[str, int] = {}

        # entity_at maps a locatable to {place: fact_gid} for at[locatable, place]
        self.entity_at: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Convenience reverse indices
        self.crate_in: Dict[str, Dict[str, int]] = defaultdict(dict)         # crate -> {truck: gid}
        self.crate_lifting: Dict[str, Dict[str, int]] = defaultdict(dict)    # crate -> {hoist: gid}

        # Discovered domain objects
        self.trucks: Set[str] = set()
        self.hoists: Set[str] = set()
        self.crates: Set[str] = set()
        self.surfaces: Set[str] = set()
        self.places: Set[str] = set()

        # Build indices from model facts (bitwise global ids)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                c, s = args
                self.on_ids[(c, s)] = fact.global_id
                self.crates.add(c)
                self.surfaces.add(s)
            elif pred == 'at' and len(args) == 2:
                loc, place = args
                self.entity_at[loc][place] = fact.global_id
                self.places.add(place)
            elif pred == 'in' and len(args) == 2:
                c, t = args
                self.in_ids[(c, t)] = fact.global_id
                self.crate_in[c][t] = fact.global_id
                self.crates.add(c)
                self.trucks.add(t)
            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self.lifting_ids[(h, c)] = fact.global_id
                self.crate_lifting[c][h] = fact.global_id
                self.hoists.add(h)
                self.crates.add(c)
            elif pred == 'clear' and len(args) == 1:
                x = args[0]
                self.clear_ids[x] = fact.global_id
                self.surfaces.add(x)
            # available[h] not needed for this LB

        # Convenience references for truck/hoist 'at' maps (if available). Missing entries return empty dict.
        self.truck_at: Dict[str, Dict[str, int]] = {t: self.entity_at.get(t, {}) for t in self.trucks}
        self.hoist_at: Dict[str, Dict[str, int]] = {h: self.entity_at.get(h, {}) for h in self.hoists}

        # Extract goal on-pairs from the bitwise goal state
        self.goal_on_pairs: List[Tuple[str, str]] = []
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                self.goal_on_pairs.append((args[0], args[1]))

    # --- Utility parsers and checkers ---
    @staticmethod
    def _parse_fact_name(name: str):
        """Parse a grounded fact name in the format +predicate[arg1,arg2]."""
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

    @staticmethod
    def _bit_true(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    def _get_place_of(self, entity: str, state_bitwise: int) -> Optional[str]:
        """Return the unique place where at(entity, place) holds, or None if not located (e.g., in truck or lifted)."""
        at_map = self.entity_at.get(entity)
        if not at_map:
            return None
        # Places are few; iterate and check bitwise membership
        for place, gid in at_map.items():
            if self._bit_true(state_bitwise, gid):
                return place
        return None

    def _crate_lifted_by(self, crate: str, state_bitwise: int) -> Optional[str]:
        """Return the hoist name if lifting(hoist, crate) holds, else None."""
        lifts = self.crate_lifting.get(crate)
        if not lifts:
            return None
        for h, gid in lifts.items():
            if self._bit_true(state_bitwise, gid):
                return h
        return None

    def _crate_in_truck(self, crate: str, state_bitwise: int) -> Optional[str]:
        """Return the truck name if in(crate, truck) holds, else None."""
        in_map = self.crate_in.get(crate)
        if not in_map:
            return None
        for t, gid in in_map.items():
            if self._bit_true(state_bitwise, gid):
                return t
        return None

    # --- Per-goal lower-bound estimator ---
    def _estimate_on_cost(self, crate: str, surface: str, state_bitwise: int) -> int:
        """
        Admissible (optimistic) lower bound on the number of actions needed to achieve on(crate, surface).

        Cases:
        - Already on -> 0
        - Crate lifted at same place as surface -> 1 (Drop)
        - Crate lifted at different place or surface place unknown -> 3 (Load/Drive/Unload/Drop lower-bounded)
        - Crate in truck at same place as surface -> 2 (Unload + Drop)
        - Crate in truck at different place or unknown -> 3 (Drive + Unload + Drop)
        - Crate and surface colocated (and crate not lifted/in-truck) -> 2 (Lift + Drop)
        - Otherwise (different places or unknowns) -> 3
        """
        gid_on = self.on_ids.get((crate, surface))
        if gid_on is not None and self._bit_true(state_bitwise, gid_on):
            return 0

        # If crate is currently lifted by some hoist
        h = self._crate_lifted_by(crate, state_bitwise)
        if h is not None:
            place_surf = self._get_place_of(surface, state_bitwise)
            place_hoist = self._get_place_of(h, state_bitwise)
            if place_surf is not None and place_hoist is not None and place_surf == place_hoist:
                return 1  # Only Drop needed
            else:
                return 3  # Need (at least) Load/Drive/Unload/Drop; LB=3

        # If crate is in some truck
        t = self._crate_in_truck(crate, state_bitwise)
        if t is not None:
            place_truck = self._get_place_of(t, state_bitwise)
            place_surf = self._get_place_of(surface, state_bitwise)
            if place_truck is not None and place_surf is not None and place_truck == place_surf:
                return 2  # Unload + Drop
            else:
                return 3  # Drive + Unload + Drop (LB)

        # Otherwise, use at-locations of crate and surface
        place_crate = self._get_place_of(crate, state_bitwise)
        place_surf = self._get_place_of(surface, state_bitwise)
        if place_crate is not None and place_surf is not None:
            if place_crate == place_surf:
                return 2  # Lift + Drop
            else:
                return 3  # Drive + (Lift/Unload) + Drop (LB)

        # Fallback if places are unknown: assume at least 3 actions required
        return 3

    # --- Heuristic interface ---
    def evaluate(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        """Return the sum of admissible per-goal lower bounds for all unsatisfied on(goals)."""
        h_val = 0
        for c, s in self.goal_on_pairs:
            h_val += self._estimate_on_cost(c, s, state_bitwise)
        return h_val

    # Allow calling the object directly as a function
    __call__ = evaluate


# Alias expected by some factories (optional)
HeuristicClass = DepotsGPT5Heuristic11
