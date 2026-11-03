from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic1(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depots domain.

    Design rationale (admissible lower bound):
    - Goals are On(crate, surface). To achieve on(c, s2) from an arbitrary state, the crate c must finally be Dropped onto s2.
    - The minimal primitive steps depend on where the crate is now and (optimistically) whether source/destination are co-located.
    - We compute a per-goal lower bound and sum them (safe because each goal concerns a different crate; no single primitive can establish two distinct 'on' facts for different crates).

    For each goal on(c, s2):
      0) If already true: cost 0.
      1) If crate is currently being lifted by some hoist:
         - If hoist and s2 are co-located (same place): need at least 1 action (Drop)
         - Else: need at least Load + Drive + Unload + Drop = 4 actions
      2) Else if crate is in a truck t:
         - If truck and s2 co-located: need at least Unload + Drop = 2
         - Else: need at least Drive + Unload + Drop = 3
      3) Else (crate is on some surface at some place):
         - Co-located with s2: at least Lift + Drop = 2
         - Else: at least Lift + Load + Drive + Unload + Drop = 5
         - Plus: you must clear all crates above c before lifting c. We add the number of crates stacked directly above c
           (each requires at least one Lift). This is a necessary cost unique to c, so summing over goals remains admissible.

    Notes on admissibility:
    - We do NOT add extra cost for clearing the destination surface s2 to avoid double counting across different goals;
      those lifts will be required by the goal crates that currently occupy s2 and are counted in their own move cost.
    - We optimistically assume availability of hoists/trucks and at most one required Drive move when transporting (lower bound).
    - If location (place) cannot be inferred for crate/support/s2 from current facts, we assume best-case co-location.

    Efficiency considerations:
    - All fact lookups are done with bitwise checks using precomputed global IDs.
    - We pre-index all facts by predicate and arguments for O(1) access at evaluation.
    - Counting crates stacked above a target uses pre-indexed adjacency (support -> top crate) and bitwise presence.

    """

    def __init__(self, model, depots_gpt5_heuristic_1: bool = True):
        super().__init__(model, name="DepotsGPT5Heuristic1")

        # Indices for fast access
        self.on_gid_by_pair: Dict[Tuple[str, str], int] = {}
        self.on_support_to_crates: Dict[str, List[Tuple[str, int]]] = {}
        self.on_crate_to_supports: Dict[str, List[Tuple[str, int]]] = {}

        self.in_gid_by_pair: Dict[Tuple[str, str], int] = {}
        self.in_by_crate: Dict[str, List[Tuple[str, int]]] = {}

        self.lifting_gid_by_pair: Dict[Tuple[str, str], int] = {}
        self.lifting_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        self.lifting_by_hoist: Dict[str, List[Tuple[str, int]]] = {}

        self.clear_gid_by_surface: Dict[str, int] = {}

        self.at_gid_by_pair: Dict[Tuple[str, str], int] = {}
        self.at_by_loc: Dict[str, List[Tuple[str, int]]] = {}
        self.at_by_place: Dict[str, List[Tuple[str, int]]] = {}

        # Object sets discovered from facts (helpful for type inference and loops)
        self.crates: Set[str] = set()
        self.surfaces: Set[str] = set()
        self.hoists: Set[str] = set()
        self.trucks: Set[str] = set()
        self.places: Set[str] = set()

        # Parse facts to build indices
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            gid = fact.global_id

            if pred == "on" and len(args) == 2:
                c, s = args
                self.on_gid_by_pair[(c, s)] = gid
                self.on_support_to_crates.setdefault(s, []).append((c, gid))
                self.on_crate_to_supports.setdefault(c, []).append((s, gid))
                # Type discovery
                self.crates.add(c)
                self.surfaces.add(s)

            elif pred == "in" and len(args) == 2:
                c, t = args
                self.in_gid_by_pair[(c, t)] = gid
                self.in_by_crate.setdefault(c, []).append((t, gid))
                self.crates.add(c)
                self.trucks.add(t)

            elif pred == "lifting" and len(args) == 2:
                h, c = args
                self.lifting_gid_by_pair[(h, c)] = gid
                self.lifting_by_crate.setdefault(c, []).append((h, gid))
                self.lifting_by_hoist.setdefault(h, []).append((c, gid))
                self.hoists.add(h)
                self.crates.add(c)

            elif pred == "clear" and len(args) == 1:
                s = args[0]
                self.clear_gid_by_surface[s] = gid
                self.surfaces.add(s)

            elif pred == "at" and len(args) == 2:
                loc, p = args
                self.at_gid_by_pair[(loc, p)] = gid
                self.at_by_loc.setdefault(loc, []).append((p, gid))
                self.at_by_place.setdefault(p, []).append((loc, gid))
                # Type discovery (best-effort):
                self.places.add(p)
                # loc may be hoist/truck/crate/pallet/surface
                if any(loc.startswith(pref) for pref in ("truck",)):
                    self.trucks.add(loc)
                if any(loc.startswith(pref) for pref in ("hoist",)):
                    self.hoists.add(loc)
                # We cannot fully infer crates/pallets here reliably; covered above via other preds

        # Parse goal facts once (bitset -> explicit names -> parsed pairs)
        self.goal_on_facts: List[Tuple[str, str, int]] = []  # (crate, surface, gid)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                c, s = args
                gid = self.on_gid_by_pair.get((c, s))
                if gid is not None:
                    self.goal_on_facts.append((c, s, gid))

    def evaluate(self, state_bitwise: int, task_network: List) -> int:
        """
        Compute the heuristic value for the given state and current (possibly partial) task network.
        We mainly use the goal On-facts and current state to sum up per-goal optimistic costs as outlined above.
        """
        h_total = 0
        if not self.goal_on_facts:
            return 0

        for c, s2, gid in self.goal_on_facts:
            # If goal already satisfied, cost 0 for this pair
            if self._has_fact(state_bitwise, gid):
                continue

            # Estimate minimal remaining steps to achieve on[c, s2]
            cost = self._estimate_on_cost(state_bitwise, c, s2)
            h_total += cost

        return h_total

    # ---------------------- Internal helpers ----------------------

    def _estimate_on_cost(self, state: int, c: str, s2: str) -> int:
        """
        Per-goal admissible estimate for achieving on(c, s2) from the current state.
        Cases:
          - c is lifted: 1 if co-located else 4
          - c is in truck: 2 if co-located else 3
          - c is on some surface (or otherwise at a place): base 2 if co-located else 5, plus number of crates above c
        """
        # 1) If crate currently lifted by any hoist
        lifted_info = self.lifting_by_crate.get(c, [])
        for h, gid in lifted_info:
            if self._has_fact(state, gid):
                # If we know co-location (hoist and s2 at same place) -> 1 (Drop), else 4 (Load+Drive+Unload+Drop)
                if self._co_located_lifted(state, h, s2):
                    return 1
                else:
                    return 4

        # 2) If crate currently in some truck
        in_info = self.in_by_crate.get(c, [])
        for t, gid in in_info:
            if self._has_fact(state, gid):
                if self._co_located_truck_surface(state, t, s2):
                    return 2  # Unload + Drop
                else:
                    return 3  # Drive + Unload + Drop

        # 3) Otherwise, assume crate is on some surface stack at some place
        #    Base cost: 2 if co-located, else 5, plus lifts to clear any crates above c
        above_c = self._count_above_crate(state, c)
        coloc = self._co_located_crate_surface(state, c, s2)
        base = 2 if coloc else 5
        return base + above_c

    def _count_above_crate(self, state: int, c: str) -> int:
        """
        Count how many crates are stacked directly above crate c in the current state.
        We follow on[x, current] upwards while true.
        """
        count = 0
        current_surface = c
        visited = set()
        while True:
            if current_surface in visited:
                break  # safety guard
            visited.add(current_surface)
            top_list = self.on_support_to_crates.get(current_surface, [])
            next_above: Optional[str] = None
            for above_c, on_gid in top_list:
                if self._has_fact(state, on_gid):
                    next_above = above_c
                    break
            if next_above is None:
                break
            count += 1
            current_surface = next_above
        return count

    def _co_located_lifted(self, state: int, h: str, s2: str) -> bool:
        """
        Check if a lifted crate by hoist h is co-located with surface s2 (same place).
        We look for at[h, p] and at[s2, p] for some p.
        If we cannot determine, return True (optimistic lower bound).
        """
        place_h = self._place_of_loc(state, h)
        place_s2 = self._place_of_surface(state, s2)
        if place_h is None or place_s2 is None:
            return True  # optimistic
        return place_h == place_s2

    def _co_located_truck_surface(self, state: int, t: str, s2: str) -> bool:
        place_t = self._place_of_loc(state, t)
        place_s2 = self._place_of_surface(state, s2)
        if place_t is None or place_s2 is None:
            return True  # optimistic
        return place_t == place_s2

    def _co_located_crate_surface(self, state: int, c: str, s2: str) -> bool:
        """
        Check if crate c (presumed on some stack) is co-located with surface s2.
        We derive place(c) from the surface chain it rests on; place(s2) similarly.
        If any is unknown, assume True (optimistic lower bound).
        """
        place_c = self._place_of_crate(state, c)
        place_s2 = self._place_of_surface(state, s2)
        if place_c is None or place_s2 is None:
            return True
        return place_c == place_s2

    def _place_of_crate(self, state: int, c: str) -> Optional[str]:
        """
        Derive the place of crate c:
          - If at[c, p] known true, return p.
          - Else, find its support surface s where on[c, s] holds, then return place(s) recursively.
          - If none resolvable, return None.
        """
        # Direct at[c, p]
        if c in self.at_by_loc:
            for p, gid in self.at_by_loc[c]:
                if self._has_fact(state, gid):
                    return p
        # On[c, s] -> place(s)
        for s, on_gid in self.on_crate_to_supports.get(c, []):
            if self._has_fact(state, on_gid):
                return self._place_of_surface(state, s)
        return None

    def _place_of_surface(self, state: int, s: str) -> Optional[str]:
        """
        Derive place of a surface s (pallet or crate surface):
          - If at[s, p] known true, return p
          - If s is a crate and on[s, s_below] holds, recurse down until a surface with at[., p]
          - Otherwise None
        """
        # Direct at[s, p]
        if s in self.at_by_loc:
            for p, gid in self.at_by_loc[s]:
                if self._has_fact(state, gid):
                    return p
        # If s is a crate, it might sit on something else
        if s in self.on_crate_to_supports:
            for s_below, on_gid in self.on_crate_to_supports[s]:
                if self._has_fact(state, on_gid):
                    return self._place_of_surface(state, s_below)
        return None

    def _place_of_loc(self, state: int, loc: str) -> Optional[str]:
        """
        Return place p for locatable 'loc' if any at[loc, p] fact is true in current state.
        """
        if loc in self.at_by_loc:
            for p, gid in self.at_by_loc[loc]:
                if self._has_fact(state, gid):
                    return p
        return None

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names in format +predicate[arg1,arg2] or -predicate[arg1,arg2]
        Returns (predicate, [args...]) or (None, []) if unparsable.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _has_fact(state_bitwise: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return (state_bitwise >> gid) & 1 == 1


# Backwards-compatible alias for dynamic loading if needed
def depots_gpt5_heuristic_1(model):
    return DepotsGPT5Heuristic1(model)
