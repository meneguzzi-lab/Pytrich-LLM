
# Domain-Dependent Heuristic for HDDL Depots domain
# Class name: DepotsGPT5Heuristic6
# This heuristic leverages domain knowledge of Depots to compute a fast, informative
# lower bound on the number of primitive actions still needed to achieve remaining goals.
#
# Key ideas:
# - Goals are all of the form on(crate, surface). We sum a per-goal lower bound.
# - We inspect the current state to determine where the crate is (at place, in a truck, or lifted),
#   where the target surface is (at a place, in a truck, or lifted), and if the surface is clear.
# - We estimate minimal actions required:
#   * Same place: Lift + Drop = 2 (plus 1 if the target surface isn't clear; plus 1 if crate isn't clear).
#   * Different places: Lift + Load + [Drive] + Unload + Drop = 4 + (0/1 drive)
#     (plus 1 if target surface isn't clear; plus availability checks for hoists at origin/destination).
#   * Crate already in truck at destination place: Unload + Drop = 2 (plus clear/availability costs).
#   * Crate currently being lifted: at same place -> Drop = 1; otherwise Load + [Drive] + Unload + Drop = 3 + (0/1 drive).
# - Hoist availability: If we need a Lift at origin or an Unload at destination and no hoist at
#   that place is available, we add +1 action to make a hoist available (lower bound).
# - Clearing the target surface: If clear(surface) is false, we add +1 (at least one Lift is required
#   to free the surface). We do not further charge for disposing of the lifted blocking crate, ensuring
#   admissibility.
# - Anchoring the target surface: If the surface is in a truck or being lifted, we add +1 to anchor
#   it at some place (Unload or Drop) before we can Drop onto it.
#
# All counts are conservative (lower bounds), ignoring additional ordering/resource constraints, thus
# remaining admissible. The heuristic uses only fast bitwise membership checks and pre-indexed facts.

from typing import Dict, Tuple, List, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depots domain.

    Usage:
    - Instantiate with the model and optional flag depots_gpt5_heuristic_6 (kept for compatibility/toggling).
    - Call h(state_bitwise, task_network) to get the heuristic value.

    Design rationale:
    - The domain core is placing crates onto target surfaces using hoists and (optionally) trucks.
    - We compute a per-goal lower bound by reasoning about where the crate and target surface are,
      whether the target surface is clear, whether trucks must drive, and whether hoists at the relevant
      places are currently available.
    - All contributions are minimal counts of necessary primitive actions, summed across goals.
    """

    def __init__(self, model, depots_gpt5_heuristic_6: bool = True):
        super().__init__(model)
        self.enabled = depots_gpt5_heuristic_6

        # Pre-index model facts for fast lookup
        self.fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self.at_index: Dict[str, List[object]] = {}
        self.in_index: Dict[str, List[object]] = {}
        self.lifting_index_by_crate: Dict[str, List[object]] = {}
        self.lifting_index_by_hoist: Dict[str, List[object]] = {}
        self.clear_index: Dict[str, object] = {}
        self.available_index: Dict[str, object] = {}
        self.on_index: Dict[Tuple[str, str], object] = {}
        self.at_truck_index: Dict[str, List[object]] = {}

        self.hoists: Set[str] = set()
        self.trucks: Set[str] = set()
        self.places: Set[str] = set()

        # Parse all facts and build indices
        self._build_fact_indices()

        # Extract goals (list of (crate, surface) from +on[c,s])
        self.goal_pairs: List[Tuple[str, str]] = self._extract_goal_pairs()

    # ------------------------ Framework helpers ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
        else:
            name_ = name
        bracket_idx = name_.find("[")
        if bracket_idx > 0 and name_.endswith("]"):
            predicate = name_[:bracket_idx]
            args_str = name_[bracket_idx + 1 : -1]
            args = args_str.split(",") if args_str else []
            return predicate, args
        return None, []

    def _get_fact(self, pred: str, args: Tuple[str, ...]):
        return self.fact_by_key.get((pred, tuple(args)))

    def _is_true(self, state_bitwise: int, fact_obj) -> bool:
        if fact_obj is None:
            return False
        try:
            gid = fact_obj.global_id
        except Exception:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    # ------------------------ Index building ------------------------
    def _build_fact_indices(self):
        # Facts are accessible by global id via model.facts
        # We only index positive facts (names may begin with '+')
        for f in self.model.facts:
            name = getattr(f, "name", "")
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            if not name.startswith("+"):
                # Only use positive atoms for indexing
                continue
            key = (pred, tuple(args))
            self.fact_by_key[key] = f

            if pred == "at" and len(args) == 2:
                obj, place = args
                self.places.add(place)
                self.at_index.setdefault(obj, []).append(f)
                # Trucks we detect by accumulating from in[] targets or by name heuristic
                if obj.startswith("truck"):
                    self.trucks.add(obj)
                    self.at_truck_index.setdefault(obj, []).append(f)

            elif pred == "in" and len(args) == 2:
                crate, truck = args
                self.in_index.setdefault(crate, []).append(f)
                if truck.startswith("truck"):
                    self.trucks.add(truck)

            elif pred == "lifting" and len(args) == 2:
                hoist, crate = args
                self.lifting_index_by_crate.setdefault(crate, []).append(f)
                self.lifting_index_by_hoist.setdefault(hoist, []).append(f)

            elif pred == "clear" and len(args) == 1:
                surf = args[0]
                self.clear_index[surf] = f

            elif pred == "available" and len(args) == 1:
                hoist = args[0]
                self.available_index[hoist] = f
                self.hoists.add(hoist)

            elif pred == "on" and len(args) == 2:
                crate, surf = args
                self.on_index[(crate, surf)] = f

    def _extract_goal_pairs(self) -> List[Tuple[str, str]]:
        goal_pairs: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                goal_pairs.append((args[0], args[1]))
        return goal_pairs

    # ------------------------ State query helpers ------------------------
    def _find_place_of_obj(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Returns place where +at[obj, place] holds
        for f in self.at_index.get(obj, []):
            if self._is_true(state_bitwise, f):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    return args[1]
        return None

    def _find_truck_of_crate(self, state_bitwise: int, crate: str) -> Optional[str]:
        for f in self.in_index.get(crate, []):
            if self._is_true(state_bitwise, f):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    return args[1]
        return None

    def _find_lifting_hoist_of_item(self, state_bitwise: int, item: str) -> Optional[str]:
        # Item can be a crate (commonly) or a surface (crate). If lifting[h,item] true -> return h
        for f in self.lifting_index_by_crate.get(item, []):
            if self._is_true(state_bitwise, f):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    return args[0]
        return None

    def _any_available_hoist_at(self, state_bitwise: int, place: Optional[str]) -> bool:
        if place is None:
            return False
        # Any hoist h with +at[h,place] and +available[h]
        for h, fav in self.available_index.items():
            if not self._is_true(state_bitwise, fav):
                continue
            # Check if at[h, place]
            for f_at in self.at_index.get(h, []):
                if self._is_true(state_bitwise, f_at):
                    _, args = self._parse_fact_name(f_at.name)
                    if len(args) == 2 and args[1] == place:
                        return True
        return False

    def _place_of_truck(self, state_bitwise: int, truck: str) -> Optional[str]:
        for f in self.at_truck_index.get(truck, []):
            if self._is_true(state_bitwise, f):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    return args[1]
        return None

    # ------------------------ Per-goal cost estimation ------------------------
    def _goal_lower_bound(self, state_bitwise: int, crate: str, surface: str) -> int:
        # If already satisfied
        f_on = self._get_fact("on", (crate, surface))
        if self._is_true(state_bitwise, f_on):
            return 0

        cost = 0

        # Determine surface anchoring/place
        p_s2: Optional[str] = None
        anchor_cost = 0

        # Case 1: surface is at some place already
        p_s2 = self._find_place_of_obj(state_bitwise, surface)

        if p_s2 is None:
            # Case 2: surface in a truck -> need to anchor (Unload) at least once
            t_s2 = None
            # Try in[surface, truck]
            for f_in_list in [self.in_index.get(surface, [])]:
                for f in f_in_list:
                    if self._is_true(state_bitwise, f):
                        _, args = self._parse_fact_name(f.name)
                        if len(args) == 2:
                            t_s2 = args[1]
                            break
                if t_s2:
                    break
            if t_s2 is not None:
                anchor_cost += 1  # Unload surface to some place
                p_s2 = self._place_of_truck(state_bitwise, t_s2)

        if p_s2 is None:
            # Case 3: surface is being lifted by some hoist -> need to Drop it somewhere (anchor)
            h_s2 = self._find_lifting_hoist_of_item(state_bitwise, surface)
            if h_s2 is not None:
                anchor_cost += 1  # Drop surface at hoist's place
                p_s2 = self._find_place_of_obj(state_bitwise, h_s2)

        # Clearing the target surface if needed (one Lift suffices as LB)
        f_clear_s2 = self.clear_index.get(surface)
        if f_clear_s2 is not None and (not self._is_true(state_bitwise, f_clear_s2)):
            cost += 1

        # Add anchoring cost (if surface was in truck or being lifted)
        cost += anchor_cost

        # Determine crate status and place
        # Check if crate is being lifted
        h_c = self._find_lifting_hoist_of_item(state_bitwise, crate)
        if h_c is not None:
            p_c = self._find_place_of_obj(state_bitwise, h_c)
            # If same place as surface: only Drop needed
            if p_s2 is not None and p_c == p_s2:
                # Just Drop
                cost += 1
                return cost
            else:
                # Need to Load + [Drive] + Unload + Drop
                drive = 1 if p_s2 is None or (p_c is not None and p_s2 is not None and p_c != p_s2) else 0
                # Need available hoist at destination for Unload
                need_av_dest = 0 if self._any_available_hoist_at(state_bitwise, p_s2) else 1
                cost += 3 + drive + need_av_dest
                return cost

        # Check if crate is in a truck
        t_c = self._find_truck_of_crate(state_bitwise, crate)
        if t_c is not None:
            p_t = self._place_of_truck(state_bitwise, t_c)
            drive = 1 if (p_s2 is not None and p_t is not None and p_t != p_s2) else 0
            # Need available hoist at destination for Unload
            need_av_dest = 0 if self._any_available_hoist_at(state_bitwise, p_s2) else 1
            # Unload + Drop
            cost += 2 + drive + need_av_dest
            return cost

        # Else, crate should be at some place
        p_c = self._find_place_of_obj(state_bitwise, crate)
        if p_c is None:
            # Fallback: unknown; assume at least Lift+Drop = 2
            cost += 2
            return cost

        if p_s2 is not None and p_c == p_s2:
            # Same place: Lift + Drop = 2, but need clear(crate) for Lift and available hoist at place
            f_clear_c = self.clear_index.get(crate)
            need_clear_c = 0 if (f_clear_c is not None and self._is_true(state_bitwise, f_clear_c)) else 1
            need_av = 0 if self._any_available_hoist_at(state_bitwise, p_c) else 1
            cost += 2 + need_clear_c + need_av
            return cost
        else:
            # Different places: Lift + Load + [Drive] + Unload + Drop = 4 + drive
            drive = 1  # at least one drive needed in cross-place transfer (LB)
            f_clear_c = self.clear_index.get(crate)
            need_clear_c = 0 if (f_clear_c is not None and self._is_true(state_bitwise, f_clear_c)) else 1
            need_av_origin = 0 if self._any_available_hoist_at(state_bitwise, p_c) else 1
            need_av_dest = 0 if self._any_available_hoist_at(state_bitwise, p_s2) else 1
            cost += 4 + drive + need_clear_c + need_av_origin + need_av_dest
            return cost

    # ------------------------ Heuristic interface ------------------------
    def h(self, state_bitwise: int, task_network=None) -> int:
        if not self.enabled:
            return 0
        total = 0
        for c, s in self.goal_pairs:
            total += self._goal_lower_bound(state_bitwise, c, s)
        return max(0, total)


# Optional convenience alias used by some loaders
def get_heuristic_class():
    return DepotsGPT5Heuristic6
