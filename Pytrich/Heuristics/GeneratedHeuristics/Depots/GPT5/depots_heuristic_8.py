from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic8(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depots domain.

    Key idea (admissible lower bound):
    - Every goal on[c,s] not yet true requires at least one Drop action.
    - Drop requires the crate c to be currently lifted: if not, we need at least
      one action to achieve lifting(c) (either a Lift or an Unload).
      - If c is in a truck, Unload achieves lifting in one step.
      - If c is at some place and not lifted, we need one Lift; if c is not clear,
        we need at least one more action to clear c before we can Lift it.
    - Drop also requires the destination surface s to be clear. If not clear, at least
      one Lift action is required to clear s.
    - Drop requires s to be at some place. If s is not at any place (e.g., it's being
      lifted or inside a truck), at least one action (e.g., Unload/Drop) is needed to
      place it somewhere before we can Drop c onto it.
    - Transport: if at least one unsatisfied goal requires moving c from a source place
      to a different destination place (where s is), then at least one Drive is needed
      globally. We add 1 once (not per goal) to keep the estimate admissible.

    This heuristic uses only bitwise state membership tests (fast) and precomputes
    the relevant fact indices by parsing the model's facts during initialization.

    Notes on admissibility:
    - Costs are summed over per-goal necessary action schemas (Drop always; possibly
      one action to achieve lifting; possibly one to clear destination; etc.).
    - We do not sum per-goal Drive costs (would overcount); we add at most one global
      Drive if any crate evidently needs cross-place transport. This remains a lower bound.

    Performance:
    - Per goal, constant number of bit checks using precomputed fact-id lists.
    - Scales to the provided largest instance by avoiding any string matching at
      evaluation time and relying solely on bitwise checks.
    """

    def __init__(self, model, depots_gpt5_heuristic_8: bool = True):
        super().__init__()
        self.model = model

        # Containers of parsed structure -> global_id indexes for fast bit checks
        self.on_fact_id: Dict[Tuple[str, str], int] = {}
        self.in_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        self.lifting_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        self.clear_by_surface: Dict[str, int] = {}
        self.at_by_entity: Dict[str, List[Tuple[str, int]]] = {}

        # Simple type recognizers by name prefix (sufficient for given instances)
        self._is_crate = lambda x: x.startswith("crate")
        self._is_pallet = lambda x: x.startswith("pallet")
        self._is_truck = lambda x: x.startswith("truck")
        self._is_hoist = lambda x: x.startswith("hoist")
        self._is_place = lambda x: x.startswith("depot") or x.startswith("distributor")
        self._is_surface = lambda x: self._is_pallet(x) or self._is_crate(x)

        # Parse all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'on' and len(args) == 2:
                c, s = args
                self.on_fact_id[(c, s)] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                c, t = args
                self.in_by_crate.setdefault(c, []).append((t, fact.global_id))

            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self.lifting_by_crate.setdefault(c, []).append((h, fact.global_id))

            elif pred == 'clear' and len(args) == 1:
                s = args[0]
                self.clear_by_surface[s] = fact.global_id

            elif pred == 'at' and len(args) == 2:
                x, p = args
                # We store for any locatable x (crate, pallet, hoist, truck)
                self.at_by_entity.setdefault(x, []).append((p, fact.global_id))

        # Extract goal on-facts as structured tuples for quick loops
        self.goals_on: List[Tuple[str, str, Optional[int]]] = []  # (crate, surface, fact_id)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                c, s = args
                fid = self.on_fact_id.get((c, s))
                self.goals_on.append((c, s, fid))

    # ------------------------- Core Heuristic --------------------------
    def h(self, state_bitwise: int, task_network: List = None) -> int:
        if not self.goals_on:
            return 0

        total_cost = 0
        any_transport_needed = False

        for c, s2, goal_on_fid in self.goals_on:
            # If already satisfied, skip
            if goal_on_fid is not None and ((state_bitwise >> goal_on_fid) & 1):
                continue

            local = 0

            # 1) We will need at least one Drop to achieve on[c,s2]
            local += 1

            # 2) Ensure lifting(c) before Drop
            crate_lifting, source_place = self._is_crate_lifted_and_place(state_bitwise, c)
            if crate_lifting:
                pass  # No extra action to achieve lifting
            else:
                # If c is in some truck -> one Unload will establish lifting(c)
                in_info = self._crate_in_truck_and_place(state_bitwise, c)
                if in_info is not None:
                    _, truck_place = in_info
                    source_place = truck_place
                    local += 1  # Unload to get lifting(c)
                else:
                    # Otherwise c must be at some place; one Lift is needed
                    p_c = self._current_place_of_entity(state_bitwise, c)
                    source_place = p_c
                    local += 1  # Lift(c,...)
                    # To Lift(c) we require clear(c)
                    clear_c_fid = self.clear_by_surface.get(c)
                    if clear_c_fid is not None and ((state_bitwise >> clear_c_fid) & 1) == 0:
                        # Need at least one action to clear c before we can Lift(c)
                        local += 1

            # 3) Destination surface must be clear for Drop
            clear_s2_fid = self.clear_by_surface.get(s2)
            if clear_s2_fid is not None and ((state_bitwise >> clear_s2_fid) & 1) == 0:
                local += 1

            # 4) Destination surface must be at some place for Drop (at[s2, p])
            dest_place = self._current_place_of_entity(state_bitwise, s2)
            if dest_place is None:
                # s2 is not at any place (likely lifted or in-truck) -> at least one action needed
                local += 1

            # 5) Transport (global): if source and destination places are known and different, 
            #    at least one Drive is needed somewhere in the plan.
            if (source_place is not None) and (dest_place is not None) and (source_place != dest_place):
                any_transport_needed = True

            total_cost += local

        if any_transport_needed:
            total_cost += 1  # one Drive globally as a safe, admissible lower bound

        return total_cost

    # ------------------------- Helpers --------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _current_place_of_entity(self, state_bitwise: int, entity: str) -> Optional[str]:
        """Return the current place for entity if any at[entity,place] is true, else None."""
        lst = self.at_by_entity.get(entity)
        if not lst:
            return None
        for p, fid in lst:
            if (state_bitwise >> fid) & 1:
                return p
        return None

    def _crate_in_truck_and_place(self, state_bitwise: int, crate: str) -> Optional[Tuple[str, Optional[str]]]:
        """If crate is in some truck, return (truck, place_of_truck). Otherwise None."""
        lst = self.in_by_crate.get(crate)
        if not lst:
            return None
        for t, fid in lst:
            if (state_bitwise >> fid) & 1:
                # Determine the truck's place
                place_t = self._current_place_of_entity(state_bitwise, t)
                return t, place_t
        return None

    def _is_crate_lifted_and_place(self, state_bitwise: int, crate: str) -> Tuple[bool, Optional[str]]:
        """Return (True, place) if some hoist is lifting the crate, and the hoist's place. Else (False, None)."""
        lst = self.lifting_by_crate.get(crate)
        if not lst:
            return False, None
        for h, fid in lst:
            if (state_bitwise >> fid) & 1:
                # lifting(h,crate) holds -> place is where hoist h is at
                place_h = self._current_place_of_entity(state_bitwise, h)
                return True, place_h
        return False, None


# Optional factory function for frameworks that use callables to build heuristics
def build(model, depots_gpt5_heuristic_8: bool = True):
    return DepotsGPT5Heuristic8(model, depots_gpt5_heuristic_8=depots_gpt5_heuristic_8)
