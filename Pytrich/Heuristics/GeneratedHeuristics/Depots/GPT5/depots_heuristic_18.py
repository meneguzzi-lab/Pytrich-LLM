from typing import Dict, List, Tuple, Optional, DefaultDict
from collections import defaultdict

# Fallback-safe import for the base Heuristic class
from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic18(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depots domain under Pytrich.

    Core idea (admissible lower bound): For each unsatisfied goal on[c, s],
    at least the following primitive actions are needed in the best case:
      - 1 Drop (final step to establish on[c, s]).
      - Achieve lifting(c):
          * If already lifting any hoist holds c: 0
          * Else if c is in some truck: at least 1 Unload
          * Else (c is on some surface or at a place): at least 1 Lift
      - Movement across places: if c's current place != place(s): at least 1 Drive
      - Clear target surface s: if not clear(s): at least 1 Lift (to clear s)
      - Clear c before lifting (when c is on a surface and not in truck or lifting): if not clear(c): at least +1

    This sums a safe, domain-informed lower bound per goal. The counts are
    conservative (never overestimate true cost), while exploiting key domain
    structure: hoists/lifting, trucks/in, clear/stacking, and place mismatches.

    Notes:
      - We use bitwise checks on facts (global_id) for speed.
      - Goals are read from model.goals via state_explicit_repr(...), then parsed.
      - We pre-index facts (on/at/in/lifting/clear) to enable O(1)/O(k) queries.
      - The heuristic is independent of the task network for safety and speed,
        but is tailored to the hierarchical structure by mirroring the cheapest
        decomposition patterns for do_put_on.

    """

    def __init__(self, model, depots_gpt5_heuristic_18: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.cfg = depots_gpt5_heuristic_18 or {}

        # Fast lookup: fact name -> global_id
        self._fact_name_to_id: Dict[str, int] = {}

        # Indexes to query state quickly
        self._on_fact_id: Dict[Tuple[str, str], int] = {}
        self._clear_fact_id: Dict[str, int] = {}
        self._at_index: DefaultDict[str, List[Tuple[str, int]]] = defaultdict(list)  # obj -> [(place, fact_id), ...]
        self._in_index: DefaultDict[str, List[Tuple[str, int]]] = defaultdict(list)  # crate -> [(truck, fact_id), ...]
        self._lifting_index: DefaultDict[str, List[Tuple[str, int, str]]] = defaultdict(list)  # crate -> [(hoist, fact_id, hoist)]

        # Build indices
        for fact in self.model.facts:
            name = fact.name
            self._fact_name_to_id[name] = fact.global_id
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                c, s = args
                self._on_fact_id[(c, s)] = fact.global_id
            elif pred == 'clear' and len(args) == 1:
                self._clear_fact_id[args[0]] = fact.global_id
            elif pred == 'at' and len(args) == 2:
                obj, place = args
                self._at_index[obj].append((place, fact.global_id))
            elif pred == 'in' and len(args) == 2:
                c, t = args
                self._in_index[c].append((t, fact.global_id))
            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self._lifting_index[c].append((h, fact.global_id, h))

        # Parse and cache goal on-facts
        self._goal_triples: List[Tuple[str, str, int]] = []  # (crate, surface, goal_fact_id)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                c, s = args
                fid = self._fact_name_to_id.get(gname, None)
                if fid is not None:
                    self._goal_triples.append((c, s, fid))

        # Optional debug flag
        self._debug = bool(self.cfg.get('debug', False))

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            core = name[1:]
            bidx = core.find('[')
            if bidx > 0 and core.endswith(']'):
                predicate = core[:bidx]
                args_str = core[bidx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return (state_bitwise >> fact_id) & 1 == 1

    def _true_place(self, state_bitwise: int, obj: str, local_cache: Dict[str, Optional[str]]) -> Optional[str]:
        # Per-evaluation cache to avoid repeated scans
        if obj in local_cache:
            return local_cache[obj]
        place = None
        for p, fid in self._at_index.get(obj, ()):  # type: ignore
            if self._is_true(state_bitwise, fid):
                place = p
                break
        local_cache[obj] = place
        return place

    def _crate_status(self, state_bitwise: int, crate: str, local_cache_places: Dict[str, Optional[str]]):
        """
        Returns a tuple describing the crate's current status:
          - ('lifting', hoist, place)
          - ('in_truck', truck, place_of_truck)
          - ('on_or_at', None, place_of_crate)
        """
        # Check if any hoist is currently lifting this crate
        for h, fid, hoist_obj in self._lifting_index.get(crate, ()):  # type: ignore
            if self._is_true(state_bitwise, fid):
                h_place = self._true_place(state_bitwise, hoist_obj, local_cache_places)
                return 'lifting', h, h_place

        # Check if crate is in any truck
        for t, fid in self._in_index.get(crate, ()):  # type: ignore
            if self._is_true(state_bitwise, fid):
                t_place = self._true_place(state_bitwise, t, local_cache_places)
                return 'in_truck', t, t_place

        # Otherwise assume it's somewhere at a place via at[crate, place]
        c_place = self._true_place(state_bitwise, crate, local_cache_places)
        return 'on_or_at', None, c_place

    def _clear_true(self, state_bitwise: int, obj: str) -> bool:
        fid = self._clear_fact_id.get(obj, None)
        return self._is_true(state_bitwise, fid)

    def evaluate(self, state_bitwise: int, task_network: List) -> int:
        if state_bitwise is None:
            return 0

        # Quick exit: if all goals achieved, h=0
        goals_ok = True
        for _c, _s, fid in self._goal_triples:
            if not self._is_true(state_bitwise, fid):
                goals_ok = False
                break
        if goals_ok:
            return 0

        h_val = 0
        local_place_cache: Dict[str, Optional[str]] = {}

        for c, s, goal_fid in self._goal_triples:
            # Skip satisfied goals
            if self._is_true(state_bitwise, goal_fid):
                continue

            # Base cost: must perform a Drop to achieve on[c, s]
            cost = 1

            # Determine current status/location of crate c
            status, carrier, c_place = self._crate_status(state_bitwise, c, local_place_cache)

            # Determine target surface place
            s_place = self._true_place(state_bitwise, s, local_place_cache)

            # Acquire 'lifting(c)' cheaply
            if status == 'lifting':
                # Already in lifting configuration: 0 extra for this subgoal
                pass
            elif status == 'in_truck':
                # Need at least one Unload to get lifting(c)
                cost += 1
            else:  # on_or_at
                # Need at least one Lift to get lifting(c)
                cost += 1
                # If c is not clear, clearing c requires at least one Lift of the blocker
                if not self._clear_true(state_bitwise, c):
                    cost += 1

            # Movement need: if places differ, at least one Drive is needed at some point
            # Use the most informative current location we have
            if c_place is not None and s_place is not None and c_place != s_place:
                cost += 1

            # Target surface must be clear before the final Drop
            if not self._clear_true(state_bitwise, s):
                cost += 1

            h_val += cost

        return h_val


# Exported factory (optional convenience for Pytrich configs)
def build(model, depots_gpt5_heuristic_18: Optional[dict] = None):
    return DepotsGPT5Heuristic18(model, depots_gpt5_heuristic_18)
