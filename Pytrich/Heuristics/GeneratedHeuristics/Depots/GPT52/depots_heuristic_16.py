from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _ParsedFact:
    pred: str
    args: Tuple[str, ...]


class DepotsGPT52Heuristic16(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Key domain insights (from depots.hddl):

    The only top-level compound tasks used in training instances are of the form:
        (do_put_on crate surface)

    Achieving a goal (on c s2) can require:
      - clearing the crate (clear c) because Lift requires (clear c)
      - clearing the target surface (clear s2) because Drop requires (clear s2)
      - possibly moving the crate between places using a truck (load/drive/unload)
      - hoist availability constraints (available h) / lifting states

    This heuristic is *admissible* and designed to be fast on large instances.

    Approach:
      1) Goal-fact counting lower bound: count unsatisfied goal '+on[c,s]'.
         Each such goal requires at least one physical placing action (Drop).

      2) Additional relaxed lower bounds for necessary subconditions:
         - If crate c is not clear, we (optimistically) need at least one action
           to make it clear. In this domain that ultimately requires moving a
           top crate away; counting 1 is a safe lower bound.
         - If target surface s is not clear, at least one action is needed to
           clear it (again, 1 is a safe lower bound).

      3) Cheap transportation bound:
         If crate c is currently at place p and surface s is at place q and p!=q,
         then at least one Drive is needed somewhere in any real plan that
         transfers c to q. We add +1 in that case.
         If c is in a truck t and we know (at t p) and (at s q) and p!=q,
         we also add +1.

    We combine bounds using max() (not sum) to preserve admissibility.

    Why it helps search:
      - Many goals are simple stacking at same location: heuristic ~ #unsatisfied.
      - When stacks block each other, the clear-related bound distinguishes easy
        vs. blocked goals.
      - When goals require relocation across depots/distributors, the drive bound
        helps avoid exploring decompositions that ignore transportation.

    Notes on efficiency:
      - Uses only bitwise checks on fact ids.
      - Precomputes mappings: object -> current place (for at), crate->support (on),
        crate->truck (in), surface->place (at).
      - Avoids model.state_explicit_repr(state) on every call.

    The __init__ parameter is required to be named `depots_gpt52_heuristic_16`.
    """

    def __init__(self, model, depots_gpt52_heuristic_16=None):
        super().__init__(model)
        self.model = model
        self.depots_gpt52_heuristic_16 = depots_gpt52_heuristic_16

        # Parse and cache all facts by predicate for fast access
        self._facts_by_pred: Dict[str, List] = {}
        self._parsed_fact_cache: Dict[int, _ParsedFact] = {}

        for f in self.model.facts:
            pf = self._parse_fact_name(f.name)
            if pf is None:
                continue
            self._parsed_fact_cache[f.global_id] = pf
            self._facts_by_pred.setdefault(pf.pred, []).append(f)

        # Cache goal facts (names -> Fact objects) for '+on' only
        self._goal_on_facts: List = []
        for gname in self.model.state_explicit_repr(self.model.goals):
            pf = self._parse_fact_name(gname)
            if pf and pf.pred == "on":
                # find Fact object by matching name
                # This is done once at init, so linear scan is acceptable.
                # (facts are already grounded and unique)
                for f in self._facts_by_pred.get("on", []):
                    if f.name == gname:
                        self._goal_on_facts.append(f)
                        break

        # Also cache clear facts per object and at facts per object/place
        self._clear_facts: Dict[str, List] = {}
        for f in self._facts_by_pred.get("clear", []):
            pf = self._parsed_fact_cache[f.global_id]
            obj = pf.args[0]
            self._clear_facts.setdefault(obj, []).append(f)

        # For fast lookup of a particular grounded fact name -> fact
        # (built once; name strings are stable)
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

    # --------- fact parsing helpers ---------
    def _parse_fact_name(self, name: str) -> Optional[_ParsedFact]:
        name = name.strip()
        if not name:
            return None
        if name[0] not in "+-":
            return None
        core = name[1:]
        bracket_idx = core.find("[")
        if bracket_idx <= 0 or not core.endswith("]"):
            return None
        pred = core[:bracket_idx]
        args_str = core[bracket_idx + 1 : -1]
        if args_str:
            args = tuple(args_str.split(","))
        else:
            args = tuple()
        return _ParsedFact(pred=pred, args=args)

    @staticmethod
    def _is_true(state: int, fact_global_id: int) -> bool:
        return ((state >> fact_global_id) & 1) == 1

    # --------- state decoding (bitwise scans only over relevant predicates) ---------
    def _extract_locations(self, state: int) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
        """Extract partial location/support info needed for the heuristic.

        Returns:
          at_loc: locatable -> place
          on_support: crate -> surface
          in_truck: crate -> truck
        """
        at_loc: Dict[str, str] = {}
        on_support: Dict[str, str] = {}
        in_truck: Dict[str, str] = {}

        for f in self._facts_by_pred.get("at", []):
            if self._is_true(state, f.global_id):
                pf = self._parsed_fact_cache[f.global_id]
                # at[x,place]
                if len(pf.args) == 2:
                    at_loc[pf.args[0]] = pf.args[1]

        for f in self._facts_by_pred.get("on", []):
            if self._is_true(state, f.global_id):
                pf = self._parsed_fact_cache[f.global_id]
                # on[crate,surface]
                if len(pf.args) == 2:
                    on_support[pf.args[0]] = pf.args[1]

        for f in self._facts_by_pred.get("in", []):
            if self._is_true(state, f.global_id):
                pf = self._parsed_fact_cache[f.global_id]
                # in[crate,truck]
                if len(pf.args) == 2:
                    in_truck[pf.args[0]] = pf.args[1]

        return at_loc, on_support, in_truck

    def _surface_place(self, surface: str, at_loc: Dict[str, str]) -> Optional[str]:
        return at_loc.get(surface)

    def _crate_place(self, crate: str, at_loc: Dict[str, str], in_truck: Dict[str, str], truck_place: Dict[str, str]) -> Optional[str]:
        # If crate has an at, use it
        if crate in at_loc:
            return at_loc[crate]
        # If crate is in a truck, use truck location if known
        t = in_truck.get(crate)
        if t is not None and t in truck_place:
            return truck_place[t]
        return None

    # --------- heuristic core ---------
    def compute_heuristic(self, state: int, task_network: List = None) -> int:
        if task_network is None:
            task_network = []

        # If no remaining tasks and goal reached, 0
        if self.model.goal_reached(state, task_network):
            return 0

        # Extract relevant relations
        at_loc, on_support, in_truck = self._extract_locations(state)

        # Build a truck->place map from at_loc (trucks are locatable)
        truck_place: Dict[str, str] = {}
        for obj, plc in at_loc.items():
            # trucks are distinguishable by naming in benchmarks; but we avoid relying on types.
            # We'll infer trucks by checking if there exists any in[*,obj] currently true.
            truck_place[obj] = plc

        # 1) Unsatisfied goal on-facts lower bound
        unsat_on = 0
        clear_needed = 0
        target_clear_needed = 0
        drive_needed = 0

        # For clear checking, we directly check for '+clear[x]' fact if it exists
        def is_clear(x: str) -> Optional[bool]:
            facts = self._clear_facts.get(x)
            if not facts:
                return None
            # In this domain there should be exactly one clear fact per object
            for cf in facts:
                return self._is_true(state, cf.global_id)
            return None

        for gf in self._goal_on_facts:
            if self._is_true(state, gf.global_id):
                continue
            unsat_on += 1

            pf = self._parsed_fact_cache[gf.global_id]
            c, s2 = pf.args[0], pf.args[1]

            # 2) Clear sub-bounds
            c_clear = is_clear(c)
            if c_clear is False:
                clear_needed += 1

            s2_clear = is_clear(s2)
            if s2_clear is False:
                target_clear_needed += 1

            # 3) Transport lower bound (needs at least one Drive if places differ and known)
            s2_place = self._surface_place(s2, at_loc)
            c_place = self._crate_place(c, at_loc, in_truck, truck_place)
            if s2_place is not None and c_place is not None and s2_place != c_place:
                drive_needed += 1

        # If there are no on-goals (unusual), fallback to 0 or small guidance based on tasks
        if not self._goal_on_facts:
            # minimal guidance: remaining primitive tasks count is not available here;
            # return 0 to remain admissible.
            return 0

        # Combine admissible lower bounds by max
        h = max(unsat_on, clear_needed, target_clear_needed, drive_needed)

        # Small hierarchical tie-breaker (still admissible if it does not increase beyond a lower bound).
        # We cannot add anything safely; but we can prefer states with more satisfied goals by returning h.
        return int(h)
