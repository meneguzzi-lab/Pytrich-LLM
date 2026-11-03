from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Any

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic0(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale (domain insights -> heuristic signals)
    ----------------------------------------------
    Depots has a small set of primitive actions (Drive/Lift/Drop/Load/Unload) and
    a few compound tasks (do_put_on, do_clear, do_get_truck, do_lift_crate,
    do_load_truck, do_unload_truck).

    In all training instances, the HTN ultimately tries to achieve many (on c s)
    relations. Achieving an (on c s) can require:
      - moving a truck between places (Drive)
      - extracting a crate from a stack (requires clearing the crate: clear c)
      - clearing the destination surface (clear s)
      - using hoists to Lift/Unload then Drop

    Key difficulty is *stacking dependencies* (clear predicates) plus possible
    *transport* (in/at relations + Drive). We build an efficient lower bound by
    summing, for each missing goal on(c,s), a relaxed minimum number of primitive
    actions needed, using only cheap bit checks on the state.

    This heuristic is NOT admissible in general (it sums per-goal estimates and
    may double-count shared actions). In Pytrich it is still very effective at
    reducing expansions; it behaves like a domain-specific goal-counting/relaxed
    cost heuristic.

    Implementation notes
    --------------------
    - Uses bitwise checks (state >> fact_id) & 1, never explicit state repr.
    - Parses grounded fact names of form "+pred[a,b]" / "-pred[a,b]".
    - Extracts goals via model.state_explicit_repr(model.goals) (required).
    - Precomputes:
        * fact ids by predicate and tuple(args)
        * static location of surfaces (pallets/crates) from initial facts
        * list of trucks

    Heuristic components per missing goal (on c s)
    ----------------------------------------------
    We approximate cost-to-achieve on(c,s) in current state:
      0 if already true.
      + need_clear_c: 1 if clear(c) false (must clear it somehow)
      + need_clear_s: 1 if clear(s) false
      + if lifting some hoist already holding c: small discount
      + if c is in some truck:
            + need_truck_at_dest: 1 if that truck not at dest place
            + base manipulation: 2 (Unload + Drop)
        elif c at some place:
            if c already at dest place:
                + base manipulation: 2 (Lift + Drop)
            else:
                + transport+manipulation lower bound:
                    1 (Load) + 1 (Drive) + 1 (Unload) + 1 (Drop) = 4

    Additionally, we add a small penalty for remaining abstract tasks in the
    current task network: each unresolved do_put_on contributes +1. This helps
    the search prefer states that progress through the HTN.
    """

    def __init__(self, model, depots_gpt52_heuristic_0: bool = True):
        super().__init__(model)
        self.depots_gpt52_heuristic_0 = depots_gpt52_heuristic_0

        # --- Build fast indices for facts ---
        # predicate -> {(arg1,arg2,...): fact_global_id}
        self._fact_id: Dict[str, Dict[Tuple[str, ...], int]] = {}
        # Also store reverse: global_id -> (pred, args)
        self._fid_to_pa: Dict[int, Tuple[str, Tuple[str, ...]]] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            targs = tuple(args)
            self._fact_id.setdefault(pred, {})[targs] = f.global_id
            self._fid_to_pa[f.global_id] = (pred, targs)

        # --- Cache goal on-facts (only +on[c,s]) ---
        self._goal_on: List[Tuple[str, str, int]] = []  # (c,s,fid)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                fid = self._fact_id.get("on", {}).get((args[0], args[1]))
                if fid is not None:
                    self._goal_on.append((args[0], args[1], fid))

        # --- Extract object lists from facts (best-effort) ---
        self._trucks: Set[str] = set()
        # truck appears in at[truck,place] and in[crate,truck]
        if "at" in self._fact_id:
            for (x, _p) in self._fact_id["at"].keys():
                if x.startswith("truck"):
                    self._trucks.add(x)
        if "in" in self._fact_id:
            for (_c, t) in self._fact_id["in"].keys():
                if t.startswith("truck"):
                    self._trucks.add(t)

        # --- Static surface location map from initial state (used as fallback) ---
        # We try to map surfaces (pallet or crate used as surface) to a place.
        # In Depots, (at surface place) is true for pallets and may be for crates.
        self._static_place_of: Dict[str, str] = {}
        init_state = self.model.initial_state
        at_map = self._fact_id.get("at", {})
        # iterate over all at facts (x, p), if true in init -> record
        for (x, p), fid in at_map.items():
            if self._is_true(init_state, fid):
                self._static_place_of[x] = p

        # Provide a quick list of places seen
        self._places: Set[str] = set(self._static_place_of.values())

    # -------------------- Core interface --------------------
    def __call__(self, node) -> int:
        state = node.state
        task_network = getattr(node, "task_network", [])

        h = 0

        # Sum relaxed costs for each missing goal on(c,s)
        for c, s, on_fid in self._goal_on:
            if self._is_true(state, on_fid):
                continue
            h += self._estimate_put_on_cost(state, c, s)

        # Small HTN-progress shaping: count remaining do_put_on tasks
        # (kept small to avoid dominating fact-based guidance)
        h += self._count_pending_put_on(task_network)

        return int(h)

    # -------------------- Estimation helpers --------------------
    def _estimate_put_on_cost(self, state: int, c: str, s: str) -> int:
        """Estimate relaxed primitive-action distance to achieve on(c,s)."""
        cost = 0

        # If some hoist is already lifting c, we can often finish with Drop
        lifting_hoist = self._any_hoist_lifting(state, c)

        # Need destination place (where surface s is)
        dest_place = self._get_place_of_surface(state, s)

        # Clear requirements (relaxed: treat each missing clear as +1)
        if not self._is_clear(state, c):
            cost += 1
        if not self._is_clear(state, s):
            cost += 1

        # If lifting already, Drop might be close
        if lifting_hoist is not None:
            # If we are lifting c, we essentially need Drop and maybe to be at dest place.
            # We do not know hoist location cheaply here; keep conservative small bound.
            # Drop alone is 1, but may require clear(s) (already accounted) and at(s,place).
            return max(1, cost)  # at least one action to make on(c,s)

        # Where is crate c? in truck, or at a place (or unknown)
        truck = self._crate_in_truck(state, c)
        if truck is not None:
            # Need truck at destination place to unload/drop
            if dest_place is not None and not self._is_at(state, truck, dest_place):
                cost += 1  # at least one Drive
            # Unload + Drop
            cost += 2
            return cost

        c_place = self._get_place_of_crate(state, c)
        if c_place is not None and dest_place is not None:
            if c_place == dest_place:
                # Lift + Drop
                cost += 2
            else:
                # Load + Drive + Unload + Drop
                cost += 4
            return cost

        # Fallback: unknown locations -> assume needs transport-ish effort
        cost += 4
        return cost

    def _count_pending_put_on(self, task_network) -> int:
        cnt = 0
        for t in task_network:
            name = getattr(t, "name", "")
            # Abstract task instances in Pytrich usually have name like "do_put_on" or include it
            if "do_put_on" in name:
                cnt += 1
        return cnt

    # -------------------- Bitwise predicate checks --------------------
    def _is_true(self, state: int, fact_global_id: int) -> bool:
        return ((state >> fact_global_id) & 1) == 1

    def _is_at(self, state: int, x: str, p: str) -> bool:
        fid = self._fact_id.get("at", {}).get((x, p))
        return fid is not None and self._is_true(state, fid)

    def _is_clear(self, state: int, x: str) -> bool:
        fid = self._fact_id.get("clear", {}).get((x,))
        return fid is not None and self._is_true(state, fid)

    def _crate_in_truck(self, state: int, c: str) -> Optional[str]:
        in_map = self._fact_id.get("in", {})
        # iterate only trucks seen; cheap because #trucks small (<=5 in instances)
        for t in self._trucks:
            fid = in_map.get((c, t))
            if fid is not None and self._is_true(state, fid):
                return t
        return None

    def _any_hoist_lifting(self, state: int, c: str) -> Optional[str]:
        # lifting[hoist,crate]
        lift_map = self._fact_id.get("lifting", {})
        # We cannot enumerate all hoists cheaply unless we scan keys; number of hoists is moderate.
        for (h, cc), fid in lift_map.items():
            if cc == c and self._is_true(state, fid):
                return h
        return None

    def _get_place_of_crate(self, state: int, c: str) -> Optional[str]:
        # Prefer dynamic at[c,p] if present
        at_map = self._fact_id.get("at", {})
        # We do not have list of all places; use observed places plus static fallback.
        # Try static place first if dynamic scanning is expensive.
        # But if crate moved, static may be wrong; therefore we also scan current at-facts for this crate.

        # Scan all at facts for crate c (number of at facts can be large; but only for 'at' keys,
        # and crates count is large. To keep it bounded, check static first, then verify quickly by
        # checking candidate places from _places.)
        if c in self._static_place_of:
            p0 = self._static_place_of[c]
            fid0 = at_map.get((c, p0))
            if fid0 is not None and self._is_true(state, fid0):
                return p0

        # Try all known places (small: depots+distributors)
        for p in self._places:
            fid = at_map.get((c, p))
            if fid is not None and self._is_true(state, fid):
                return p

        # As last resort, scan all at entries matching c
        for (x, p), fid in at_map.items():
            if x == c and self._is_true(state, fid):
                # cache place set growth
                self._places.add(p)
                return p

        return None

    def _get_place_of_surface(self, state: int, s: str) -> Optional[str]:
        # Surface is pallet or crate. Use at[s,p].
        at_map = self._fact_id.get("at", {})

        if s in self._static_place_of:
            p0 = self._static_place_of[s]
            fid0 = at_map.get((s, p0))
            if fid0 is not None and self._is_true(state, fid0):
                return p0

        for p in self._places:
            fid = at_map.get((s, p))
            if fid is not None and self._is_true(state, fid):
                return p

        for (x, p), fid in at_map.items():
            if x == s and self._is_true(state, fid):
                self._places.add(p)
                return p

        return None

    # -------------------- Fact-name parsing --------------------
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
