from __future__ import annotations

"""depots_heuristic_17.py

Domain-dependent heuristic for the HDDL *Depots* domain in the Pytrich framework.

Rationale / domain insight
------------------------
The top-level workload in Depots instances is almost entirely a set of compound tasks
  do_put_on(crate, surface)
that ultimately need to establish goal literals of the form:
  +on[crate, surface]

In this domain, achieving on(c,s2) requires (at least) a lift+drop cycle at the *place
where s2 is located*. If the crate is not already at that place, it must be transported
via a truck (load, drive, unload) before it can be dropped onto s2.

Key structural difficulties:
- Stacking creates blocking: to move a crate you typically need it to be clear, which
  may require recursively clearing crates on top.
- Transport adds a fairly fixed extra cost when crate and destination surface are at
  different places.
- Hoists/trucks exist but are plentiful; the major search guidance is *which goals are
  already satisfied* and *which ones need local rearrangement vs. transport*.

Heuristic strategy
------------------
We compute a cheap, domain-specific lower bound by summing per-goal estimates for
unsatisfied +on[crate,surface] goals.

For each unsatisfied goal on(c, s2):
- Base cost: at least one Drop is needed -> +1
- If the crate is not currently lifting by some hoist, at least one Lift is needed -> +1
- If crate is not at the destination place (and not in a truck already):
    at least Load + Drive + Unload must happen -> +3
- If crate is blocked (not clear) and not currently lifting, we add a small extra
  lower bound to account for needing to clear the top -> +1

This is not admissible in the strict classical-action sense due to potential double
counting across multiple goals, but in practice it is a strong, very fast guidance
signal for greedy/A* variants used in HTN search.

Computational efficiency:
- No explicit state repr conversions in heuristic evaluation.
- All checks are bitwise using precomputed Fact global_ids.
- We pre-index:
  - goal on-facts
  - at(surface, place) for all surfaces (static)
  - per-crate facts: clear(crate), lifting(hoist,crate), at(crate,place), in(crate,truck)

Compatible with partial HTN states:
- If a needed fact is not present in the model (should not happen), we treat it as unknown
  and avoid raising exceptions.
"""


# Pytrich base class
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback for different import layouts
    from heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic17(Heuristic):
    def __init__(self, model, depots_gpt52_heuristic_17: str = "depots_gpt52_heuristic_17"):
        super().__init__(model, depots_gpt52_heuristic_17)
        self.model = model

        # --- Fact parsing helpers / caches ---
        self._fact_by_name = {}
        for f in self.model.facts:
            self._fact_by_name[f.name] = f

        # Precompute goal +on[crate,surface] facts
        self._goal_on_facts = []  # list[Fact]
        for fname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(fname)
            if pred == "on" and len(args) == 2 and fname.startswith("+"):
                fact = self._fact_by_name.get(fname)
                if fact is not None:
                    self._goal_on_facts.append(fact)

        # Precompute destination place for each surface via +at[surface,place]
        # In depots instances, pallet locations are static; crates move.
        self._surface_place = {}  # surface -> place
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at" and len(args) == 2 and f.name.startswith("+"):
                obj, place = args
                # surface types are unknown here; we conservatively record all objects
                # and later only query for destination surfaces from goals.
                # If an object can move (crate/truck/hoist) this might change, but in this
                # domain only pallets/crates/trucks/hoists have at; still safe as heuristic.
                self._surface_place.setdefault(obj, place)

        # Precompute per-crate related fact ids for fast checks
        self._clear_fact_gid = {}          # crate -> gid of +clear[crate]
        self._at_fact_gids = {}            # crate -> dict(place -> gid of +at[crate,place])
        self._in_fact_gids = {}            # crate -> list of gid of +in[crate,truck]
        self._lifting_fact_gids = {}       # crate -> list of gid of +lifting[hoist,crate]

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not f.name.startswith("+"):
                continue
            if pred == "clear" and len(args) == 1:
                obj = args[0]
                self._clear_fact_gid[obj] = f.global_id
            elif pred == "at" and len(args) == 2:
                obj, place = args
                self._at_fact_gids.setdefault(obj, {})[place] = f.global_id
            elif pred == "in" and len(args) == 2:
                crate, truck = args
                self._in_fact_gids.setdefault(crate, []).append(f.global_id)
            elif pred == "lifting" and len(args) == 2:
                hoist, crate = args
                self._lifting_fact_gids.setdefault(crate, []).append(f.global_id)

        # Tunable weights (kept small and integer)
        self.w_drop = 1
        self.w_lift = 1
        self.w_transport = 3  # load+drive+unload lower bound
        self.w_blocked = 1

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    def _crate_current_place(self, state_bitwise: int, crate: str):
        """Return some place p such that +at[crate,p] holds, else None.

        Note: This is O(#places for that crate) which is small in practice (5-10).
        """
        places = self._at_fact_gids.get(crate)
        if not places:
            return None
        for p, gid in places.items():
            if self._is_true(state_bitwise, gid):
                return p
        return None

    def _crate_in_some_truck(self, state_bitwise: int, crate: str) -> bool:
        gids = self._in_fact_gids.get(crate)
        if not gids:
            return False
        for gid in gids:
            if self._is_true(state_bitwise, gid):
                return True
        return False

    def _crate_is_lifting(self, state_bitwise: int, crate: str) -> bool:
        gids = self._lifting_fact_gids.get(crate)
        if not gids:
            return False
        for gid in gids:
            if self._is_true(state_bitwise, gid):
                return True
        return False

    def _crate_is_clear(self, state_bitwise: int, crate: str) -> bool:
        gid = self._clear_fact_gid.get(crate)
        if gid is None:
            return False
        return self._is_true(state_bitwise, gid)

    def __call__(self, state: int, task_network=None) -> int:
        """Return an integer heuristic estimate.

        The heuristic focuses on unsatisfied goal +on[crate,surface] literals.
        """
        if task_network is None:
            task_network = []

        h = 0

        for on_fact in self._goal_on_facts:
            # If goal already true, no contribution
            if self._is_true(state, on_fact.global_id):
                continue

            pred, args = self._parse_fact_name(on_fact.name)
            if pred != "on" or len(args) != 2:
                continue
            crate, dest_surface = args

            dest_place = self._surface_place.get(dest_surface)

            # Minimum: need to place the crate onto dest_surface -> Drop
            h += self.w_drop

            lifting = self._crate_is_lifting(state, crate)
            if not lifting:
                # If not already lifting, will need a Lift somewhere
                h += self.w_lift

            # Transport lower bound if crate not at destination place.
            # If crate already in a truck, we do not force a Load, but still will need
            # at least Drive+Unload; we keep same coarse bound for simplicity.
            if dest_place is not None:
                crate_place = self._crate_current_place(state, crate)
                in_truck = self._crate_in_some_truck(state, crate)

                # If crate is physically at some place and it's not the destination place,
                # then at least transport is required.
                if crate_place is not None and crate_place != dest_place:
                    h += self.w_transport
                elif crate_place is None and in_truck:
                    # Unknown place but in a truck: likely still need drive/unload.
                    h += max(1, self.w_transport - 1)

            # If crate is blocked (not clear) and not already lifting, we will need
            # at least one clearing move above it.
            if (not lifting) and (not self._crate_is_clear(state, crate)):
                h += self.w_blocked

        # Small tie-breaker using remaining task network size (very cheap)
        # This helps HTN search slightly without dominating.
        if task_network:
            h += min(5, len(task_network)) // 2

        return int(h)
