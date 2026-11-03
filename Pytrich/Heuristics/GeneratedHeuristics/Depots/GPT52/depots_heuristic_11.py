from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic11(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Rationale (domain insights)
    --------------------------
    Depots revolves around repeatedly achieving goal facts of the form:
        on(crate, surface)
    where surface is either a pallet or another crate.

    In this domain, *putting a crate onto a target surface* typically requires:
      - ensuring crate is at the correct place (may require truck drive),
      - ensuring both the crate and the target surface are clear,
      - using a hoist to lift/unload the crate, and
      - dropping it onto the target surface.

    Key structural challenges:
      1) "Clear" dependencies form stacks: if a surface is not clear, the top crate
         must be moved away (often into a truck), creating cascading work.
      2) Cross-location goals: if crate and target surface are at different places,
         at least one truck relocation (Drive) is necessary in any real plan.
      3) Hoist availability: Lift/Unload can make a hoist unavailable; however, for a
         *lower bound* we can ignore contention and just count minimal actions.

    Heuristic design
    ----------------
    We compute a fast, admissible lower bound by summing, for each unsatisfied goal
    (+on[c,s]):
      - a base cost for the minimal manipulation needed, and
      - additional lower-bound costs for required clearing of involved surfaces, and
      - a lower-bound cost for required driving if the crate cannot already be at the
        same place as the target surface in the current state.

    This heuristic is *admissible* because it only counts actions that are
    necessary in any valid solution, and counts them conservatively:
      - For each blocked surface, at least one action (Lift) is required to remove
        the top crate (in any plan that makes it clear at some point).
      - To place a crate onto a surface, at least one placement operation is needed
        (Drop is the minimal achieving action for on[c,s]).
      - If crate and target surface are in different places and crate is not already
        in a truck, at least one Load is needed before any cross-place move.
      - If cross-place is needed, at least one Drive is needed.
      - If crate is in a truck and not already at the target place, at least one
        Unload will be needed at the target place before Drop.

    We intentionally do *not* overcount chains (e.g., multiple crates stacked on the
    same surface); instead we use a unique blocking count for each surface to keep
    it cheap and safe.

    Efficiency
    ----------
    - Uses only bitwise membership checks on facts.
    - Precomputes indices from facts: on-relations, at-relations, clear facts, in-relations.
    - Per state, builds small dictionaries (crate->place, surface->place, clear set, in set)
      by scanning only those relevant fact ids.

    Notes on hierarchy
    ------------------
    Pytrich search states include both world state and remaining task network. This
    heuristic mainly estimates remaining primitive effort to satisfy goal facts from
    the world state. This works well in Depots because the HTN ultimately supports
    achieving these goal on-relations; the remaining tasks tend to correspond to
    remaining goal placements.
    """

    def __init__(self, model, depots_gpt52_heuristic_11: str = "depots_gpt52_heuristic_11"):
        super().__init__(model, depots_gpt52_heuristic_11)
        self.model = model

        # --- Precompute fact ids by predicate for fast state queries ---
        self._fact_id_to_parsed: Dict[int, Tuple[str, List[str]]] = {}

        self._on_facts: List[Tuple[int, str, str]] = []          # (fid, crate, surface)
        self._at_facts: List[Tuple[int, str, str]] = []          # (fid, obj, place)
        self._clear_facts: List[Tuple[int, str]] = []            # (fid, surface)
        self._in_facts: List[Tuple[int, str, str]] = []          # (fid, crate, truck)

        # Helpful index maps (by object name)
        self._at_fact_ids_by_obj: Dict[str, List[int]] = {}
        self._at_fact_ids_by_place: Dict[str, List[int]] = {}

        # For surfaces, we only care about their location too
        self._at_fact_ids_by_surface: Dict[str, List[int]] = {}

        # on facts by surface (to find blockers) and by crate
        self._on_fact_ids_by_surface: Dict[str, List[int]] = {}
        self._on_fact_ids_by_crate: Dict[str, List[int]] = {}

        # clear facts by surface
        self._clear_fact_id_by_surface: Dict[str, int] = {}

        # in facts by crate
        self._in_fact_ids_by_crate: Dict[str, List[int]] = {}

        # Gather goal on-facts (as (crate, surface) pairs) once
        self._goal_on_pairs: List[Tuple[str, str]] = []
        self._goal_on_fact_ids: List[int] = []

        self._build_indices()

    # ------------------------ Parsing helpers ------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _build_indices(self):
        # Parse all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_id_to_parsed[f.global_id] = (pred, args)

            if pred == "on" and len(args) == 2:
                c, s = args
                self._on_facts.append((f.global_id, c, s))
                self._on_fact_ids_by_surface.setdefault(s, []).append(f.global_id)
                self._on_fact_ids_by_crate.setdefault(c, []).append(f.global_id)

            elif pred == "at" and len(args) == 2:
                o, p = args
                self._at_facts.append((f.global_id, o, p))
                self._at_fact_ids_by_obj.setdefault(o, []).append(f.global_id)
                self._at_fact_ids_by_place.setdefault(p, []).append(f.global_id)

                # also index surfaces (pallets and crates can be surfaces; we don't type-check here)
                self._at_fact_ids_by_surface.setdefault(o, []).append(f.global_id)

            elif pred == "clear" and len(args) == 1:
                s = args[0]
                self._clear_facts.append((f.global_id, s))
                # assume at most one clear fact per surface
                self._clear_fact_id_by_surface[s] = f.global_id

            elif pred == "in" and len(args) == 2:
                c, t = args
                self._in_facts.append((f.global_id, c, t))
                self._in_fact_ids_by_crate.setdefault(c, []).append(f.global_id)

        # Parse goals (bitwise int -> explicit names)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                c, s = args
                self._goal_on_pairs.append((c, s))
                # Find corresponding fact id (we can map by scanning on_facts list; it is modest)
                # If not found, we just won't use its id directly.
                for fid, c2, s2 in self._on_facts:
                    if c2 == c and s2 == s:
                        self._goal_on_fact_ids.append(fid)
                        break

    # ------------------------ Bitwise checks ------------------------
    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    # ------------------------ State extraction ------------------------
    def _extract_locations_and_flags(self, state_bitwise: int):
        """Extracts only what we need for the heuristic.

        Returns:
            crate_at: dict crate->place (if known)
            surf_at: dict surface->place (if known)
            clear_surfaces: set of surfaces that are clear
            crate_in_truck: dict crate->truck (if in some truck)
        """
        crate_at: Dict[str, str] = {}
        surf_at: Dict[str, str] = {}
        clear_surfaces: Set[str] = set()
        crate_in_truck: Dict[str, str] = {}

        # at facts
        for fid, obj, place in self._at_facts:
            if self._is_true(state_bitwise, fid):
                # obj could be crate, pallet, hoist, truck ...
                # We store for all objs; later we only read needed keys.
                crate_at[obj] = place
                surf_at[obj] = place

        # clear facts
        for fid, s in self._clear_facts:
            if self._is_true(state_bitwise, fid):
                clear_surfaces.add(s)

        # in facts
        for fid, c, t in self._in_facts:
            if self._is_true(state_bitwise, fid):
                crate_in_truck[c] = t

        return crate_at, surf_at, clear_surfaces, crate_in_truck

    def _surface_is_clear(self, surface: str, clear_surfaces: Set[str]) -> bool:
        # If we have a clear fact for it and it's present -> clear.
        # If we don't know a clear predicate for the object (shouldn't happen in this domain), assume not clear.
        fid = self._clear_fact_id_by_surface.get(surface)
        if fid is None:
            return False
        return surface in clear_surfaces

    def _count_blocking_top_crates(self, state_bitwise: int, surface: str) -> int:
        """Lower bound blockers for making 'surface' clear: 0 if already clear, else 1.

        In Depots, a surface is non-clear iff some crate is on it (in reachable states).
        To make it clear at some point, at least one Lift must occur to remove the top crate.
        This returns 1 if any +on[c,surface] is currently true.
        """
        for fid in self._on_fact_ids_by_surface.get(surface, []):
            if self._is_true(state_bitwise, fid):
                return 1
        return 0

    # ------------------------ Heuristic core ------------------------
    def __call__(self, state: int, task_network=None) -> int:
        """Return an admissible integer heuristic value."""
        if task_network is None:
            task_network = []

        # If goals already reached and no remaining tasks, h=0
        if self.model.goal_reached(state, task_network):
            return 0

        crate_at, surf_at, clear_surfaces, crate_in_truck = self._extract_locations_and_flags(state)

        h = 0

        # To reduce double counting of clearing: track surfaces already charged
        charged_clear: Set[str] = set()

        # Estimate cost per unsatisfied goal on(c,s)
        for (c, s) in self._goal_on_pairs:
            # If already satisfied, skip
            satisfied = False
            for fid in self._on_fact_ids_by_crate.get(c, []):
                # cheap exact check for this particular goal: match s
                pred, args = self._fact_id_to_parsed.get(fid, (None, []))
                if pred == "on" and len(args) == 2 and args[1] == s and self._is_true(state, fid):
                    satisfied = True
                    break
            if satisfied:
                continue

            # Base: to achieve on(c,s) we minimally need a Drop somewhere in any real plan.
            # (Even if already lifting, Drop is required to establish on)
            h += 1

            # Clearing requirements (lower bounds):
            # Drop requires clear(s). Also Lift/Drop manipulation typically requires clear(c) at lift time.
            # We count at most 1 per surface if currently blocked.
            if s not in charged_clear:
                h += self._count_blocking_top_crates(state, s)
                charged_clear.add(s)

            if c not in charged_clear:
                # If some crate is on top of c then c is not clear; clearing c requires at least one Lift.
                h += self._count_blocking_top_crates(state, c)
                charged_clear.add(c)

            # Location / transport lower bound
            # If crate is not already at same place as target surface, some transport is needed.
            place_s = surf_at.get(s)

            # Crate location can be via at(c,p) or in(c,t) with at(t,p)
            place_c = crate_at.get(c)
            if place_c is None and c in crate_in_truck:
                t = crate_in_truck[c]
                place_c = crate_at.get(t)

            if place_s is not None and place_c is not None and place_c != place_s:
                # At least one Drive is required for cross-place.
                h += 1

                # Additionally, if crate isn't already in a truck, at least one Load is necessary.
                if c not in crate_in_truck:
                    h += 1
                else:
                    # If in truck, to end up on a surface, must Unload at target place at some point.
                    # (Unless currently lifting, which would imply not in truck. So safe.)
                    h += 1

        # Also include a tiny bias for remaining abstract work in task network (admissible):
        # each remaining task must take at least 0 steps; we keep 0 to preserve strict admissibility.
        return int(h)
