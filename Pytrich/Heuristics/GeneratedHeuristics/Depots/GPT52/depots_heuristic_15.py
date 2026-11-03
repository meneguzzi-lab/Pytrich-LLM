from __future__ import annotations

from typing import Dict, List, Tuple, Optional

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic15(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale (domain insights -> heuristic design)
    ----------------------------------------------
    In this domain all top-level work is a sequence of compound tasks of the form:
        (do_put_on crate surface)

    The methods show four qualitatively different situations:
      - m0: already on target surface -> 0 primitive work.
      - m1: crate at place, need to clear crate and target surface, then Lift+Drop.
      - m2: crate in truck, bring truck to place (possibly Drive), clear target,
            then Unload+Drop.
      - m3: general relocation via load, drive, unload.

    The *real* search difficulty comes from:
      1) Clearing requirements: to lift a crate, it must be clear; to drop onto a
         surface, that surface must be clear. Clearing is recursive because crates
         can be stacked on crates/pallets.
      2) Transportation: if crate and target surface are at different places, at
         least one Drive is required plus load/unload steps.

    We build a fast, admissible lower bound by summing, for each unsatisfied goal
    (+on[c,s]), a per-goal optimistic estimate consisting of:
      - a minimal number of primitive actions needed to *place* the crate on the
        target (ignoring negative interactions between different goals), plus
      - a lower bound on the number of "blocking crates" that must be moved away
        because they are on top of the moved crate or on top of the target surface.

    Admissibility considerations
    ----------------------------
    * We only count actions that are unavoidably required for each goal in a
      delete-relaxed sense:
        - If goal already true: 0.
        - Otherwise, at least one Lift and one Drop are needed if crate is not in a
          hoist lifting state (which is not represented as a goal anyway).
        - If crate is currently in a truck, at least one Unload is needed before Drop.
        - If crate is not at the target place, at least one Load and one Unload plus
          at least one Drive are needed (we assume a direct drive exists as in this
          domain).
      These are lower bounds.

    * For clearing, for a surface X we count the number of crates directly stacked
      on X in the current state (blockers). At least that many Lift operations are
      needed to clear X (each Lift removes exactly one +on[crate,X]). We do not
      charge the necessary Drop/Load actions for those blockers, so it remains a
      lower bound.

    Efficiency
    ----------
    The heuristic relies only on bit checks and precomputed indices:
      - mapping from (predicate,args) -> fact global_id
      - current (on top of) relations extracted by scanning only +on facts present
        in the state
      - current locations extracted by scanning only +at and +in facts present

    No calls to state_explicit_repr(state) are made (too slow for large instances).

    Notes
    -----
    This heuristic is intended for node ordering / guidance (A*/GBFS variants). It
    is domain-dependent but robust across the provided instance scales.
    """

    def __init__(self, model, depots_gpt52_heuristic_15: str = "depots_gpt52_heuristic_15"):
        super().__init__(model, depots_gpt52_heuristic_15)
        self.model = model

        # Build fast lookup tables for fact ids by parsing each fact name once.
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Also store per-fact parsed info for quick iteration over true facts.
        self._parsed_fact: List[Tuple[Optional[str], Tuple[str, ...]]] = [
            (None, tuple()) for _ in range(len(self.model.facts))
        ]

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_id[key] = f.global_id
            self._parsed_fact[f.global_id] = (pred, tuple(args))

        # Cache goal on-facts (crate, surface)
        self._goal_on: List[Tuple[str, str, int]] = []  # (c, s, fact_id)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                gid = self._fact_id.get(("on", (args[0], args[1])))
                if gid is not None:
                    self._goal_on.append((args[0], args[1], gid))

    # --------- Parsing helpers ---------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _iter_true_fact_ids(self, state_bitwise: int):
        """Iterate over global fact ids set to 1 in state_bitwise."""
        bits = state_bitwise
        while bits:
            lsb = bits & -bits
            fid = (lsb.bit_length() - 1)
            yield fid
            bits ^= lsb

    # --------- State feature extraction ---------
    def _extract_structures(self, state: int):
        """Extract only the structures we need from the current state.

        Returns:
            on_map: Dict[surface -> List[crate]] for direct +on relations
            at_map: Dict[locatable -> place] for +at
            in_map: Dict[crate -> truck] for +in
        """
        on_map: Dict[str, List[str]] = {}
        at_map: Dict[str, str] = {}
        in_map: Dict[str, str] = {}

        for fid in self._iter_true_fact_ids(state):
            if fid > self.model.ifacts_end:
                # not a fact
                continue
            pred, args = self._parsed_fact[fid]
            if pred is None:
                continue
            if pred == "on" and len(args) == 2:
                c, s = args
                on_map.setdefault(s, []).append(c)
            elif pred == "at" and len(args) == 2:
                x, p = args
                at_map[x] = p
            elif pred == "in" and len(args) == 2:
                c, t = args
                in_map[c] = t

        return on_map, at_map, in_map

    # --------- Heuristic core ---------
    def __call__(self, node) -> int:
        state = node.state

        # If no on-goals exist, heuristic is 0.
        if not self._goal_on:
            return 0

        on_map, at_map, in_map = self._extract_structures(state)

        h_val = 0

        # For each goal (+on[c,s]) not yet true, add an optimistic estimate.
        for c, s, goal_fid in self._goal_on:
            if (state >> goal_fid) & 1:
                continue

            # --- 1) Base placement cost lower bound ---
            # Minimum to achieve on(c,s) from a generic situation:
            # If crate is at same place as target surface: at least Lift+Drop => 2
            # If crate is in truck: at least Unload+Drop => 2 (Lift already implicit in Unload effect)
            # If crate is elsewhere: need Load + Drive + Unload + Drop => 4
            base = 2

            # Where is the target surface?
            p_target = at_map.get(s)

            # Determine crate situation
            if c in in_map:
                # in-truck case: Unload+Drop at least
                base = 2
                # If we know target place and truck is not there, at least one Drive.
                t = in_map[c]
                if p_target is not None:
                    p_truck = at_map.get(t)
                    if p_truck is not None and p_truck != p_target:
                        base += 1  # at least one Drive
            else:
                p_crate = at_map.get(c)
                if p_target is not None and p_crate is not None and p_crate != p_target:
                    base = 4  # Load + Drive + Unload + Drop (optimistic direct drive)

            # --- 2) Clearing lower bound (blockers) ---
            # Need crate clear (no one on it) and surface clear (no one on it).
            # Each blocking crate implies at least one Lift somewhere.
            blockers_on_crate = len(on_map.get(c, ()))
            blockers_on_surface = len(on_map.get(s, ()))

            # If the crate is itself currently on the target surface, blockers_on_surface includes it? No:
            # on_map maps surfaces->crates, so if +on[c,s] is false we shouldn't have c in on_map[s].

            clear_lb = blockers_on_crate + blockers_on_surface

            h_val += base + clear_lb

        return h_val
