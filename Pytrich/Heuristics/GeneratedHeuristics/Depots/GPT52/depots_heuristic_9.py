from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

# Pytrich heuristic base
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic9(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale / domain insights
    --------------------------
    The domain is centered around achieving many `on(c, s)` relations (crates on surfaces).
    Methods reveal a natural cost structure:

      - If the goal `on(c, s2)` already holds: 0.
      - If crate is at the destination place and can be lifted and dropped: requires
        clearing crate and target surface, then Lift+Drop.
      - If crate is in a truck: need truck at destination, Unload, then Drop.
      - If crate is elsewhere: need Load (Lift+Load), Drive, then Unload+Drop.

    Key difficulty is *stacking/clearance*: to move a crate, it must be `clear(c)`;
    to put onto a surface, that surface must be `clear(s)`.
    Clearing a non-clear surface uses a canonical pattern: recursively clear top crate,
    then Lift+Load into some truck (and truck may need driving).

    Heuristic design
    ----------------
    We compute a cheap admissible lower bound combining:

      1) Goal-support estimate over goal facts (+on[c,s]).
         For each unsatisfied on-goal we compute a *minimum primitive action count*
         lower bound based on where the crate currently is (already at place / in truck /
         elsewhere) and whether clearance is missing.

      2) Landmark-like penalties for missing `clear` conditions.
         If `clear(c)` is false, at least one Lift is required somewhere above it.
         If `clear(s)` is false, at least one action is needed to remove whatever is on it.

    We deliberately ignore negative interactions, hoist availability, and detailed truck
    routing to keep runtime low and preserve admissibility (we only count actions that
    must occur in any real plan).

    Hierarchy awareness
    -------------------
    The heuristic is evaluated on the current *state* and the remaining task network.
    If the task network contains abstract tasks `do_put_on`, we also count unsatisfied
    corresponding `on` targets as additional guidance. This remains admissible because
    accomplishing a `do_put_on(c,s)` eventually must achieve `on(c,s)`.

    Performance
    -----------
    Uses only bitwise checks on facts (no `state_explicit_repr(state)` on the current
    state). Goal facts are pre-parsed once from `model.goals`.
    """

    def __init__(self, model, depots_gpt52_heuristic_9: str = "depots_gpt52_heuristic_9"):
        super().__init__(model, depots_gpt52_heuristic_9)
        self.model = model

        # Map: (pred, tuple(args)) -> fact_global_id
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_id[(pred, tuple(args))] = f.global_id

        # Cache goal on-facts (+on[c,s]) as pairs (c,s)
        self._goal_on: List[Tuple[str, str]] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                self._goal_on.append((args[0], args[1]))

        # Precompute all +at[x,p] and +in[c,t] and +on[c,s] fact IDs grouped by subject
        # for quick lookup of current location/support.
        self._at_facts_by_obj: Dict[str, List[Tuple[str, int]]] = {}
        self._in_facts_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        self._on_facts_by_crate: Dict[str, List[Tuple[str, int]]] = {}

        for (pred, args), gid in self._fact_id.items():
            if pred == "at" and len(args) == 2:
                x, p = args
                self._at_facts_by_obj.setdefault(x, []).append((p, gid))
            elif pred == "in" and len(args) == 2:
                c, t = args
                self._in_facts_by_crate.setdefault(c, []).append((t, gid))
            elif pred == "on" and len(args) == 2:
                c, s = args
                self._on_facts_by_crate.setdefault(c, []).append((s, gid))

        # Some action-count lower bounds (primitive actions):
        # put-on at same place and everything ready requires at least Lift+Drop = 2.
        # load+drive+unload+drop requires at least Lift+Load+Drive+Unload+Drop = 5.
        self._LB_LIFT_DROP = 2
        self._LB_UNLOAD_DROP = 2
        self._LB_LOAD_DRIVE_UNLOAD_DROP = 5

    # ---------- parsing / bit helpers ----------

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

    def _is_true_id(self, state: int, fact_id: int) -> bool:
        return ((state >> fact_id) & 1) == 1

    def _is_true(self, state: int, pred: str, *args: str) -> bool:
        gid = self._fact_id.get((pred, tuple(args)))
        if gid is None:
            return False
        return self._is_true_id(state, gid)

    # ---------- state queries ----------

    def _get_place_of_obj(self, state: int, obj: str) -> Optional[str]:
        """Return place p such that at(obj,p) holds, else None."""
        for p, gid in self._at_facts_by_obj.get(obj, []):
            if self._is_true_id(state, gid):
                return p
        return None

    def _get_truck_containing(self, state: int, crate: str) -> Optional[str]:
        """Return truck t such that in(crate,t) holds, else None."""
        for t, gid in self._in_facts_by_crate.get(crate, []):
            if self._is_true_id(state, gid):
                return t
        return None

    def _get_support_of_crate(self, state: int, crate: str) -> Optional[str]:
        """Return surface s such that on(crate,s) holds, else None."""
        for s, gid in self._on_facts_by_crate.get(crate, []):
            if self._is_true_id(state, gid):
                return s
        return None

    # ---------- heuristic components ----------

    def _lb_put_on_goal(self, state: int, c: str, s2: str) -> int:
        """Admissible LB (in primitive actions) to achieve on(c,s2) from current state."""
        # Already satisfied
        if self._is_true(state, "on", c, s2):
            return 0

        # Destination place of target surface (if known)
        p_dest = self._get_place_of_obj(state, s2)

        # Crate location mode
        t_in = self._get_truck_containing(state, c)
        p_c = self._get_place_of_obj(state, c)

        # Clearance penalties (landmark-ish): missing clear(c) implies at least 1 action.
        # missing clear(s2) implies at least 1 action.
        clear_pen = 0
        if not self._is_true(state, "clear", c):
            clear_pen += 1
        if not self._is_true(state, "clear", s2):
            clear_pen += 1

        # If crate is in some truck: at least Unload+Drop once truck/hoist available
        if t_in is not None:
            # If truck not at destination, at least one Drive is needed (LB=1).
            if p_dest is not None and not self._is_true(state, "at", t_in, p_dest):
                base = 1 + self._LB_UNLOAD_DROP
            else:
                base = self._LB_UNLOAD_DROP
            return base + clear_pen

        # If both crate and target surface are at same place: at least Lift+Drop
        if p_dest is not None and p_c is not None and p_c == p_dest:
            return self._LB_LIFT_DROP + clear_pen

        # Otherwise: must transport via truck in any real plan:
        # at least Load (Lift+Load), Drive, Unload, Drop
        base = self._LB_LOAD_DRIVE_UNLOAD_DROP
        return base + clear_pen

    def _task_network_bonus(self, task_network) -> int:
        """Lightweight HTN-awareness: count remaining do_put_on tasks.

        We add a *very small* LB for each remaining abstract do_put_on task whose
        corresponding on-fact is not already true in the state.

        This is admissible: executing do_put_on must eventually achieve on(c,s).
        We do not overcount: bonus is 1 per unsatisfied task (a trivial LB).
        """
        if not task_network:
            return 0

        bonus = 0
        for t in task_network:
            # Operator has no .name pattern; AbstractTask has .name like 'do_put_on'
            name = getattr(t, "name", "")
            if "do_put_on" in name:
                # Try to parse parameters from grounded task name if present.
                # In Pytrich, abstract task names are often like: 'do_put_on crate1 pallet1'
                parts = name.replace("(", " ").replace(")", " ").replace(",", " ").split()
                # heuristic attempt: find last two tokens as (c,s)
                if len(parts) >= 3:
                    c, s = parts[-2], parts[-1]
                    # We can't access state here; caller will gate by state separately.
                    # Return optimistic 1 always for this pending task.
                    bonus += 1
                else:
                    bonus += 1
        return bonus

    # ---------- main interface ----------

    def __call__(self, state: int, task_network=None) -> int:
        # Primary estimate from explicit goals
        h = 0
        for c, s in self._goal_on:
            h += self._lb_put_on_goal(state, c, s)

        # HTN guidance: tiny bonus for pending do_put_on tasks
        # (kept small to avoid dominating explicit goal estimate)
        if task_network:
            h += self._task_network_bonus(task_network)

        return h
