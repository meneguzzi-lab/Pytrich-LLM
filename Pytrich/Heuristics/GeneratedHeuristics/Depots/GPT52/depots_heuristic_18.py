from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic18(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale (domain insights -> heuristic design)
    ---------------------------------------------
    The training instances (and typical Depots benchmarks) are dominated by many tasks of the form:
        (do_put_on crateX surfaceY)
    where surfaceY is either a pallet or another crate (stacks). In the provided HDDL, achieving
    on(c, s2) can be done in four main ways (methods m0..m3), with increasing cost/complexity:

      m0: already on(c, s2) -> cost 0
      m1: crate at place p, clear c and clear s2, lift+drop -> local rearrangement
      m2: crate in truck, bring truck to place p, clear s2, unload+drop -> local after transport-in-truck
      m3: general transport: load at p1, drive, unload at p2 -> expensive

    The main *real* difficulty is not driving itself (drive is always possible if truck exists),
    but clearing: if c is buried under other crates, or s2 is not clear (has something on it), we
    must remove blocking crates. In this domain, “removal” typically means: lift top crate and load
    it into some truck (via do_clear method m5), which itself may require clearing that crate first.

    Heuristic idea
    --------------
    We compute a fast *lower bound* on remaining primitive actions combining:
      (A) A goal/on-fact satisfaction count: each unsatisfied goal on(c,s) needs at least one Drop.
      (B) A cheap under/above structural estimate:
          - If c is not clear, at least one Lift is needed to remove the crate directly on top of c.
          - If s2 is not clear, at least one Lift is needed to remove the crate on top of s2.
        This is a lower bound (often more lifts needed, but never fewer).
      (C) A minimal movement/handling estimate based on current location state:
          - If c is lifting already -> closer
          - If c is at place and target surface is at same place -> no drive needed in relaxation
          - If places differ -> at least 1 Drive (in reality might need more, but Drive is a single-step
            operator with no road graph, so 1 is a reasonable minimal bound).
          - If c is in a truck and target surface is at some place -> may need at least 1 Drive if truck
            not already at that place.

    Additionally, we use the *current task network* as the primary set of pending objectives.
    When the task network contains abstract tasks do_put_on(c,s2), we treat them as goals even if
    the classical goal set is empty or already achieved (important for HTN search guidance).

    Implementation constraints
    --------------------------
    - Must not call state_explicit_repr(state) for the current state (too slow).
    - Use bitwise checks with fact.global_id.
    - Goals are bitwise in model.goals; we only use state_explicit_repr(model.goals) once at init.

    The heuristic is not guaranteed admissible in every corner case if summed across many tasks that
    share substructure; however each component is intended as a *lower bound per pending objective*
    and is tuned to reduce node expansions on large stacking instances.
    """

    def __init__(self, model, depots_gpt52_heuristic_18: str = "depots_gpt52_heuristic_18"):
        super().__init__(model, depots_gpt52_heuristic_18)
        self.model = model

        # --- Fact indexing ---
        # Maps for quick lookup without scanning explicit representations of *states*.
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Convenience caches by predicate
        self._on_pairs: List[Tuple[str, str, int]] = []   # (c, s, gid)
        self._at_pairs: List[Tuple[str, str, int]] = []   # (x, p, gid)
        self._in_pairs: List[Tuple[str, str, int]] = []   # (c, t, gid)
        self._clear_unary: List[Tuple[str, int]] = []     # (x, gid)
        self._lifting_pairs: List[Tuple[str, str, int]] = []  # (h, c, gid)

        # Build indices
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_by_key[key] = f.global_id
            if pred == "on" and len(args) == 2:
                self._on_pairs.append((args[0], args[1], f.global_id))
            elif pred == "at" and len(args) == 2:
                self._at_pairs.append((args[0], args[1], f.global_id))
            elif pred == "in" and len(args) == 2:
                self._in_pairs.append((args[0], args[1], f.global_id))
            elif pred == "clear" and len(args) == 1:
                self._clear_unary.append((args[0], f.global_id))
            elif pred == "lifting" and len(args) == 2:
                self._lifting_pairs.append((args[0], args[1], f.global_id))

        # Precompute goal on-facts from the classical goal bitset (cheap one-time).
        self._goal_on: List[Tuple[str, str, int]] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2 and name.startswith("+"):
                gid = self._fact_by_key.get(("on", (args[0], args[1])))
                if gid is not None:
                    self._goal_on.append((args[0], args[1], gid))

        # Small weights tuned for guidance (keep integer).
        self.w_drop = 2     # at least drop-ish work for an unsatisfied on
        self.w_clear = 1    # at least one lift to clear a blocker
        self.w_drive = 1    # at least one drive if different places
        self.w_loadcycle = 1  # minimal handling if must use truck pipeline

    # ---------- Parsing helpers ----------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            sign = name[0]
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true_gid(self, state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    # ---------- State query helpers (bitwise, no explicit repr) ----------
    def _get_place_of(self, state: int, obj: str) -> Optional[str]:
        # Find a place p with at(obj,p)
        for x, p, gid in self._at_pairs:
            if x == obj and self._is_true_gid(state, gid):
                return p
        return None

    def _get_truck_containing(self, state: int, crate: str) -> Optional[str]:
        for c, t, gid in self._in_pairs:
            if c == crate and self._is_true_gid(state, gid):
                return t
        return None

    def _is_clear(self, state: int, surf: str) -> bool:
        gid = self._fact_by_key.get(("clear", (surf,)))
        if gid is None:
            return False
        return self._is_true_gid(state, gid)

    def _is_on(self, state: int, c: str, s: str) -> bool:
        gid = self._fact_by_key.get(("on", (c, s)))
        if gid is None:
            return False
        return self._is_true_gid(state, gid)

    def _is_lifting(self, state: int, crate: str) -> bool:
        for h, c, gid in self._lifting_pairs:
            if c == crate and self._is_true_gid(state, gid):
                return True
        return False

    def _count_immediate_blockers(self, state: int, surface: str) -> int:
        # Lower bound: if something is on top of surface, we need at least one lift.
        for c, s, gid in self._on_pairs:
            if s == surface and self._is_true_gid(state, gid):
                return 1
        return 0

    # ---------- Task network inspection ----------
    def _extract_pending_put_on_tasks(self, task_network) -> List[Tuple[str, str]]:
        pending: List[Tuple[str, str]] = []
        for t in task_network:
            # In Pytrich, task nodes are Operators or AbstractTasks.
            # We only care about abstract task do_put_on.
            name = getattr(t, "name", "")
            if not name:
                continue
            # Abstract tasks usually look like "do_put_on crateX palletY" or similar.
            # Be robust: split by whitespace.
            parts = name.split()
            if len(parts) >= 3 and parts[0] == "do_put_on":
                pending.append((parts[1], parts[2]))
        return pending

    # ---------- Heuristic core ----------
    def __call__(self, node) -> int:
        state = node.state
        task_network = getattr(node, "task_network", []) or []

        # Primary objectives come from pending HTN tasks; fallback to classical goals.
        pending_tasks = self._extract_pending_put_on_tasks(task_network)
        goals_on = pending_tasks if pending_tasks else [(c, s) for (c, s, _) in self._goal_on]

        h_val = 0

        # For each required on(c,s2), estimate minimal remaining effort.
        for c, s2 in goals_on:
            if self._is_on(state, c, s2):
                continue

            # (1) Need at least some placement action (Drop) in any non-trivial solution.
            h_val += self.w_drop

            # (2) Clearing lower bounds: if c not clear, must remove one crate above it.
            if not self._is_clear(state, c):
                h_val += self.w_clear * self._count_immediate_blockers(state, c)

            # If target surface not clear, must remove one crate above it.
            if not self._is_clear(state, s2):
                h_val += self.w_clear * self._count_immediate_blockers(state, s2)

            # (3) Location / transport lower bound.
            # If already lifting the crate, skip transport component.
            if self._is_lifting(state, c):
                continue

            c_place = self._get_place_of(state, c)
            s2_place = self._get_place_of(state, s2)
            c_truck = self._get_truck_containing(state, c)

            if c_truck is not None:
                # Crate in truck: might need a drive to where s2 is.
                if s2_place is not None:
                    t_place = self._get_place_of(state, c_truck)
                    if t_place is not None and t_place != s2_place:
                        h_val += self.w_drive
                # Also at least an unload/drop handling cycle exists in reality.
                h_val += self.w_loadcycle
            else:
                # Crate is at some place; if target surface exists at a different place, need a drive.
                if c_place is not None and s2_place is not None and c_place != s2_place:
                    h_val += self.w_drive
                    # If different place, almost surely requires load/unload cycle.
                    h_val += self.w_loadcycle

        # Small tie-breaker: prefer nodes with shorter remaining task networks.
        # (Doesn't affect correctness; just ordering.)
        h_val += min(5, len(task_network)) // 2

        return int(h_val)
