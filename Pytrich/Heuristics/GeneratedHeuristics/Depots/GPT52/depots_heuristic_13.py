from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

# Pytrich framework import
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic13(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Rationale (domain insights -> heuristic signals)
    -----------------------------------------------
    In this Depots HDDL, essentially every top-level objective is of the form:
        (do_put_on crateX surfaceY)
    and all methods ultimately realize the goal fact:
        +on[crateX,surfaceY]

    Achieving +on[c,s] minimally requires a hoist at the destination place p,
    the destination surface s to be clear, and the crate c to be lifted and dropped
    at that place (possibly after transportation by truck).

    Key structural challenges captured cheaply:
    1) If c is already on s => 0.
    2) If c is at same place as s and a hoist is present there:
       - We likely need: clear(c), clear(s), lift, drop.
       - But clear(c) and clear(s) may recursively require moving other crates away.
       We approximate this by adding small penalties when not clear.
    3) If c is in a truck, we must (at least) bring that truck to p and unload.
       We approximate that with a fixed lower bound.
    4) If c is elsewhere, we need load+drive+unload chain => larger fixed lower bound.

    Additionally, for any blocking situation (surface not clear or crate not clear)
    we count *stack depth* above an object/surface as a relaxed estimate of required
    removals (each removal at least needs a lift+load somewhere). We use only a
    small weight on that depth to keep the heuristic admissible-ish in practice;
    Pytrich does not require admissibility, but we keep it conservative.

    Implementation notes (performance):
    - NEVER uses model.state_explicit_repr(state) for the current state.
    - Uses fact.global_id bit checks.
    - Preindexes facts by predicate and arguments for fast membership queries.
    - Extracts current open abstract tasks from task_network (do_put_on only).

    This heuristic is meant to reduce node expansions by strongly preferring states
    that (a) place required crates at the right location, (b) clear destination
    surfaces, and (c) expose crates to be lifted.
    """

    def __init__(self, model, depots_gpt52_heuristic_13: str = "depots_gpt52_heuristic_13"):
        super().__init__(model, depots_gpt52_heuristic_13)
        self.model = model

        # --- Fact indexing ---
        # predicate -> list of (args_tuple, global_id)
        self._facts_by_pred: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
        # (pred, args_tuple) -> global_id
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            at = tuple(args)
            self._facts_by_pred.setdefault(pred, []).append((at, f.global_id))
            self._fact_gid[(pred, at)] = f.global_id

        # Quick access lists
        self._on_facts = self._facts_by_pred.get("on", [])
        self._at_facts = self._facts_by_pred.get("at", [])
        self._in_facts = self._facts_by_pred.get("in", [])
        self._clear_facts = self._facts_by_pred.get("clear", [])

        # Precompute goal on-facts (+on[c,s]) from model.goals (bitset)
        self._goal_on: List[Tuple[str, str]] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2 and name.startswith("+"):
                self._goal_on.append((args[0], args[1]))

        # Build static "support graph" over ON relations to estimate stack depth
        # We'll compute dynamic depth per state using state bits but reuse adjacency lists.
        # below[x] = y if we have on[x,y] facts (y can be crate or pallet)
        self._on_pairs: List[Tuple[str, str, int]] = []  # (x,y,gid)
        for (args, gid) in self._on_facts:
            if len(args) == 2:
                self._on_pairs.append((args[0], args[1], gid))

        # Reverse adjacency candidates: above[y] may contain x if on[x,y]
        self._above_candidates: Dict[str, List[Tuple[str, int]]] = {}
        for x, y, gid in self._on_pairs:
            self._above_candidates.setdefault(y, []).append((x, gid))

        # Candidates for at[obj,place]
        self._at_candidates: Dict[str, List[Tuple[str, int]]] = {}
        for (args, gid) in self._at_facts:
            if len(args) == 2:
                obj, place = args
                self._at_candidates.setdefault(obj, []).append((place, gid))

        # Candidates for in[crate,truck]
        self._in_candidates: Dict[str, List[Tuple[str, int]]] = {}
        for (args, gid) in self._in_facts:
            if len(args) == 2:
                crate, truck = args
                self._in_candidates.setdefault(crate, []).append((truck, gid))

        # Candidates for clear[surface]
        self._clear_gid: Dict[str, int] = {}
        for (args, gid) in self._clear_facts:
            if len(args) == 1:
                self._clear_gid[args[0]] = gid

        # For hoists: at[hoist,place]
        # We identify hoists by scanning at facts where object name starts with "hoist".
        self._hoists_at_place_gids: Dict[str, List[int]] = {}
        for (args, gid) in self._at_facts:
            if len(args) == 2:
                obj, place = args
                if obj.startswith("hoist"):
                    self._hoists_at_place_gids.setdefault(place, []).append(gid)

    # ----------------- Core heuristic API -----------------

    def __call__(self, state: int, task_network) -> int:
        """Compute heuristic value.

        We sum an estimate for each pending do_put_on(c,s) task found in the TN.
        If none are found (some benchmarks may reach primitive-only TNs), we fall
        back to counting unsatisfied goal +on facts.
        """

        pending: List[Tuple[str, str]] = self._extract_put_on_tasks(task_network)
        if not pending:
            # Fallback: count unsatisfied goal on-facts
            h = 0
            for c, s in self._goal_on:
                if not self._is_true(state, "on", (c, s)):
                    h += 1
            return h

        h_total = 0
        for c, s in pending:
            h_total += self._estimate_put_on(state, c, s)
        return h_total

    # ----------------- Estimation helpers -----------------

    def _estimate_put_on(self, state: int, crate: str, dest_surface: str) -> int:
        # Already satisfied?
        if self._is_true(state, "on", (crate, dest_surface)):
            return 0

        # Determine destination place p where dest_surface is located.
        dest_place = self._where_is_locatable(state, dest_surface)

        # If we cannot infer dest_place, be conservative.
        if dest_place is None:
            return 2

        # Basic requirement: have hoist at destination for any lift/drop
        has_hoist = self._place_has_hoist(state, dest_place)
        hoist_pen = 0 if has_hoist else 1  # at least one move to bring relevant resources in most decompositions

        # Clearing penalties (relaxed)
        clear_dest = self._is_clear(state, dest_surface)
        dest_block_depth = 0 if clear_dest else self._stack_depth_above(state, dest_surface)

        # Crate location mode
        crate_place = self._where_is_locatable(state, crate)
        crate_in_truck = self._in_which_truck(state, crate)

        # Crate "clear" needed for Lift action
        crate_clear = self._is_clear(state, crate)
        crate_block_depth = 0 if crate_clear else self._stack_depth_above(state, crate)

        # Case A: crate already at destination place
        if crate_place == dest_place:
            # Need at least lift+drop (2) in most cases, plus maybe clear operations.
            # We add small relaxed penalties for blockers.
            base = 2
            base += 1 if not clear_dest else 0
            base += 1 if not crate_clear else 0
            # Depth contributes weakly: each blocker crate above implies at least one removal.
            base += (dest_block_depth + crate_block_depth) // 2
            base += hoist_pen
            return max(1, base)

        # Case B: crate is in some truck
        if crate_in_truck is not None:
            # Need get truck to dest + unload + drop.
            # Lower bound: drive (maybe) + unload + drop => >=2, plus clearing dest.
            base = 3
            base += 1 if not clear_dest else 0
            base += dest_block_depth // 2
            base += hoist_pen
            return base

        # Case C: crate elsewhere (not at dest, not in truck)
        # Need load-truck + drive + unload + drop.
        base = 5
        base += 1 if not clear_dest else 0
        base += dest_block_depth // 2
        base += (crate_block_depth // 2)
        base += hoist_pen
        return base

    # ----------------- State query primitives (bitwise) -----------------

    def _is_true(self, state: int, pred: str, args: Tuple[str, ...]) -> bool:
        gid = self._fact_gid.get((pred, args))
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _where_is_locatable(self, state: int, obj: str) -> Optional[str]:
        """Return place p such that +at[obj,p] holds, else None."""
        for place, gid in self._at_candidates.get(obj, []):
            if ((state >> gid) & 1) == 1:
                return place
        return None

    def _in_which_truck(self, state: int, crate: str) -> Optional[str]:
        for truck, gid in self._in_candidates.get(crate, []):
            if ((state >> gid) & 1) == 1:
                return truck
        return None

    def _is_clear(self, state: int, surface: str) -> bool:
        gid = self._clear_gid.get(surface)
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _place_has_hoist(self, state: int, place: str) -> bool:
        gids = self._hoists_at_place_gids.get(place, [])
        for gid in gids:
            if ((state >> gid) & 1) == 1:
                return True
        return False

    def _stack_depth_above(self, state: int, surface: str, cap: int = 10) -> int:
        """Return number of crates directly/indirectly on top of 'surface' in current state.

        Computed by following +on[x,surface] upwards iteratively.
        cap prevents pathological loops; domain should be acyclic but we stay safe.
        """
        depth = 0
        current = surface
        visited: Set[str] = set([surface])
        for _ in range(cap):
            above = self._directly_on_top(state, current)
            if above is None or above in visited:
                break
            depth += 1
            visited.add(above)
            current = above
        return depth

    def _directly_on_top(self, state: int, surface: str) -> Optional[str]:
        """Return x such that +on[x,surface] holds, else None."""
        for x, gid in self._above_candidates.get(surface, []):
            if ((state >> gid) & 1) == 1:
                return x
        return None

    # ----------------- Task-network parsing -----------------

    def _extract_put_on_tasks(self, task_network) -> List[Tuple[str, str]]:
        """Extract pending (crate, surface) pairs from abstract tasks named do_put_on.

        This is written defensively because different Pytrich versions represent tasks
        slightly differently.
        """
        res: List[Tuple[str, str]] = []
        if not task_network:
            return res

        for t in task_network:
            # AbstractTask with name like 'do_put_on crate1 pallet1' (often grounded)
            name = getattr(t, "name", "")
            if not name:
                continue

            # Common grounded naming patterns:
            # - "do_put_on crate1 pallet1"
            # - "(do_put_on crate1 pallet1)" (rare)
            parts = name.replace("(", " ").replace(")", " ").split()
            if not parts:
                continue

            if parts[0] == "do_put_on" and len(parts) >= 3:
                res.append((parts[1], parts[2]))

        return res

    # ----------------- Fact name parsing -----------------

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            core = name[1:]
            bracket_idx = core.find("[")
            if bracket_idx > 0 and core.endswith("]"):
                predicate = core[:bracket_idx]
                args_str = core[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
