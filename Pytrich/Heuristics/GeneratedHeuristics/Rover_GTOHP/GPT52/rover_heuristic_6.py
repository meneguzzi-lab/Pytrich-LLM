from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _PredSig:
    """Compact parsed signature for a grounded fact name."""

    pred: str
    args: Tuple[str, ...]


class RoverGPT52Heuristic6(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    Design rationale (domain insights)
    -------------------------------
    The training problems consist of a large ordered HTN over three families of tasks:
      - get_soil_data(wp)  -> navigate -> empty_store -> sample_soil -> send_soil_data
      - get_rock_data(wp)  -> navigate -> empty_store -> sample_rock -> send_rock_data
      - get_image_data(o,m)-> calibrate -> navigate -> take_image -> send_image_data

    The *actual* goals in these benchmarks are the communicated_* facts.
    Communication requires being at some waypoint that is visible to the lander waypoint.
    Navigation cost dominates on large instances, and can involve recursion with visited/unvisit.

    This heuristic combines:
      1) A fast goal-count baseline (unachieved communicated_* goals).
      2) A relaxed, admissible navigation lower bound: shortest-path hops using can_traverse edges
         (ignores visibility and visited bookkeeping, so it is a lower bound).
      3) A lower bound on needing to acquire data before communication:
         - soil/rock: if have_*_analysis is missing, add at least one sample action (+1) and
           distance to reach that waypoint.
         - images: if have_image missing, add at least one take_image (+1) and distance to reach
           some waypoint from which the objective is visible (min over visible_from).
      4) A minimal communication positioning bound: if the communicated goal is missing, add
         distance to reach either the data waypoint itself (direct method possible) or any waypoint
         visible to the lander waypoint (relay method possible). We take min of these.

    To keep admissibility (lower bound):
      - We ignore negative preconditions like visited, ignore visibility constraints during traverse,
        ignore store fullness constraints, ignore calibration details (we never add extra cost for
        calibrate/drop/nop).
      - We aggregate by taking MAX across remaining goals (a critical-path style bound). Summing
        per-goal costs can overcount shared travel, so we avoid that.

    Efficiency
    ----------
    - All state checks use bitwise operations on fact global_id.
    - Static graph distances are precomputed once with BFS from each waypoint (hop-count).
    - Parsing of facts is done once at init to build quick indices.

    This heuristic is intended to strongly reduce node expansions on large instances by giving
    informative (yet cheap) lower bounds driven by navigation structure.
    """

    def __init__(self, model, rover_gpt52_heuristic_6: bool = True):
        super().__init__(model)
        self.rover_gpt52_heuristic_6 = rover_gpt52_heuristic_6

        # --- Parse all facts once and index relevant predicates ---
        self._facts_by_pred: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}
        self._fact_gid_by_sig: Dict[_PredSig, int] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            targs = tuple(args)
            self._facts_by_pred.setdefault(pred, []).append((targs, f.global_id))
            self._fact_gid_by_sig[_PredSig(pred, targs)] = f.global_id

        # Goals are a bitmask; store goal global_ids and structured goal info.
        self._goal_fact_gids: List[int] = [
            self.model.facts[bit].global_id
            for bit in range(self.model.goals.bit_length())
            if (self.model.goals >> bit) & 1
        ]
        self._goal_sigs: List[_PredSig] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(name)
            if pred is not None:
                self._goal_sigs.append(_PredSig(pred, tuple(args)))

        # --- Extract lander waypoint (assume single lander as in benchmarks) ---
        self._lander_wp: Optional[str] = None
        for (args, gid) in self._facts_by_pred.get("at_lander", []):
            # args: (lander, waypoint)
            if len(args) == 2:
                self._lander_wp = args[1]
                break

        # --- Build waypoint set and can_traverse graph (ignoring visibility) ---
        self._waypoints: Set[str] = set()
        for (args, gid) in self._facts_by_pred.get("can_traverse", []):
            if len(args) == 3:
                _, w1, w2 = args
                self._waypoints.add(w1)
                self._waypoints.add(w2)

        self._adj: Dict[str, List[str]] = {w: [] for w in self._waypoints}
        for (args, gid) in self._facts_by_pred.get("can_traverse", []):
            if len(args) == 3:
                _, w1, w2 = args
                if w1 in self._adj:
                    self._adj[w1].append(w2)

        # --- Precompute all-pairs hop distances with BFS per waypoint ---
        self._dist: Dict[str, Dict[str, int]] = {}
        for w in self._waypoints:
            self._dist[w] = self._bfs_dist(w)

        # --- Visible to lander set (waypoints that can communicate) ---
        self._comm_wps: Set[str] = set()
        if self._lander_wp is not None:
            # visible[w1, lander_wp]
            for (args, gid) in self._facts_by_pred.get("visible", []):
                if len(args) == 2 and args[1] == self._lander_wp:
                    self._comm_wps.add(args[0])
            # If none found (edge-case), allow lander waypoint itself
            self._comm_wps.add(self._lander_wp)

        # --- For each objective, collect waypoints where it is visible_from ---
        self._obj_visible_wps: Dict[str, List[str]] = {}
        for (args, gid) in self._facts_by_pred.get("visible_from", []):
            if len(args) == 2:
                obj, wp = args
                self._obj_visible_wps.setdefault(obj, []).append(wp)

    # -------------------- Core heuristic API --------------------

    def __call__(self, node) -> int:
        state = node.state

        # Quick satisfied check
        if (state & self.model.goals) == self.model.goals:
            return 0

        # Determine rover locations present in state (there may be many rovers)
        rover_at: Dict[str, str] = {}
        for (args, gid) in self._facts_by_pred.get("at", []):
            if len(args) == 2:
                rover, wp = args
                if (state >> gid) & 1:
                    rover_at[rover] = wp

        # If we cannot detect any rover positions, fall back to goal-count.
        if not rover_at:
            return self._unsat_goal_count(state)

        base = self._unsat_goal_count(state)

        # Critical-path style bound over unsatisfied communicated goals.
        max_extra = 0
        for gs in self._goal_sigs:
            if not self._is_true_sig(state, gs):
                extra = self._estimate_for_goal(state, gs, rover_at)
                if extra > max_extra:
                    max_extra = extra

        return max(base, max_extra)

    # -------------------- Goal estimation --------------------

    def _estimate_for_goal(self, state: int, gs: _PredSig, rover_at: Dict[str, str]) -> int:
        """Estimate a lower bound for achieving a single goal fact signature.

        Returns a *lower bound* on number of primitive actions, dominated by navigation hop count.
        """

        pred = gs.pred
        args = gs.args

        # communicated_soil_data[wp]
        if pred == "communicated_soil_data" and len(args) == 1:
            wp = args[0]
            return self._lb_communicate_sample(state, kind="soil", wp=wp, rover_at=rover_at)

        # communicated_rock_data[wp]
        if pred == "communicated_rock_data" and len(args) == 1:
            wp = args[0]
            return self._lb_communicate_sample(state, kind="rock", wp=wp, rover_at=rover_at)

        # communicated_image_data[obj, mode]
        if pred == "communicated_image_data" and len(args) == 2:
            obj, mode = args
            return self._lb_communicate_image(state, obj=obj, mode=mode, rover_at=rover_at)

        # Fallback: at least one action
        return 1

    def _lb_communicate_sample(self, state: int, kind: str, wp: str, rover_at: Dict[str, str]) -> int:
        """Lower bound for communicated_soil_data/communicated_rock_data at waypoint wp."""

        have_pred = "have_soil_analysis" if kind == "soil" else "have_rock_analysis"
        comm_pred = "communicated_soil_data" if kind == "soil" else "communicated_rock_data"

        # If already communicated, no cost.
        if self._is_true_sig(state, _PredSig(comm_pred, (wp,))):
            return 0

        # Need to (at least) have the analysis at wp.
        have_any = False
        for (a, gid) in self._facts_by_pred.get(have_pred, []):
            # have_*[rover, wp]
            if len(a) == 2 and a[1] == wp and ((state >> gid) & 1):
                have_any = True
                break

        # Choose best rover to do remaining work.
        best = 10 ** 9
        for rover, rwp in rover_at.items():
            d_to_wp = self._dist_between(rwp, wp)

            # If we don't have analysis yet, at least reach wp and do sample action (1)
            # (ignores empty_store/drop requirements => still admissible)
            acq = 0 if have_any else (d_to_wp + 1)

            # Communication: can be from wp directly (if visible to lander) or from any comm waypoint.
            d_comm = self._lb_to_comm_position(rwp, wp)
            # Need at least one communicate action.
            comm = d_comm + 1

            best = min(best, acq + comm)

        # If distances are unknown (disconnected), fall back to base 1.
        if best >= 10 ** 9:
            return 1
        return best

    def _lb_communicate_image(self, state: int, obj: str, mode: str, rover_at: Dict[str, str]) -> int:
        """Lower bound for communicated_image_data[obj,mode]."""

        comm_sig = _PredSig("communicated_image_data", (obj, mode))
        if self._is_true_sig(state, comm_sig):
            return 0

        # Do we already have the image on any rover?
        have_any = False
        for (a, gid) in self._facts_by_pred.get("have_image", []):
            # have_image[rover, obj, mode]
            if len(a) == 3 and a[1] == obj and a[2] == mode and ((state >> gid) & 1):
                have_any = True
                break

        # Waypoints from which objective is visible.
        vis_wps = self._obj_visible_wps.get(obj, [])

        best = 10 ** 9
        for rover, rwp in rover_at.items():
            # Acquisition: if no image yet, must go to some wp where obj visible and take_image (1)
            if have_any:
                acq = 0
                # Communication from current rover position
                d_comm = self._lb_to_comm_position(rwp, data_wp=None)
                best = min(best, acq + d_comm + 1)
            else:
                # If we don't know any visibility waypoint, cannot estimate well.
                if not vis_wps:
                    # at least one take_image + one communicate
                    best = min(best, 2)
                    continue

                d_to_best_vis = min(self._dist_between(rwp, w) for w in vis_wps)
                acq = d_to_best_vis + 1  # take_image

                # After taking image at that waypoint, can communicate from there (or elsewhere).
                # Lower bound: communicate from some comm waypoint; use the same best vis wp for LB.
                # (min over comm positions from that wp)
                # Note: For admissibility, we can even use rwp instead of vis wp, but using vis is tighter.
                comm_from_vis = min(self._lb_to_comm_position(w, data_wp=None) for w in vis_wps)
                best = min(best, acq + comm_from_vis + 1)

        if best >= 10 ** 9:
            return 1
        return best

    def _lb_to_comm_position(self, from_wp: str, data_wp: Optional[str],) -> int:
        """Lower bound on navigation hops to a waypoint from which communication is possible.

        Two communication methods exist in the domain:
          - direct (at rover at data_wp, visible(data_wp, lander_wp))
          - relay  (at rover at w1, visible(w1, lander_wp))

        For soil/rock, data_wp is known; for images, data_wp is irrelevant (None).

        We ignore visibility between waypoints during navigation and only use can_traverse hops.
        """

        candidates: List[str] = []
        if self._comm_wps:
            candidates.extend(self._comm_wps)

        if data_wp is not None and self._lander_wp is not None:
            # If data_wp itself is a communication position, include it.
            # (In domain, requires visible(data_wp, lander_wp); we under-approx by not requiring it,
            #  but including it can only reduce distance => still admissible for a lower bound.)
            candidates.append(data_wp)

        if not candidates:
            return 0

        return min(self._dist_between(from_wp, c) for c in candidates)

    # -------------------- Utilities --------------------

    def _unsat_goal_count(self, state: int) -> int:
        cnt = 0
        goals_mask = self.model.goals
        # Iterate goal bits only
        for bit in range(goals_mask.bit_length()):
            if (goals_mask >> bit) & 1:
                if ((state >> bit) & 1) == 0:
                    cnt += 1
        return cnt

    def _is_true_sig(self, state: int, sig: _PredSig) -> bool:
        gid = self._fact_gid_by_sig.get(sig)
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _dist_between(self, w1: str, w2: str) -> int:
        if w1 == w2:
            return 0
        dmap = self._dist.get(w1)
        if dmap is None:
            return 10 ** 6
        return dmap.get(w2, 10 ** 6)

    def _bfs_dist(self, start: str) -> Dict[str, int]:
        """Unweighted shortest path hops over can_traverse adjacency."""
        dist = {start: 0}
        q = [start]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            for v in self._adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

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
"
}