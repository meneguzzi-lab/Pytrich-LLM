from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic19(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    RATIONALE / DOMAIN INSIGHTS
    --------------------------
    The training problems (p01..p30) decompose high-level tasks into a small set
    of primitive actions with a very regular structure:

      * get_soil_data(wp):   navigate-to-wp, (empty_store if needed), sample_soil, communicate
      * get_rock_data(wp):   navigate-to-wp, (empty_store if needed), sample_rock, communicate
      * get_image_data(o,m): calibrate (navigate + calibrate), navigate-to imaging waypoint,
                             take_image, communicate

    The true cost is dominated by navigation steps; sampling/calibration/communication are
    typically single actions once positioned.

    This heuristic provides a fast, informative LOWER BOUND by combining:
      (1) Goal/Task landmark counting: each missing communicated_* goal implies at least
          one communication action, and (for soil/rock/image) implies at least one prior
          acquire action (sample/take_image) unless already present.
      (2) Navigation lower bound: for pending objectives we estimate minimum graph distance
          (in traverse graph) from any rover's current waypoint to any waypoint that can
          directly communicate with the lander (a waypoint that is visible to the lander).
          This under-approximates the needed moves because actual plans must also visit
          the sample/imaging waypoint, calibrate, etc.

    IMPORTANT: We keep it admissible by taking MAX of independent lower bounds, not SUM.
    (Summing would double count shared navigation across multiple goals.)

    PERFORMANCE NOTES
    -----------------
    - Uses only bitwise checks against fact global_ids (no explicit state repr on states).
    - Builds traverse distances with BFS per rover from its current location.
    - Pre-indexes facts by predicate for O(1) access to relevant fact ids.

    Interface:
      __init__(self, model, rover_gpt52_heuristic_19: bool = True)
    """

    def __init__(self, model, rover_gpt52_heuristic_19: bool = True):
        super().__init__(model)
        self.enabled = rover_gpt52_heuristic_19

        # --- Parse all facts into predicate -> list of (fact_id, args) ---
        self._fact_args: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
        self._by_pred: Dict[str, List[int]] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            args_t = tuple(args)
            self._fact_args[f.global_id] = (pred, args_t)
            self._by_pred.setdefault(pred, []).append(f.global_id)

        # --- Cache goal facts (bitwise int -> ids) ---
        goal_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_fact_ids: List[int] = []
        self._goal_comm_soil: Set[str] = set()
        self._goal_comm_rock: Set[str] = set()
        self._goal_comm_img: Set[Tuple[str, str]] = set()  # (objective, mode)

        # map name->id (facts are unique)
        name_to_id = {f.name: f.global_id for f in self.model.facts}
        for gn in goal_names:
            fid = name_to_id.get(gn)
            if fid is None:
                continue
            self._goal_fact_ids.append(fid)
            pred, args = self._parse_fact_name(gn)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_comm_soil.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_comm_rock.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_comm_img.add((args[0], args[1]))

        # --- Build static indices ---
        # Rover locations facts: at(rover, waypoint)
        self._at_rover_facts: List[int] = self._by_pred.get("at", [])
        # But 'at' is overloaded (rover at, lander at). We'll filter by arg name prefix at runtime.

        # Lander position facts: at_lander(lander, waypoint)
        self._at_lander_facts: List[int] = self._by_pred.get("at_lander", [])

        # Visibility facts between waypoints: visible(w1,w2)
        self._visible_facts: List[int] = self._by_pred.get("visible", [])

        # Traversal edges: can_traverse(rover, w1, w2)
        self._can_traverse_facts: List[int] = self._by_pred.get("can_traverse", [])

        # For imaging: visible_from(objective, waypoint)
        self._visible_from_facts: List[int] = self._by_pred.get("visible_from", [])

        # Already acquired data facts:
        self._have_soil_facts: List[int] = self._by_pred.get("have_soil_analysis", [])
        self._have_rock_facts: List[int] = self._by_pred.get("have_rock_analysis", [])
        self._have_img_facts: List[int] = self._by_pred.get("have_image", [])

        # --- Precompute static graphs ---
        # Traverse graph is rover-dependent in principle, but in instances it is per rover.
        # We'll build adjacency per rover symbol.
        self._adj_by_rover: Dict[str, Dict[str, List[str]]] = {}
        for fid in self._can_traverse_facts:
            pred, args = self._fact_args[fid]
            # args: (rover, from, to)
            if len(args) != 3:
                continue
            r, w1, w2 = args
            self._adj_by_rover.setdefault(r, {}).setdefault(w1, []).append(w2)

        # Precompute lander waypoint(s) are not static in general, but typically fixed.
        # We'll compute per state because at_lander is a fact.

        # Precompute for each objective the set of waypoints from which it is visible.
        self._obj_to_wps: Dict[str, Set[str]] = {}
        for fid in self._visible_from_facts:
            _, args = self._fact_args[fid]
            if len(args) != 2:
                continue
            obj, wp = args
            self._obj_to_wps.setdefault(obj, set()).add(wp)

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

    @staticmethod
    def _is_true(state: int, fact_id: int) -> bool:
        return ((state >> fact_id) & 1) == 1

    def _current_rover_positions(self, state: int) -> Dict[str, str]:
        """Return rover -> waypoint for rovers whose (at rover wp) is true."""
        pos: Dict[str, str] = {}
        for fid in self._at_rover_facts:
            if not self._is_true(state, fid):
                continue
            pred, args = self._fact_args.get(fid, (None, ()))
            if pred != "at" or len(args) != 2:
                continue
            a0, a1 = args
            # Heuristic filter: rovers are named like rover0, rover1, ...
            if a0.startswith("rover"):
                pos[a0] = a1
        return pos

    def _lander_waypoints(self, state: int) -> Set[str]:
        wps: Set[str] = set()
        for fid in self._at_lander_facts:
            if not self._is_true(state, fid):
                continue
            _, args = self._fact_args[fid]
            if len(args) == 2:
                # (lander, waypoint)
                wps.add(args[1])
        return wps

    def _wps_visible_to_lander(self, state: int, lander_wps: Set[str]) -> Set[str]:
        """Waypoints w such that visible(w, lander_wp) for some lander_wp."""
        if not lander_wps:
            return set()
        vis: Set[str] = set()
        for fid in self._visible_facts:
            if not self._is_true(state, fid):
                continue
            _, args = self._fact_args[fid]
            if len(args) != 2:
                continue
            w1, w2 = args
            if w2 in lander_wps:
                vis.add(w1)
        return vis

    def _bfs_dist(self, rover: str, start: str) -> Dict[str, int]:
        """Unweighted shortest path distances using can_traverse edges for given rover."""
        adj = self._adj_by_rover.get(rover)
        if not adj:
            return {start: 0}
        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _min_dist_to_set(self, dist: Dict[str, int], targets: Set[str]) -> Optional[int]:
        best = None
        for t in targets:
            d = dist.get(t)
            if d is None:
                continue
            if best is None or d < best:
                best = d
        return best

    def _pending_communication_counts(self, state: int) -> Tuple[int, int, int]:
        """Return (pending_soil, pending_rock, pending_img)."""
        pending_soil = 0
        pending_rock = 0
        pending_img = 0

        # communicated_soil_data[wp]
        for wp in self._goal_comm_soil:
            # find corresponding fact id(s) via predicate index: cheap enough in practice because goals list is small
            # We'll do direct scan over goal_fact_ids and parse those.
            pass

        # Instead of searching by name, just check goal facts bitwise using stored goal_fact_ids.
        for fid in self._goal_fact_ids:
            if self._is_true(state, fid):
                continue
            pred, args = self._fact_args.get(fid, (None, ()))
            if pred == "communicated_soil_data" and len(args) == 1:
                pending_soil += 1
            elif pred == "communicated_rock_data" and len(args) == 1:
                pending_rock += 1
            elif pred == "communicated_image_data" and len(args) == 2:
                pending_img += 1

        return pending_soil, pending_rock, pending_img

    def _need_any_data_acquisition(self, state: int) -> bool:
        """Whether at least one pending communicated goal still lacks its underlying 'have_*' fact.

        This supports a simple landmark bound: if something must be communicated but isn't yet
        possessed on any rover, then at least one acquire action (sample/take_image) is needed.
        """
        # Soil
        if self._goal_comm_soil:
            for wp in self._goal_comm_soil:
                # if already communicated, ignore
                # Check communicated_soil_data[wp] fact among goal_fact_ids
                communicated = False
                for fid in self._goal_fact_ids:
                    pred, args = self._fact_args.get(fid, (None, ()))
                    if pred == "communicated_soil_data" and len(args) == 1 and args[0] == wp:
                        communicated = self._is_true(state, fid)
                        break
                if communicated:
                    continue
                # if any have_soil_analysis(rover, wp) is true, acquisition done
                have_it = False
                for hf in self._have_soil_facts:
                    if not self._is_true(state, hf):
                        continue
                    _, hargs = self._fact_args[hf]
                    if len(hargs) == 2 and hargs[1] == wp:
                        have_it = True
                        break
                if not have_it:
                    return True

        # Rock
        if self._goal_comm_rock:
            for wp in self._goal_comm_rock:
                communicated = False
                for fid in self._goal_fact_ids:
                    pred, args = self._fact_args.get(fid, (None, ()))
                    if pred == "communicated_rock_data" and len(args) == 1 and args[0] == wp:
                        communicated = self._is_true(state, fid)
                        break
                if communicated:
                    continue
                have_it = False
                for hf in self._have_rock_facts:
                    if not self._is_true(state, hf):
                        continue
                    _, hargs = self._fact_args[hf]
                    if len(hargs) == 2 and hargs[1] == wp:
                        have_it = True
                        break
                if not have_it:
                    return True

        # Images
        if self._goal_comm_img:
            for (obj, mode) in self._goal_comm_img:
                communicated = False
                for fid in self._goal_fact_ids:
                    pred, args = self._fact_args.get(fid, (None, ()))
                    if pred == "communicated_image_data" and len(args) == 2 and args[0] == obj and args[1] == mode:
                        communicated = self._is_true(state, fid)
                        break
                if communicated:
                    continue
                have_it = False
                for hf in self._have_img_facts:
                    if not self._is_true(state, hf):
                        continue
                    _, hargs = self._fact_args[hf]
                    # have_image(rover, objective, mode)
                    if len(hargs) == 3 and hargs[1] == obj and hargs[2] == mode:
                        have_it = True
                        break
                if not have_it:
                    return True

        return False

    def __call__(self, node) -> int:
        if not self.enabled:
            return 0

        state = node.state

        # If already reached, 0
        if (self.model.goals & state) == self.model.goals:
            return 0

        # (A) Landmark lower bound from pending communicated goals and required acquisition.
        pending_soil, pending_rock, pending_img = self._pending_communication_counts(state)
        pending_total_comm = pending_soil + pending_rock + pending_img

        # At least one action per missing communicated_* fact.
        lb_comm = pending_total_comm

        # If there exists at least one pending communicated goal whose underlying have_* is not yet true,
        # then we need at least one acquisition action (sample_soil/sample_rock/take_image).
        lb_acquire = 1 if self._need_any_data_acquisition(state) else 0

        lb_landmarks = max(lb_comm, lb_acquire)

        # (B) Navigation lower bound: if anything remains to be communicated, you must reach some waypoint
        # that is visible to the lander. Estimate minimal moves for any rover to reach such a waypoint.
        lb_nav = 0
        if pending_total_comm > 0:
            lander_wps = self._lander_waypoints(state)
            vis_to_lander = self._wps_visible_to_lander(state, lander_wps)
            rover_pos = self._current_rover_positions(state)
            best = None
            if vis_to_lander and rover_pos:
                for r, start_wp in rover_pos.items():
                    dist = self._bfs_dist(r, start_wp)
                    d = self._min_dist_to_set(dist, vis_to_lander)
                    if d is None:
                        continue
                    if best is None or d < best:
                        best = d
            lb_nav = best if best is not None else 0

        # Final admissible combination.
        return max(lb_landmarks, lb_nav)
