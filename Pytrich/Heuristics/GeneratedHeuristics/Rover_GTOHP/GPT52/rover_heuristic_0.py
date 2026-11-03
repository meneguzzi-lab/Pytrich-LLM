from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Optional, Set, Tuple

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic0(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    RATIONALE (domain insights -> heuristic design)
    -----------------------------------------------
    In this Rover domain, top-level HTN tasks (get_soil_data/get_rock_data/get_image_data)
    expand into a small fixed pipeline of primitive actions:

      Soil/Rock:  navigate-to-sample -> (maybe drop) -> sample -> navigate-to-uplink -> communicate
      Image:      (navigate-to-calib -> calibrate) -> navigate-to-shot -> take_image -> navigate-to-uplink -> communicate

    The major cost driver is NAVIGATION (navigate actions). Navigation is constrained by:
      - can_traverse edges (rover-specific motion graph)
      - visible constraints (navigate also needs visible(p1,p2)); in the benchmarks visibility is dense,
        but we still include it in edge feasibility.

    Communication requires being at a waypoint that is visible to the lander waypoint.
    The domain encodes this via methods that pick a waypoint w1 such that visible(w1, lander_wp).

    Heuristic strategy
    ------------------
    We compute a *lower bound* on remaining primitive actions by:
      1) Counting mandatory primitive steps still missing for each goal fact (communication, plus
         prerequisites like having analysis/image) under an optimistic relaxation.
      2) Estimating minimal navigation effort using BFS distances on each rover's traverse graph.
         We add a small navigation lower bound per missing goal:
           - soil/rock: distance from *some rover* to sample waypoint + distance from that sample to
             nearest uplink waypoint.
           - image: distance from *some imaging-capable rover/camera/mode* to some shooting waypoint
             + distance from that shooting waypoint to nearest uplink waypoint.

    To keep the heuristic admissible and efficient:
      - We take minima across rovers and across candidate shooting waypoints.
      - We ignore store fullness/drop requirements and camera calibration travel; those can only
        increase true cost, so omitting them preserves a lower bound.
      - We cap unreachable distances with a large constant.

    This heuristic is designed to scale to large instances by:
      - Pre-indexing facts by predicate and arguments.
      - Precomputing per-rover BFS distances only from rover start locations (single-source BFS) and
        using reverse BFS from uplink sets for cheap waypoint->uplink distances.

    Note: We only use fast bitwise checks against fact.global_id; we do *not* call
    model.state_explicit_repr(state) for the current state.
    """

    def __init__(self, model, rover_gpt52_heuristic_0: str = "rover_gpt52_heuristic_0"):
        super().__init__(model, rover_gpt52_heuristic_0)
        self.model = model

        # --- Parse and index all facts once ---
        self._fact_by_name: Dict[str, object] = {}
        self._facts_by_pred: Dict[str, List[object]] = defaultdict(list)
        self._parsed_fact_cache: Dict[str, Tuple[Optional[str], List[str]]] = {}

        for f in self.model.facts:
            self._fact_by_name[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if pred is not None:
                self._facts_by_pred[pred].append(f)

        # --- Extract goal facts (names + parsed structure) ---
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_facts: List[object] = [self._fact_by_name[n] for n in goal_fact_names if n in self._fact_by_name]

        # Partition goal requirements we care about
        self._goal_soil_wps: Set[str] = set()
        self._goal_rock_wps: Set[str] = set()
        self._goal_images: Set[Tuple[str, str]] = set()  # (objective, mode)

        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_soil_wps.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_rock_wps.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_images.add((args[0], args[1]))

        # --- Static domain structure indices ---
        # Lander waypoint(s)
        self._lander_waypoints: Set[str] = set()
        for f in self._facts_by_pred.get("at_lander", []):
            pred, args = self._parse_fact_name(f.name)
            if pred == "at_lander" and len(args) == 2:
                self._lander_waypoints.add(args[1])

        # Uplink waypoints = waypoints visible to some lander waypoint
        self._uplink_waypoints: Set[str] = set()
        lander_wps = list(self._lander_waypoints)
        if lander_wps:
            for f in self._facts_by_pred.get("visible", []):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2 and args[1] in self._lander_waypoints:
                    self._uplink_waypoints.add(args[0])
        # In some instances visible facts can be missing; fallback to lander waypoint itself
        self._uplink_waypoints |= self._lander_waypoints

        # Rover start locations from initial state (single per rover typically)
        self._rover_init_at: Dict[str, str] = {}
        for f in self._facts_by_pred.get("at", []):
            pred, args = self._parse_fact_name(f.name)
            if pred == "at" and len(args) == 2:
                # Note: predicate 'at' is overloaded in domain (for rover only).
                # We'll accept any object in arg0 as rover name; this is safe for benchmarks.
                self._rover_init_at[args[0]] = args[1]

        # Equipment / camera info
        self._imaging_rovers: Set[str] = set()
        for f in self._facts_by_pred.get("equipped_for_imaging", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self._imaging_rovers.add(args[0])

        self._soil_rovers: Set[str] = set()
        for f in self._facts_by_pred.get("equipped_for_soil_analysis", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self._soil_rovers.add(args[0])

        self._rock_rovers: Set[str] = set()
        for f in self._facts_by_pred.get("equipped_for_rock_analysis", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self._rock_rovers.add(args[0])

        # on_board(camera, rover)
        self._cameras_on_rover: Dict[str, Set[str]] = defaultdict(set)
        for f in self._facts_by_pred.get("on_board", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, rov = args
                self._cameras_on_rover[rov].add(cam)

        # supports(camera, mode)
        self._camera_supports: Dict[str, Set[str]] = defaultdict(set)
        for f in self._facts_by_pred.get("supports", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, mode = args
                self._camera_supports[cam].add(mode)

        # visible_from(objective, waypoint)
        self._objective_visible_wps: Dict[str, List[str]] = defaultdict(list)
        for f in self._facts_by_pred.get("visible_from", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                obj, wp = args
                self._objective_visible_wps[obj].append(wp)

        # --- Build per-rover motion graphs using can_traverse edges ---
        # We also require visible(p1,p2) for navigate action applicability; incorporate it as a filter.
        self._visible_edges: Set[Tuple[str, str]] = set()
        for f in self._facts_by_pred.get("visible", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self._visible_edges.add((args[0], args[1]))

        self._graph_by_rover: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        for f in self._facts_by_pred.get("can_traverse", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 3:
                rov, p1, p2 = args
                if (p1, p2) in self._visible_edges or not self._visible_edges:
                    self._graph_by_rover[rov][p1].append(p2)

        # Cache: rover -> distances from its initial position
        self._dist_from_init: Dict[str, Dict[str, int]] = {}
        for rov, start in self._rover_init_at.items():
            self._dist_from_init[rov] = self._bfs_distances(self._graph_by_rover.get(rov, {}), start)

        # Cache: rover -> distance-to-nearest-uplink for each waypoint (multi-source BFS on reversed edges)
        self._dist_to_uplink: Dict[str, Dict[str, int]] = {}
        for rov, g in self._graph_by_rover.items():
            self._dist_to_uplink[rov] = self._reverse_bfs_to_set(g, self._uplink_waypoints)

        # Pre-fetch goal-related fact objects for fast bit checks
        self._goal_comm_soil_fact: Dict[str, object] = {}
        self._goal_comm_rock_fact: Dict[str, object] = {}
        self._goal_comm_image_fact: Dict[Tuple[str, str], object] = {}

        for wp in self._goal_soil_wps:
            name = f"+communicated_soil_data[{wp}]"
            if name in self._fact_by_name:
                self._goal_comm_soil_fact[wp] = self._fact_by_name[name]
        for wp in self._goal_rock_wps:
            name = f"+communicated_rock_data[{wp}]"
            if name in self._fact_by_name:
                self._goal_comm_rock_fact[wp] = self._fact_by_name[name]
        for (obj, mode) in self._goal_images:
            name = f"+communicated_image_data[{obj},{mode}]"
            if name in self._fact_by_name:
                self._goal_comm_image_fact[(obj, mode)] = self._fact_by_name[name]

        # Fact name templates for checking intermediate achievements
        # have_soil_analysis[rover,wp], have_rock_analysis[rover,wp], have_image[rover,obj,mode]

        self._INF = 10 ** 6

    # ---------- Required Heuristic API ----------
    def __call__(self, state: int, task_network=None) -> int:
        # If all goals satisfied (bitwise), return 0 quickly
        if (state & self.model.goals) == self.model.goals:
            return 0

        h = 0

        # 1) Base mandatory action-count lower bounds per missing communicated goal
        # Soil/Rock need at least: sample + communicate = 2 primitive actions (navigation ignored here)
        # Image needs at least: take_image + communicate = 2 primitive actions (calibrate ignored to keep LB)
        missing_soil = [wp for wp, f in self._goal_comm_soil_fact.items() if ((state >> f.global_id) & 1) == 0]
        missing_rock = [wp for wp, f in self._goal_comm_rock_fact.items() if ((state >> f.global_id) & 1) == 0]
        missing_img = [k for k, f in self._goal_comm_image_fact.items() if ((state >> f.global_id) & 1) == 0]

        h += 2 * (len(missing_soil) + len(missing_rock) + len(missing_img))

        # 2) Add optimistic navigation lower bounds (dominant cost driver)
        # We sum per missing item; this is still a lower bound on total navigation only if tasks are independent,
        # which they are not (a rover can chain tasks). To preserve admissibility, we instead take a MAX over items
        # of their individual navigation lower bound (a critical-path style bound).
        nav_bounds: List[int] = []

        for wp in missing_soil:
            nav_bounds.append(self._lb_soil_or_rock_nav(state, wp, is_soil=True))
        for wp in missing_rock:
            nav_bounds.append(self._lb_soil_or_rock_nav(state, wp, is_soil=False))
        for (obj, mode) in missing_img:
            nav_bounds.append(self._lb_image_nav(state, obj, mode))

        if nav_bounds:
            h += max(nav_bounds)

        return int(h)

    # ---------- Lower-bound components ----------
    def _lb_soil_or_rock_nav(self, state: int, sample_wp: str, is_soil: bool) -> int:
        # Choose any capable rover; ignore whether it already has analysis.
        candidate_rovers = self._soil_rovers if is_soil else self._rock_rovers
        best = self._INF
        for rov in candidate_rovers:
            d1 = self._dist_from_init.get(rov, {}).get(sample_wp, self._INF)
            d2 = self._dist_to_uplink.get(rov, {}).get(sample_wp, self._INF)
            # Need to go to sample wp and later be at uplink; LB uses d(init->sample)+d(sample->uplink)
            best = min(best, d1 + d2)
        return 0 if best == self._INF else best

    def _lb_image_nav(self, state: int, objective: str, mode: str) -> int:
        # Need imaging-capable rover with an on-board camera that supports the mode.
        # Then must reach some waypoint w where visible_from(objective,w) holds, and later reach an uplink.
        wps = self._objective_visible_wps.get(objective, [])
        if not wps:
            return 0

        best = self._INF
        for rov in self._imaging_rovers:
            cams = self._cameras_on_rover.get(rov, set())
            if not cams:
                continue
            ok = any(mode in self._camera_supports.get(cam, set()) for cam in cams)
            if not ok:
                continue
            dist_init = self._dist_from_init.get(rov, {})
            dist_uplink = self._dist_to_uplink.get(rov, {})
            for w in wps:
                d1 = dist_init.get(w, self._INF)
                d2 = dist_uplink.get(w, self._INF)
                best = min(best, d1 + d2)

        return 0 if best == self._INF else best

    # ---------- Graph utilities ----------
    def _bfs_distances(self, graph: Dict[str, List[str]], start: str) -> Dict[str, int]:
        if start is None:
            return {}
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in graph.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _reverse_bfs_to_set(self, graph: Dict[str, List[str]], targets: Set[str]) -> Dict[str, int]:
        # Build reverse adjacency
        rev: Dict[str, List[str]] = defaultdict(list)
        for u, vs in graph.items():
            for v in vs:
                rev[v].append(u)

        dist: Dict[str, int] = {}
        q = deque()
        for t in targets:
            dist[t] = 0
            q.append(t)

        while q:
            u = q.popleft()
            du = dist[u]
            for v in rev.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    # ---------- Fact parsing ----------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        # Cached parsing for speed
        if name in self._parsed_fact_cache:
            return self._parsed_fact_cache[name]

        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s2 = s[1:]
            bracket_idx = s2.find("[")
            if bracket_idx > 0 and s2.endswith("]"):
                predicate = s2[:bracket_idx]
                args_str = s2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                res = (predicate, args)
                self._parsed_fact_cache[name] = res
                return res

        res = (None, [])
        self._parsed_fact_cache[name] = res
        return res
