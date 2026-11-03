from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic9(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic features)
    ------------------------------------------------------
    Rover problems here are essentially *deliver-information* problems with three pipelines:

      1) Soil data:    navigate -> (empty store if needed) -> sample_soil -> communicate_soil_data
      2) Rock data:    navigate -> (empty store if needed) -> sample_rock -> communicate_rock_data
      3) Image data:   (calibrate camera) -> navigate -> take_image -> communicate_image_data

    Key structural bottleneck in this domain:
      - Communication requires being at a waypoint that is *visible* to the lander waypoint.
      - Sampling requires reaching the sample waypoint.
      - Imaging requires a waypoint from which objective is visible, and a camera that supports the mode.
      - Calibration is consumed by take_image (calibrated becomes false), so each image typically needs
        a calibrate action beforehand (unless already calibrated immediately before the shot).

    Heuristic strategy:
      - We compute a *lower bound* on remaining work by summing, for each unmet goal fact,
        a cheap estimate of the minimal number of primitive actions needed to achieve it.
      - The estimate is *state-aware* using fast bitwise checks, but uses a relaxed model:
          * ignores store fullness interactions except a 0/1 penalty if the only relevant store is full
          * ignores ordering interactions among multiple goals
          * ignores that calibration is consumed (we still count at most one calibrate per image)
      - Navigation cost is approximated via a precomputed shortest-path distance over traversable edges
        for each rover (from can_traverse). Each navigate action moves along one edge, so distance is a
        lower bound on required navigate steps.
      - Communication positioning is approximated by choosing the best (closest) waypoint that can see
        the lander waypoint (using visible graph), for *some* rover that can provide the data.

    This heuristic is not necessarily admissible (due to summation over possibly overlapping actions),
    but is effective for reducing node expansions in practice.

    Implementation notes for Pytrich:
      - Never use state_explicit_repr(state) for checking facts in search nodes.
      - Use (state >> fact.global_id) & 1 for O(1) checks.
      - Goals are stored as a bitwise int in model.goals; we obtain goal fact names via
        model.state_explicit_repr(model.goals) once in __init__.
    """

    def __init__(self, model, rover_gpt52_heuristic_9: str = "rover_gpt52_heuristic_9"):
        super().__init__(model, rover_gpt52_heuristic_9)
        self.model = model

        # --- Fact indexing helpers ---
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # Cache parsed predicate/args for all facts (name -> (pred, tuple(args)))
        self._parsed: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred:
                self._parsed[f.name] = (pred, tuple(args))

        # --- Extract static structures: rovers, stores, lander locations, graph edges ---
        self._rovers: Set[str] = set()
        self._stores_of: Dict[str, str] = {}  # rover -> store
        self._lander_at_fact_ids: List[int] = []  # global_ids of +at_lander[l,w]

        # Visibility graph between waypoints, used by navigate precondition and comm visibility.
        # (We only need visibility between waypoints; domain has visible[p1,p2])
        self._visible_wp: Set[Tuple[str, str]] = set()

        # Traversal graph per rover (can_traverse is rover-specific)
        self._can_traverse_edges: Dict[str, List[Tuple[str, str]]] = {}

        for name, (pred, args) in self._parsed.items():
            if pred == "store_of" and len(args) == 2:
                store, rover = args
                self._stores_of[rover] = store
                self._rovers.add(rover)
            elif pred == "at" and len(args) == 2:
                rover, _wp = args
                # only rovers appear in at[rover,wp] in this domain
                if rover.startswith("rover"):
                    self._rovers.add(rover)
            elif pred == "visible" and len(args) == 2:
                self._visible_wp.add((args[0], args[1]))
            elif pred == "can_traverse" and len(args) == 3:
                r, a, b = args
                self._rovers.add(r)
                self._can_traverse_edges.setdefault(r, []).append((a, b))
            elif pred == "at_lander" and len(args) == 2:
                self._lander_at_fact_ids.append(self._fact_by_name[name].global_id)

        # Precompute waypoint set (appears in any visible/can_traverse/at_lander/etc.)
        self._waypoints: Set[str] = set()
        for (a, b) in self._visible_wp:
            self._waypoints.add(a)
            self._waypoints.add(b)
        for r, edges in self._can_traverse_edges.items():
            for (a, b) in edges:
                self._waypoints.add(a)
                self._waypoints.add(b)

        # --- Precompute all-pairs shortest path distances per rover (BFS on directed edges) ---
        # Distances count navigate steps.
        self._dist: Dict[str, Dict[str, Dict[str, int]]] = {}
        for r in self._rovers:
            self._dist[r] = self._compute_all_pairs_shortest_paths(r)

        # --- Goal facts (names) ---
        self._goal_fact_names: List[str] = self.model.state_explicit_repr(self.model.goals)

        # Partition goals we know how to score
        self._goal_comm_soil: List[str] = []
        self._goal_comm_rock: List[str] = []
        self._goal_comm_img: List[Tuple[str, str]] = []  # (objective, mode)

        for gf in self._goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_comm_soil.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_comm_rock.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_comm_img.append((args[0], args[1]))

        # Precompute imaging capability helpers
        # cameras on rovers, supports, calibration_target
        self._cameras_on: Dict[str, Set[str]] = {}  # rover -> cameras
        self._camera_supports: Dict[str, Set[str]] = {}  # camera -> modes
        self._calib_target: Dict[str, Set[str]] = {}  # camera -> objectives
        self._visible_from: Dict[str, Set[str]] = {}  # objective -> waypoints

        for name, (pred, args) in self._parsed.items():
            if pred == "on_board" and len(args) == 2:
                cam, rover = args
                self._cameras_on.setdefault(rover, set()).add(cam)
            elif pred == "supports" and len(args) == 2:
                cam, mode = args
                self._camera_supports.setdefault(cam, set()).add(mode)
            elif pred == "calibration_target" and len(args) == 2:
                cam, obj = args
                self._calib_target.setdefault(cam, set()).add(obj)
            elif pred == "visible_from" and len(args) == 2:
                obj, wp = args
                self._visible_from.setdefault(obj, set()).add(wp)

        # Precompute for each objective: best calibration waypoint for a (camera,obj)
        # We will compute on the fly via visible_from sets + distance.

    # ------------------------ Core Pytrich heuristic API ------------------------

    def __call__(self, state: int, task_network=None) -> int:
        # If already goal state wrt facts, return 0 (task network may still exist in HTN search,
        # but in Pytrich goal test includes tn empty; heuristic is still useful as fact-based signal)
        h = 0

        # Identify current lander waypoint (assume single lander)
        lander_wp = self._current_lander_waypoint(state)

        # Current rover positions
        rover_pos = self._current_rover_positions(state)

        # Store fullness status
        store_full = self._current_store_fullness(state)

        # For each unmet goal, add an estimate
        # Soil goals
        for wp in self._goal_comm_soil:
            if not self._is_true(state, f"+communicated_soil_data[{wp}]"):
                h += self._estimate_soil_goal(state, wp, lander_wp, rover_pos, store_full)

        # Rock goals
        for wp in self._goal_comm_rock:
            if not self._is_true(state, f"+communicated_rock_data[{wp}]"):
                h += self._estimate_rock_goal(state, wp, lander_wp, rover_pos, store_full)

        # Image goals
        for (obj, mode) in self._goal_comm_img:
            if not self._is_true(state, f"+communicated_image_data[{obj},{mode}]"):
                h += self._estimate_image_goal(state, obj, mode, lander_wp, rover_pos)

        return int(h)

    # ------------------------ Estimators per goal type ------------------------

    def _estimate_soil_goal(
        self,
        state: int,
        sample_wp: str,
        lander_wp: Optional[str],
        rover_pos: Dict[str, str],
        store_full: Dict[str, bool],
    ) -> int:
        """Estimate actions to achieve communicated_soil_data[sample_wp]."""

        # If already communicated: 0
        if self._is_true(state, f"+communicated_soil_data[{sample_wp}]"):
            return 0

        # If we already have analysis for some rover at that waypoint, just need comm.
        best = 10**9
        for r in self._rovers:
            if not self._is_true(state, f"+available[{r}]"):
                # availability is required for navigate+communicate; but in many instances it stays true.
                # Still, for LB we ignore waiting/unavailability.
                pass

            have = self._is_true(state, f"+have_soil_analysis[{r},{sample_wp}]")
            if have:
                comm_cost = self._estimate_comm_cost(r, rover_pos.get(r), lander_wp)
                best = min(best, comm_cost)
            else:
                # Need to go sample then comm.
                if not self._is_true(state, f"+equipped_for_soil_analysis[{r}]"):
                    continue
                # Navigate to sample waypoint
                nav_to_sample = self._dist_steps(r, rover_pos.get(r), sample_wp)
                # Possibly empty store (drop) if full
                drop_cost = 1 if store_full.get(r, False) else 0
                # sample_soil (1)
                sample_cost = 1
                # comm
                # after sampling, rover is at sample_wp
                comm_cost = self._estimate_comm_cost(r, sample_wp, lander_wp)
                best = min(best, nav_to_sample + drop_cost + sample_cost + comm_cost)

        # If no rover feasible, return a large penalty
        if best >= 10**9:
            return 1000
        return best

    def _estimate_rock_goal(
        self,
        state: int,
        sample_wp: str,
        lander_wp: Optional[str],
        rover_pos: Dict[str, str],
        store_full: Dict[str, bool],
    ) -> int:
        """Estimate actions to achieve communicated_rock_data[sample_wp]."""

        if self._is_true(state, f"+communicated_rock_data[{sample_wp}]"):
            return 0

        best = 10**9
        for r in self._rovers:
            have = self._is_true(state, f"+have_rock_analysis[{r},{sample_wp}]")
            if have:
                comm_cost = self._estimate_comm_cost(r, rover_pos.get(r), lander_wp)
                best = min(best, comm_cost)
            else:
                if not self._is_true(state, f"+equipped_for_rock_analysis[{r}]"):
                    continue
                nav_to_sample = self._dist_steps(r, rover_pos.get(r), sample_wp)
                drop_cost = 1 if store_full.get(r, False) else 0
                sample_cost = 1
                comm_cost = self._estimate_comm_cost(r, sample_wp, lander_wp)
                best = min(best, nav_to_sample + drop_cost + sample_cost + comm_cost)

        if best >= 10**9:
            return 1000
        return best

    def _estimate_image_goal(
        self,
        state: int,
        obj: str,
        mode: str,
        lander_wp: Optional[str],
        rover_pos: Dict[str, str],
    ) -> int:
        """Estimate actions to achieve communicated_image_data[obj,mode]."""

        if self._is_true(state, f"+communicated_image_data[{obj},{mode}]"):
            return 0

        best = 10**9

        # If some rover already has the image, just communication remains.
        for r in self._rovers:
            if self._is_true(state, f"+have_image[{r},{obj},{mode}]"):
                best = min(best, self._estimate_comm_cost(r, rover_pos.get(r), lander_wp))

        # Otherwise, need to take it.
        for r in self._rovers:
            if not self._is_true(state, f"+equipped_for_imaging[{r}]"):
                continue

            cams = self._cameras_on.get(r, set())
            if not cams:
                continue

            # Candidate shooting waypoints are visible_from[obj, w]
            shoot_wps = self._visible_from.get(obj, set())
            if not shoot_wps:
                continue

            rpos = rover_pos.get(r)

            for cam in cams:
                if mode not in self._camera_supports.get(cam, set()):
                    continue

                # Calibration requirement: need calibrated[cam,r] or do calibrate.
                # For LB, count 0 if already calibrated, else 1 + navigation to some calibration waypoint.
                already_cal = self._is_true(state, f"+calibrated[{cam},{r}]")

                # Find best (lowest nav) waypoint where objective is visible for calibration_target(cam, o_cal)
                # Domain requires some calibration_target(cam,o_cal) and visible_from(o_cal, wp_cal)
                # For LB: if already calibrated -> 0. Else: nav to any wp where *some* calibration target is visible.
                cal_cost = 0
                cal_end_wp = rpos
                if not already_cal:
                    cal_cost, cal_end_wp = self._estimate_calibration_cost(r, rpos, cam)
                    if cal_cost >= 10**8:
                        continue  # cannot calibrate this camera

                # After (possible) calibration, navigate to a shooting waypoint
                for w in shoot_wps:
                    nav_to_shoot = self._dist_steps(r, cal_end_wp, w)
                    # take_image cost 1
                    take_cost = 1
                    # then communicate image
                    comm_cost = self._estimate_comm_cost(r, w, lander_wp)
                    best = min(best, cal_cost + nav_to_shoot + take_cost + comm_cost)

        if best >= 10**9:
            return 1000
        return best

    # ------------------------ Communication / calibration helpers ------------------------

    def _estimate_comm_cost(self, rover: str, rover_wp: Optional[str], lander_wp: Optional[str]) -> int:
        """Lower bound for sending any data from rover to lander.

        communicate_* requires rover at some waypoint p1 and visible[p1, lander_wp].
        If rover already at such a waypoint -> 1 (communicate action).
        Else -> navigate to nearest waypoint that can see lander + 1.
        """
        if lander_wp is None or rover_wp is None:
            return 5  # fallback

        # If already visible, just communicate
        if (rover_wp, lander_wp) in self._visible_wp:
            return 1

        # Otherwise find nearest waypoint v with visible[v, lander_wp]
        best_nav = 10**9
        for v in self._waypoints:
            if (v, lander_wp) in self._visible_wp:
                d = self._dist_steps(rover, rover_wp, v)
                if d < best_nav:
                    best_nav = d

        if best_nav >= 10**9:
            # If no visible waypoint known, still need at least communicate
            return 2
        return best_nav + 1

    def _estimate_calibration_cost(self, rover: str, rover_wp: Optional[str], cam: str) -> Tuple[int, Optional[str]]:
        """Return (cost, end_wp) for calibrating camera cam on rover.

        calibrate(x,c,o,p) preconditions: at rover p and visible_from(o,p) and calibration_target(c,o).
        For LB we choose best objective o among calibration_target(cam, o) and any waypoint p where visible_from(o,p).

        Cost model: navigate to p + 1 calibrate.
        """
        if rover_wp is None:
            return 10**9, rover_wp

        targets = self._calib_target.get(cam, set())
        if not targets:
            return 10**9, rover_wp

        best = 10**9
        best_wp = None
        for o in targets:
            wps = self._visible_from.get(o, set())
            for w in wps:
                d = self._dist_steps(rover, rover_wp, w)
                if d + 1 < best:
                    best = d + 1
                    best_wp = w

        if best_wp is None:
            return 10**9, rover_wp
        return best, best_wp

    # ------------------------ State extraction (bitwise only) ------------------------

    def _current_lander_waypoint(self, state: int) -> Optional[str]:
        for fid in self._lander_at_fact_ids:
            if (state >> fid) & 1:
                fact = self.model.get_component(fid)
                pred, args = self._parse_fact_name(fact.name)
                if pred == "at_lander" and len(args) == 2:
                    return args[1]
        return None

    def _current_rover_positions(self, state: int) -> Dict[str, str]:
        pos: Dict[str, str] = {}
        # Scan facts once: in Pytrich facts are indexed; scanning all facts each call is too slow.
        # Instead, derive rover positions by using known rover names and searching for +at[rover,*].
        # We do it by scanning candidate waypoints per rover with early exit.
        for r in self._rovers:
            # Fast path: try to find any at fact by enumerating waypoints.
            # This is O(|rovers|*|waypoints|) but waypoints ~130 and rovers <=25 in largest -> ~3250 checks,
            # acceptable compared to expanding many nodes.
            for w in self._waypoints:
                if self._is_true(state, f"+at[{r},{w}]"):
                    pos[r] = w
                    break
        return pos

    def _current_store_fullness(self, state: int) -> Dict[str, bool]:
        full: Dict[str, bool] = {}
        for r, s in self._stores_of.items():
            full[r] = self._is_true(state, f"+full[{s}]")
        return full

    # ------------------------ Distance computation ------------------------

    def _compute_all_pairs_shortest_paths(self, rover: str) -> Dict[str, Dict[str, int]]:
        edges = self._can_traverse_edges.get(rover, [])
        adj: Dict[str, List[str]] = {}
        for a, b in edges:
            adj.setdefault(a, []).append(b)

        # BFS from each waypoint reachable set.
        dist_from: Dict[str, Dict[str, int]] = {}
        for src in self._waypoints:
            dq = deque([src])
            dmap = {src: 0}
            while dq:
                u = dq.popleft()
                for v in adj.get(u, []):
                    if v not in dmap:
                        dmap[v] = dmap[u] + 1
                        dq.append(v)
            dist_from[src] = dmap
        return dist_from

    def _dist_steps(self, rover: str, a: Optional[str], b: Optional[str]) -> int:
        if a is None or b is None:
            return 10**6
        if a == b:
            return 0
        d = self._dist.get(rover, {}).get(a, {}).get(b)
        if d is None:
            return 10**6
        return d

    # ------------------------ Fact utilities ------------------------

    def _is_true(self, state: int, fact_name: str) -> bool:
        f = self._fact_by_name.get(fact_name)
        if f is None:
            return False
        return ((state >> f.global_id) & 1) == 1

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
