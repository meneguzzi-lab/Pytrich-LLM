from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic15(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic features)
    ------------------------------------------------------
    The Rover domain’s top-level goals are *communications*:
      - communicated_soil_data(waypoint)
      - communicated_rock_data(waypoint)
      - communicated_image_data(objective, mode)

    Achieving each such goal requires (via the HTN methods) a pipeline:
      Soil/Rock: navigate to sample waypoint -> (optionally) empty store -> sample -> navigate to a comm waypoint
                that is visible to the lander -> communicate.
      Image: calibrate (navigate to calibration visibility waypoint + calibrate) -> navigate to imaging waypoint -> take_image
             -> navigate to a comm waypoint visible to lander -> communicate.

    The dominant cost driver is navigation (many waypoints, sparse can_traverse graph). Therefore the heuristic:
      1) Precomputes shortest-path distances (BFS) on the rover traversal graph per rover.
      2) For each *unsatisfied* communication goal, estimates a relaxed lower bound cost using:
           - minimal navigation to reach required sampling/imaging waypoint
           - minimal additional navigation to reach a waypoint that can communicate with the lander
           - minimal number of required primitive actions in the pipeline (sample/communicate/calibrate/take_image/drop)
      3) Combines goal estimates with a MAX (critical-path) aggregation to remain a lower bound under parallelism/independence.
         (Summing would often overestimate because multiple goals can be satisfied along shared navigation.)

    This heuristic is not guaranteed admissible in the strictest sense because it reasons with a relaxed model and
    uses minimal action-count constants, but it is designed to be a strong, fast guide that reduces node expansions
    substantially on the provided training distribution.

    Efficiency notes
    ----------------
    - Never calls state_explicit_repr on search states; uses bitwise checks with fact global IDs.
    - Caches parsed facts and precomputed waypoint sets for communication and visibility.
    - Uses BFS over waypoint graph per rover only once at initialization.
    """

    def __init__(self, model, rover_gpt52_heuristic_15: str = "rover_gpt52_heuristic_15"):
        super().__init__(model, rover_gpt52_heuristic_15)
        self.model = model

        # --- Fact parsing / indexing ---
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._parsed_fact_cache: Dict[str, Tuple[Optional[str], Tuple[str, ...]]] = {}

        # Domain objects (guessed from facts; robust to missing type tables)
        self.rovers: set[str] = set()
        self.waypoints: set[str] = set()
        self.landers: set[str] = set()
        self.cameras: set[str] = set()
        self.objectives: set[str] = set()
        self.modes: set[str] = set()

        # Static relations
        self.at_lander: Dict[str, str] = {}  # lander -> waypoint
        self.visible_wps: Dict[str, set[str]] = defaultdict(set)  # wp -> set(wp)
        self.can_traverse: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))  # rover -> from -> {to}
        self.visible_from_obj: Dict[str, set[str]] = defaultdict(set)  # objective -> {wp}
        self.calib_target: Dict[str, str] = {}  # camera -> objective
        self.on_board: Dict[str, str] = {}  # camera -> rover
        self.supports: Dict[str, set[str]] = defaultdict(set)  # camera -> {mode}
        self.store_of: Dict[str, str] = {}  # store -> rover

        # Useful waypoint sets for comm
        self.comm_wps: set[str] = set()  # waypoints that can communicate with lander (visible to lander wp)

        # Precomputed distances: rover -> (from_wp -> {to_wp: dist})
        self._dist: Dict[str, Dict[str, Dict[str, int]]] = {}

        self._index_all_facts_and_statics()
        self._compute_comm_waypoints()
        self._precompute_all_rover_distances()

        # Cache goal fact keys for quick evaluation
        self._goal_fact_ids: List[int] = []
        self._goal_comm_soil: List[str] = []
        self._goal_comm_rock: List[str] = []
        self._goal_comm_image: List[Tuple[str, str]] = []
        self._index_goals()

    # ------------------------- Parsing helpers -------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        cached = self._parsed_fact_cache.get(name)
        if cached is not None:
            return cached

        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s2 = s[1:]
            b = s2.find("[")
            if b > 0 and s2.endswith("]"):
                pred = s2[:b]
                args_str = s2[b + 1 : -1]
                args = tuple(args_str.split(",")) if args_str else tuple()
                res = (pred, args)
                self._parsed_fact_cache[name] = res
                return res
        res = (None, tuple())
        self._parsed_fact_cache[name] = res
        return res

    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    # ------------------------- Initialization -------------------------
    def _index_all_facts_and_statics(self) -> None:
        # Build key->fact_id map and collect static-like relations from initial facts.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_by_key[(pred, args)] = f.global_id

            # Collect object names by predicate signatures we see
            if pred == "at" and len(args) == 2:
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
            elif pred in ("visible",) and len(args) == 2:
                self.waypoints.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "can_traverse" and len(args) == 3:
                self.rovers.add(args[0])
                self.waypoints.add(args[1])
                self.waypoints.add(args[2])
            elif pred == "at_lander" and len(args) == 2:
                self.landers.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "visible_from" and len(args) == 2:
                self.objectives.add(args[0])
                self.waypoints.add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self.cameras.add(args[0])
                self.objectives.add(args[1])
            elif pred == "on_board" and len(args) == 2:
                self.cameras.add(args[0])
                self.rovers.add(args[1])
            elif pred == "supports" and len(args) == 2:
                self.cameras.add(args[0])
                self.modes.add(args[1])
            elif pred == "store_of" and len(args) == 2:
                self.store_of[args[0]] = args[1]
                self.rovers.add(args[1])

        # Extract (mostly static) relations from initial state bitset
        init = self.model.initial_state
        for f in self.model.facts:
            if not self._is_true(init, f.global_id):
                continue
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            if pred == "at_lander" and len(args) == 2:
                self.at_lander[args[0]] = args[1]
            elif pred == "visible" and len(args) == 2:
                self.visible_wps[args[0]].add(args[1])
            elif pred == "can_traverse" and len(args) == 3:
                self.can_traverse[args[0]][args[1]].add(args[2])
            elif pred == "visible_from" and len(args) == 2:
                self.visible_from_obj[args[0]].add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self.calib_target[args[0]] = args[1]
            elif pred == "on_board" and len(args) == 2:
                self.on_board[args[0]] = args[1]
            elif pred == "supports" and len(args) == 2:
                self.supports[args[0]].add(args[1])

        # If multiple landers existed, we approximate by allowing communication to any; comm_wps will be union.

    def _compute_comm_waypoints(self) -> None:
        # Waypoints from which lander waypoint is visible are valid for communication method m8/m11/m14.
        for lander, lw in self.at_lander.items():
            for w1, vis_set in self.visible_wps.items():
                if lw in vis_set:
                    self.comm_wps.add(w1)
            # Also lw itself is obviously visible to itself only if explicit; but communication requires visible(w1,lw).
            # If domain instances omit visible(lw,lw), we do NOT add it blindly.

    def _precompute_all_rover_distances(self) -> None:
        # BFS all-pairs-from-each-start for each rover. Waypoints count is up to ~130, rovers up to 25.
        # Complexity acceptable: O(|R|*(|V|+|E|)*|V|) in worst case if naive all-sources.
        # We instead do BFS from each waypoint that appears as a source in can_traverse; plus we may query others.
        for r in self.rovers:
            adj = self.can_traverse.get(r, {})
            # Gather relevant sources: all waypoints we might start from or need to reach.
            sources = set(adj.keys()) | self.waypoints
            dist_from: Dict[str, Dict[str, int]] = {}
            for s in sources:
                dist_from[s] = self._bfs_dist(s, adj)
            self._dist[r] = dist_from

    def _bfs_dist(self, start: str, adj: Dict[str, set[str]]) -> Dict[str, int]:
        d: Dict[str, int] = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj.get(u, ()):  # directed edges as in can_traverse
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def _index_goals(self) -> None:
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred is None:
                continue
            fid = self._fact_by_key.get((pred, args))
            if fid is not None:
                self._goal_fact_ids.append(fid)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_comm_soil.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_comm_rock.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_comm_image.append((args[0], args[1]))

    # ------------------------- Distance queries -------------------------
    def _rover_positions(self, state: int) -> Dict[str, str]:
        # Extract rover positions by scanning facts of type at[rover,wp] would be expensive if we scan all facts.
        # We instead precompute all at-fact ids for each rover during first call lazily.
        if not hasattr(self, "_at_facts_by_rover"):
            at_by_rover = defaultdict(list)
            for (pred, args), fid in self._fact_by_key.items():
                if pred == "at" and len(args) == 2:
                    at_by_rover[args[0]].append((args[1], fid))
            self._at_facts_by_rover = dict(at_by_rover)

        pos: Dict[str, str] = {}
        for r, lst in self._at_facts_by_rover.items():
            for wp, fid in lst:
                if self._is_true(state, fid):
                    pos[r] = wp
                    break
        return pos

    def _min_dist(self, rover: str, src: str, dst: str) -> int:
        if src == dst:
            return 0
        dmap = self._dist.get(rover)
        if not dmap:
            return 10**6
        return dmap.get(src, {}).get(dst, 10**6)

    def _min_dist_to_set(self, rover: str, src: str, targets: set[str]) -> int:
        if not targets:
            return 10**6
        dmap = self._dist.get(rover, {}).get(src, {})
        best = 10**6
        for t in targets:
            if t == src:
                return 0
            dt = dmap.get(t)
            if dt is not None and dt < best:
                best = dt
        return best

    # ------------------------- Goal estimators -------------------------
    def _estimate_soil_comm_goal(self, state: int, rover_pos: Dict[str, str], wp: str) -> int:
        # If already communicated, cost 0
        fid = self._fact_by_key.get(("communicated_soil_data", (wp,)))
        if fid is not None and self._is_true(state, fid):
            return 0

        # Need: have_soil_analysis(rover, wp) then communicate from some comm waypoint.
        # Relaxation: choose best rover that is equipped and can reach wp and comm waypoint.
        best = 10**6

        for r, rwp in rover_pos.items():
            # Must be equipped_for_soil_analysis
            eq_id = self._fact_by_key.get(("equipped_for_soil_analysis", (r,)))
            if eq_id is None or not self._is_true(state, eq_id):
                continue

            have_id = self._fact_by_key.get(("have_soil_analysis", (r, wp)))
            has_analysis = have_id is not None and self._is_true(state, have_id)

            # Communication waypoint must see lander
            d_comm = self._min_dist_to_set(r, rwp if has_analysis else wp, self.comm_wps)

            if has_analysis:
                # Just navigate to comm wp and communicate.
                cost = d_comm + 1  # communicate action
            else:
                # Need navigate to wp, (maybe drop), sample, then navigate to comm wp, communicate.
                d_to_wp = self._min_dist(r, rwp, wp)
                if d_to_wp >= 10**6:
                    continue

                # Store emptying: if some store of r is full, at least one drop needed.
                drop_needed = 0
                # Find any store s with store_of(s,r)
                for (pred, args), fid_s in self._fact_by_key.items():
                    if pred == "store_of" and len(args) == 2 and args[1] == r:
                        s = args[0]
                        full_id = self._fact_by_key.get(("full", (s,)))
                        empty_id = self._fact_by_key.get(("empty", (s,)))
                        if full_id is not None and self._is_true(state, full_id) and not (empty_id is not None and self._is_true(state, empty_id)):
                            drop_needed = 1
                        break

                d_wp_to_comm = self._min_dist_to_set(r, wp, self.comm_wps)
                cost = d_to_wp + drop_needed + 1 + d_wp_to_comm + 1
                #             nav      drop?   sample   nav_comm         comm
            if cost < best:
                best = cost

        return best

    def _estimate_rock_comm_goal(self, state: int, rover_pos: Dict[str, str], wp: str) -> int:
        fid = self._fact_by_key.get(("communicated_rock_data", (wp,)))
        if fid is not None and self._is_true(state, fid):
            return 0

        best = 10**6
        for r, rwp in rover_pos.items():
            eq_id = self._fact_by_key.get(("equipped_for_rock_analysis", (r,)))
            if eq_id is None or not self._is_true(state, eq_id):
                continue

            have_id = self._fact_by_key.get(("have_rock_analysis", (r, wp)))
            has_analysis = have_id is not None and self._is_true(state, have_id)

            d_comm = self._min_dist_to_set(r, rwp if has_analysis else wp, self.comm_wps)

            if has_analysis:
                cost = d_comm + 1
            else:
                d_to_wp = self._min_dist(r, rwp, wp)
                if d_to_wp >= 10**6:
                    continue

                drop_needed = 0
                for (pred, args), _fid_s in self._fact_by_key.items():
                    if pred == "store_of" and len(args) == 2 and args[1] == r:
                        s = args[0]
                        full_id = self._fact_by_key.get(("full", (s,)))
                        empty_id = self._fact_by_key.get(("empty", (s,)))
                        if full_id is not None and self._is_true(state, full_id) and not (empty_id is not None and self._is_true(state, empty_id)):
                            drop_needed = 1
                        break

                d_wp_to_comm = self._min_dist_to_set(r, wp, self.comm_wps)
                cost = d_to_wp + drop_needed + 1 + d_wp_to_comm + 1
            if cost < best:
                best = cost

        return best

    def _estimate_image_comm_goal(self, state: int, rover_pos: Dict[str, str], obj: str, mode: str) -> int:
        fid = self._fact_by_key.get(("communicated_image_data", (obj, mode)))
        if fid is not None and self._is_true(state, fid):
            return 0

        best = 10**6

        for cam, r in self.on_board.items():
            if r not in rover_pos:
                continue
            rwp = rover_pos[r]

            # Rover must be equipped_for_imaging
            eq_id = self._fact_by_key.get(("equipped_for_imaging", (r,)))
            if eq_id is None or not self._is_true(state, eq_id):
                continue

            # Camera must support mode
            if mode not in self.supports.get(cam, set()):
                continue

            have_img_id = self._fact_by_key.get(("have_image", (r, obj, mode)))
            has_image = have_img_id is not None and self._is_true(state, have_img_id)

            if has_image:
                d_to_comm = self._min_dist_to_set(r, rwp, self.comm_wps)
                cost = d_to_comm + 1
                if cost < best:
                    best = cost
                continue

            # Need to take image from some waypoint w with visible_from(obj,w)
            img_wps = self.visible_from_obj.get(obj, set())
            if not img_wps:
                continue

            # Need calibration first: pick calibration objective for that camera, and a waypoint v where it's visible.
            calib_obj = self.calib_target.get(cam)
            if calib_obj is None:
                continue
            calib_wps = self.visible_from_obj.get(calib_obj, set())
            if not calib_wps:
                continue

            # If already calibrated(cam,r), can skip calibration travel/action.
            calibrated_id = self._fact_by_key.get(("calibrated", (cam, r)))
            is_calibrated = calibrated_id is not None and self._is_true(state, calibrated_id)

            # Distances:
            if is_calibrated:
                d_to_img = self._min_dist_to_set(r, rwp, img_wps)
                if d_to_img >= 10**6:
                    continue
                # take_image, then comm
                # after take_image, calibrated becomes false but irrelevant for this goal
                # Choose best comm after taking at image wp
                # Relax: use same best comm distance from best image wp approximated by min over all image wps
                d_img_to_comm = 10**6
                # compute min over candidate image wps using precomputed distances from each
                for iw in img_wps:
                    di = self._min_dist(r, rwp, iw)
                    if di >= 10**6:
                        continue
                    d2 = self._min_dist_to_set(r, iw, self.comm_wps)
                    if d2 < d_img_to_comm:
                        d_img_to_comm = d2
                cost = d_to_img + 1 + d_img_to_comm + 1
            else:
                # Need navigate to some calib wp, calibrate, then to image wp, take_image, then to comm wp, communicate.
                d_to_cal = self._min_dist_to_set(r, rwp, calib_wps)
                if d_to_cal >= 10**6:
                    continue

                # Best cost from some calib wp to some img wp
                d_cal_to_img = 10**6
                # approximate by trying a limited number of waypoints if sets are huge
                # (in instances many objectives are visible from almost all waypoints; still <=130)
                for cw in calib_wps:
                    # skip unreachable
                    dcw = self._min_dist(r, rwp, cw)
                    if dcw >= 10**6:
                        continue
                    dmap_cw = self._dist[r].get(cw, {})
                    for iw in img_wps:
                        di = dmap_cw.get(iw)
                        if di is not None and di < d_cal_to_img:
                            d_cal_to_img = di
                if d_cal_to_img >= 10**6:
                    continue

                # After taking image at iw, go to comm
                d_img_to_comm = 10**6
                for iw in img_wps:
                    d2 = self._min_dist_to_set(r, iw, self.comm_wps)
                    if d2 < d_img_to_comm:
                        d_img_to_comm = d2

                cost = d_to_cal + 1 + d_cal_to_img + 1 + d_img_to_comm + 1
                #        nav    calib   nav_to_img  take    nav_comm    comm

            if cost < best:
                best = cost

        return best

    # ------------------------- Main heuristic -------------------------
    def __call__(self, node) -> int:
        state = node.state

        # Quick goal test: if all goal facts already true, return 0.
        # (Task network emptiness is handled by planner; heuristic only estimates remaining effort.)
        all_true = True
        for fid in self._goal_fact_ids:
            if not self._is_true(state, fid):
                all_true = False
                break
        if all_true:
            return 0

        rover_pos = self._rover_positions(state)

        # Critical-path style aggregation: take maximum among individual communication goals.
        # This is robust when many goals can be achieved along a shared route.
        h = 0

        for wp in self._goal_comm_soil:
            c = self._estimate_soil_comm_goal(state, rover_pos, wp)
            if c > h:
                h = c

        for wp in self._goal_comm_rock:
            c = self._estimate_rock_comm_goal(state, rover_pos, wp)
            if c > h:
                h = c

        for obj, mode in self._goal_comm_image:
            c = self._estimate_image_comm_goal(state, rover_pos, obj, mode)
            if c > h:
                h = c

        # Mild tie-breaker: unfinished goals count (very small weight) to prefer states satisfying more goals.
        # Keep integer heuristic; do not dominate distance estimates.
        unsat = 0
        for fid in self._goal_fact_ids:
            if not self._is_true(state, fid):
                unsat += 1
        # Add at most 1 extra to preserve scale.
        if unsat > 0:
            h += 1

        return h
