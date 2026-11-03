from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Set

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic5(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    RATIONALE (domain insights -> heuristic components)
    -----------------------------------------------
    The Rover domain goals are of the form:
      - communicated_soil_data(waypoint)
      - communicated_rock_data(waypoint)
      - communicated_image_data(objective, mode)

    Achieving any communicated_* goal requires, at minimum:
      (1) producing the corresponding analysis/image on some rover
      (2) being at a waypoint that is visible to the lander waypoint (or the sample waypoint itself)
      (3) communicating

    The HDDL methods enforce a stereotypical structure:
      get_soil_data(wp): navigate -> empty_store -> sample_soil -> send_soil_data
      get_rock_data(wp): navigate -> empty_store -> sample_rock -> send_rock_data
      get_image_data(o,m): do_calibrate -> navigate -> take_image -> send_image_data

    Key planning challenges:
      - Navigation dominates cost: reaching sample points and communication vantage points.
      - Store must be empty before sampling: if full, an extra drop is needed.
      - Imaging requires a (camera, objective) calibration at some waypoint visible_from(objective, wp);
        and after taking an image, calibration is lost.
      - Large instances: many rovers/waypoints, need O(|V|+|E|) preprocessing and O(#goals) evaluation.

    HEURISTIC DESIGN
    ----------------
    We combine:
      A) Goal-count lower bound: each unsatisfied communicated_* goal needs at least 1 communicate action.
         Additionally if the corresponding "have_*" fact is not yet true for any rover, at least 1 more
         action is needed to create it (sample/take_image). This yields a cheap admissible bound.

      B) Navigation lower bound: for each unsatisfied goal, add a *minimal* navigation distance
         (number of navigate actions) needed to get some capable rover to a place from which the
         communication can occur.

         - For soil/rock goal at waypoint p:
              need some rover x that can do analysis to reach p (for sampling) and then to some
              waypoint w that is visible to the lander waypoint L (for communication), unless p itself
              is visible to L.
           Lower bound uses graph distances on can_traverse edges (ignore 'visited' bookkeeping;
           methods add visit/unvisit but those are abstract-level and do not appear as costs at
           primitive level other than navigate, so ignoring visited is a relaxation).

         - For image goal (o,m): need some rover with a camera that supports m and has calibration_target
           for some objective o_cal, and some waypoint w_img visible_from(o, w_img) to take image.
           Calibration requires reaching some waypoint w_cal visible_from(o_cal, w_cal).
           Lower bound estimates:
              dist(current, w_cal) + dist(w_cal, w_img) + dist(w_img, w_comm)
           where w_comm is a waypoint visible to lander.

    Aggregation: we avoid inadmissible summing over goals (which can overcount shared moves). Instead
    we take:
       h = max( A_total, B_max )
    where:
       - A_total sums per-goal minimum required non-navigation actions (safe as independent lower bound
         on number of primitive actions that must occur at least once per goal).
       - B_max takes the maximum navigation lower bound over all pending goals, a standard admissible
         way to capture at least one hard-to-reach goal without double counting travel.

    This heuristic is admissible under unit action costs (as in the provided domain) and remains fast
    via preprocessing of distances per rover.
    """

    def __init__(self, model, rover_gpt52_heuristic_5: str = "rover_gpt52_heuristic_5"):
        super().__init__(model, rover_gpt52_heuristic_5)
        self.model = model

        # ---- Parse all facts once, build indices for fast bitwise checks ----
        self._fact_gid_by_name: Dict[str, int] = {}
        for f in self.model.facts:
            self._fact_gid_by_name[f.name] = f.global_id

        # Goal facts (explicit names) are needed to iterate goal predicates
        self._goal_fact_names: List[str] = self.model.state_explicit_repr(self.model.goals)

        # ---- Domain object relations extracted from facts (static) ----
        # We extract from *all* facts in the model (not state), because they represent the grounded universe.
        self._rover_set: Set[str] = set()
        self._waypoints: Set[str] = set()
        self._landers: Set[str] = set()

        # Traversability graph per rover: adjacency list (undirected in practice, but we use directed edges).
        self._traverse_adj: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Visibility between waypoints (needed for communication and for navigate action too, but traverse edges already
        # capture possible movement; visibility for comm with lander uses visible[w1,w2]).
        self._visible_wp: Dict[str, Set[str]] = defaultdict(set)

        # Lander location(s): lander -> waypoint
        self._lander_at: Dict[str, str] = {}

        # Cameras on rovers, supports, calibration targets
        self._cameras_on_rover: Dict[str, Set[str]] = defaultdict(set)
        self._camera_supports: Dict[str, Set[str]] = defaultdict(set)
        self._camera_cal_target: Dict[str, str] = {}  # camera -> objective

        # Objective visibility: objective -> set(waypoint)
        self._objective_visible_from: Dict[str, Set[str]] = defaultdict(set)

        # Capabilities
        self._rover_soil_capable: Set[str] = set()
        self._rover_rock_capable: Set[str] = set()
        self._rover_img_capable: Set[str] = set()

        # Store on rover
        self._store_of: Dict[str, str] = {}  # rover -> store

        # Precompute sets of communication vantage waypoints relative to the (single) lander waypoint.
        self._comm_vantage_wps: Set[str] = set()

        # Parse all fact templates
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue

            if pred == "can_traverse" and len(args) == 3:
                r, w1, w2 = args
                self._rover_set.add(r)
                self._waypoints.add(w1)
                self._waypoints.add(w2)
                self._traverse_adj[r][w1].append(w2)

            elif pred == "visible" and len(args) == 2:
                w1, w2 = args
                self._waypoints.add(w1)
                self._waypoints.add(w2)
                self._visible_wp[w1].add(w2)

            elif pred == "at_lander" and len(args) == 2:
                l, w = args
                self._landers.add(l)
                self._waypoints.add(w)
                # static in instances: lander fixed
                self._lander_at[l] = w

            elif pred == "on_board" and len(args) == 2:
                c, r = args
                self._cameras_on_rover[r].add(c)
                self._rover_set.add(r)

            elif pred == "supports" and len(args) == 2:
                c, m = args
                self._camera_supports[c].add(m)

            elif pred == "calibration_target" and len(args) == 2:
                c, o = args
                self._camera_cal_target[c] = o

            elif pred == "visible_from" and len(args) == 2:
                o, w = args
                self._objective_visible_from[o].add(w)
                self._waypoints.add(w)

            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self._rover_soil_capable.add(args[0])
                self._rover_set.add(args[0])

            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self._rover_rock_capable.add(args[0])
                self._rover_set.add(args[0])

            elif pred == "equipped_for_imaging" and len(args) == 1:
                self._rover_img_capable.add(args[0])
                self._rover_set.add(args[0])

            elif pred == "store_of" and len(args) == 2:
                s, r = args
                self._store_of[r] = s
                self._rover_set.add(r)

        # Choose a lander (all benchmarks use one). If multiple, union their vantage sets.
        self._lander_waypoints: Set[str] = set(self._lander_at.values())
        for lw in self._lander_waypoints:
            # communication requires visible(w_rover, lw)
            for w1, visset in self._visible_wp.items():
                if lw in visset:
                    self._comm_vantage_wps.add(w1)

        # Precompute for each objective: list of waypoints where it can be seen
        # Already built in _objective_visible_from.

        # ---- Distance preprocessing per rover: BFS from every waypoint would be expensive.
        # Instead, for heuristic we need distances from CURRENT rover positions to a set of targets.
        # We'll do BFS from current position on demand and cache per (rover, start_wp) because starts change.
        self._dist_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

        # Cache goal predicate parsing
        self._parsed_goals = [self._parse_fact_name(g) for g in self._goal_fact_names]

    # ----------------- Core Heuristic API -----------------
    def __call__(self, node) -> int:
        state = node.state

        # If already at goal with empty task network, heuristic should be 0 (planner will check goal itself too).
        # We'll just compute; it will become 0.

        # Get current rover positions
        rover_at: Dict[str, str] = {}
        # Also store store fullness
        store_empty: Set[str] = set()
        store_full: Set[str] = set()

        # Scan relevant facts by iterating over known candidates and using bit tests.
        # For at[r,w] we don't know w; but we do know all at facts exist in model, yet enumerating all would be large.
        # We'll instead detect rover positions by scanning all facts *once* and caching at-facts gid lists per rover.
        # Lazily build index the first time.
        if not hasattr(self, "_at_fact_gids_by_rover"):
            self._at_fact_gids_by_rover = defaultdict(list)
            self._empty_gid_by_store = {}
            self._full_gid_by_store = {}
            self._have_soil_gids_by_wp = defaultdict(list)   # wp -> list of gids have_soil_analysis[rover,wp]
            self._have_rock_gids_by_wp = defaultdict(list)
            self._have_image_gids_by_obj_mode = defaultdict(list)  # (o,m) -> gids

            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if not pred:
                    continue
                if pred == "at" and len(args) == 2:
                    r, w = args
                    self._at_fact_gids_by_rover[r].append((w, f.global_id))
                elif pred == "empty" and len(args) == 1:
                    self._empty_gid_by_store[args[0]] = f.global_id
                elif pred == "full" and len(args) == 1:
                    self._full_gid_by_store[args[0]] = f.global_id
                elif pred == "have_soil_analysis" and len(args) == 2:
                    r, w = args
                    self._have_soil_gids_by_wp[w].append(f.global_id)
                elif pred == "have_rock_analysis" and len(args) == 2:
                    r, w = args
                    self._have_rock_gids_by_wp[w].append(f.global_id)
                elif pred == "have_image" and len(args) == 3:
                    r, o, m = args
                    self._have_image_gids_by_obj_mode[(o, m)].append(f.global_id)

        for r, w_gid_list in self._at_fact_gids_by_rover.items():
            for w, gid in w_gid_list:
                if (state >> gid) & 1:
                    rover_at[r] = w
                    break

        for r, s in self._store_of.items():
            egid = self._empty_gid_by_store.get(s)
            fgid = self._full_gid_by_store.get(s)
            if egid is not None and ((state >> egid) & 1):
                store_empty.add(s)
            if fgid is not None and ((state >> fgid) & 1):
                store_full.add(s)

        # Component A: sum of per-goal minimum required non-navigation actions
        non_nav_lb_sum = 0
        # Component B: maximum navigation lower bound among goals
        nav_lb_max = 0

        for (pred, args), gname in zip(self._parsed_goals, self._goal_fact_names):
            if not pred:
                continue
            gid = self._fact_gid_by_name.get(gname)
            if gid is not None and ((state >> gid) & 1):
                continue  # goal already satisfied

            if pred == "communicated_soil_data" and len(args) == 1:
                wp = args[0]
                # At least one communicate action
                non_nav_lb_sum += 1
                # If no rover has soil analysis yet for that wp, need at least one sampling action
                if not self._any_fact_true(state, self._have_soil_gids_by_wp.get(wp, [])):
                    non_nav_lb_sum += 1
                    # If the sampling store of chosen rover might be full, at least one drop might be needed;
                    # but for admissibility we only add it if *all* soil-capable rovers have full stores.
                    if self._all_capable_stores_full(state, store_full, self._rover_soil_capable):
                        non_nav_lb_sum += 1

                nav_lb_max = max(nav_lb_max, self._soil_rock_nav_lb(rover_at, wp, self._rover_soil_capable))

            elif pred == "communicated_rock_data" and len(args) == 1:
                wp = args[0]
                non_nav_lb_sum += 1
                if not self._any_fact_true(state, self._have_rock_gids_by_wp.get(wp, [])):
                    non_nav_lb_sum += 1
                    if self._all_capable_stores_full(state, store_full, self._rover_rock_capable):
                        non_nav_lb_sum += 1

                nav_lb_max = max(nav_lb_max, self._soil_rock_nav_lb(rover_at, wp, self._rover_rock_capable))

            elif pred == "communicated_image_data" and len(args) == 2:
                o, m = args
                non_nav_lb_sum += 1
                if not self._any_fact_true(state, self._have_image_gids_by_obj_mode.get((o, m), [])):
                    non_nav_lb_sum += 1  # take_image at least
                    non_nav_lb_sum += 1  # calibrate at least (needed before take_image)

                nav_lb_max = max(nav_lb_max, self._image_nav_lb(rover_at, o, m))

            else:
                # Unknown goal type; fallback: count as 1 missing
                non_nav_lb_sum += 1

        return max(non_nav_lb_sum, nav_lb_max)

    # ----------------- Navigation lower bounds -----------------
    def _soil_rock_nav_lb(self, rover_at: Dict[str, str], sample_wp: str, capable_rovers: Set[str]) -> int:
        """Lower bound on navigation steps to (potentially) achieve a soil/rock communication for sample_wp.

        Relaxation:
          - Ignore store and sampling preconditions.
          - Assume once rover reached sample_wp it can produce analysis.
          - Then it must reach some comm waypoint w such that visible(w, lander_wp).
            If sample_wp itself is visible to lander, communication can happen there.

        We take the minimum over capable rovers of:
          dist(start, sample_wp) + min_w dist(sample_wp, w)
        and return that as a navigation LB.

        If no traverse info or rover position is unknown, returns 0 (uninformative but safe).
        """
        if not self._comm_vantage_wps or not sample_wp:
            return 0

        # Precompute set of possible comm points for this sample.
        # If any lander waypoint is directly visible from sample_wp, allow communicating at sample_wp.
        comm_targets = set(self._comm_vantage_wps)
        for lw in self._lander_waypoints:
            if lw in self._visible_wp.get(sample_wp, set()):
                comm_targets.add(sample_wp)
                break

        best = None
        for r in capable_rovers:
            start = rover_at.get(r)
            if start is None:
                continue
            d1 = self._dist(r, start).get(sample_wp)
            if d1 is None:
                continue
            d2 = self._min_dist_to_any(r, sample_wp, comm_targets)
            if d2 is None:
                continue
            total = d1 + d2
            if best is None or total < best:
                best = total

        return best if best is not None else 0

    def _image_nav_lb(self, rover_at: Dict[str, str], obj: str, mode: str) -> int:
        """Lower bound on navigation steps to achieve communicated_image_data(obj,mode).

        Need a rover that:
          - is imaging capable
          - has some on-board camera c with supports(c, mode)
          - has calibration_target(c, o_cal)

        Then it must:
          - go to some w_cal in visible_from(o_cal, w_cal) to calibrate
          - go to some w_img in visible_from(obj, w_img) to take image
          - go to some w_comm s.t. visible(w_comm, lander_wp) to communicate

        We compute:
          min_{r,c}  min_{w_cal} dist(start,w_cal) + min_{w_img} dist(w_cal,w_img) + min_{w_comm} dist(w_img,w_comm)

        using precomputed shortest-path distances per rover (on demand BFS from each start point).
        """
        if not obj or not mode or not self._comm_vantage_wps:
            return 0

        img_wps = self._objective_visible_from.get(obj)
        if not img_wps:
            return 0

        best = None
        for r in self._rover_img_capable:
            start = rover_at.get(r)
            if start is None:
                continue

            # feasible cameras for (r, mode)
            cams = [c for c in self._cameras_on_rover.get(r, set()) if mode in self._camera_supports.get(c, set())]
            if not cams:
                continue

            # For each camera, get calibration objective and corresponding calibration waypoints
            for c in cams:
                o_cal = self._camera_cal_target.get(c)
                if not o_cal:
                    continue
                cal_wps = self._objective_visible_from.get(o_cal)
                if not cal_wps:
                    continue

                # Compute min over cal waypoint
                dist_from_start = self._dist(r, start)
                for w_cal in cal_wps:
                    d_start_cal = dist_from_start.get(w_cal)
                    if d_start_cal is None:
                        continue

                    # From w_cal to some w_img
                    d_cal_to_img = self._min_dist_to_any(r, w_cal, img_wps)
                    if d_cal_to_img is None:
                        continue

                    # Lower bound from image waypoint to comm vantage.
                    # We don't know which w_img achieved d_cal_to_img, so for admissibility take:
                    #   min_{w_img in img_wps} dist(w_cal,w_img) + min_{w_img in img_wps} min_{w_comm} dist(w_img,w_comm)
                    # This is still admissible as sum of two independent minima, but could under-estimate.
                    d_img_to_comm = self._min_over_sources_to_targets(r, img_wps, self._comm_vantage_wps)
                    if d_img_to_comm is None:
                        # maybe comm at img wp if directly visible to lander
                        d_img_to_comm = 0

                    total = d_start_cal + d_cal_to_img + d_img_to_comm
                    if best is None or total < best:
                        best = total

        return best if best is not None else 0

    # ----------------- Distance utilities -----------------
    def _dist(self, rover: str, start_wp: str) -> Dict[str, int]:
        key = (rover, start_wp)
        if key in self._dist_cache:
            return self._dist_cache[key]

        adj = self._traverse_adj.get(rover)
        if not adj or start_wp not in adj:
            # still allow reaching itself
            self._dist_cache[key] = {start_wp: 0}
            return self._dist_cache[key]

        dist = {start_wp: 0}
        q = deque([start_wp])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        self._dist_cache[key] = dist
        return dist

    def _min_dist_to_any(self, rover: str, start_wp: str, targets: Set[str]) -> Optional[int]:
        d = self._dist(rover, start_wp)
        best = None
        for t in targets:
            dt = d.get(t)
            if dt is None:
                continue
            if best is None or dt < best:
                best = dt
        return best

    def _min_over_sources_to_targets(self, rover: str, sources: Set[str], targets: Set[str]) -> Optional[int]:
        best = None
        for s in sources:
            ds = self._min_dist_to_any(rover, s, targets)
            if ds is None:
                continue
            if best is None or ds < best:
                best = ds
        return best

    # ----------------- Bitwise helpers -----------------
    @staticmethod
    def _parse_fact_name(name: str):
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
    def _any_fact_true(state: int, gids: List[int]) -> bool:
        for gid in gids:
            if (state >> gid) & 1:
                return True
        return False

    def _all_capable_stores_full(self, state: int, store_full: Set[str], capable_rovers: Set[str]) -> bool:
        """Return True iff every capable rover that has a store has that store full.

        This allows adding a safe lower bound of at least one drop when no capable rover can sample
        without first dropping.
        """
        any_considered = False
        for r in capable_rovers:
            s = self._store_of.get(r)
            if not s:
                continue
            any_considered = True
            if s not in store_full:
                return False
        return any_considered
