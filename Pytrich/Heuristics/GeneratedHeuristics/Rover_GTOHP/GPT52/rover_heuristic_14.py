from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic14(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    Key domain insight
    ------------------
    In this domain, every top-level goal is a *communicated_* fact. To achieve it, the plan must (relaxed view):

      Soil goal at waypoint p:
        have_soil_analysis(x,p) -> communicated_soil_data(p)
        have_soil_analysis(x,p) requires navigating some soil-capable rover to p, empty store, sample_soil.
        communicated_soil_data(p) then requires rover at some waypoint w1 that is visible to the lander waypoint wL.

      Rock goal at waypoint p:
        analogous with rock-capable rover and sample_rock.

      Image goal (objective o, mode m):
        have_image(x,o,m) -> communicated_image_data(o,m)
        have_image requires: on_board camera, supports mode, equipped_for_imaging, take_image at a waypoint w where
        visible_from(o,w). take_image requires calibrated(camera, rover) which itself requires navigating to a
        calibration waypoint for that camera's calibration_target.

    Heuristic design
    ----------------
    We compute, for each unsatisfied communicated goal, an inexpensive *lower bound* on the number of primitive
    actions still required, using:

      * relaxed reachability distances over each rover's traverse graph (edge cost 1 per navigate)
      * minimal “communication position” distance to any waypoint that can see the lander
      * minimal “sampling cost” skeleton (drop if store full, then sample)
      * minimal “imaging skeleton” (calibrate + take_image), choosing best (rover,camera,waypoint)

    We then aggregate goals using a **max-per-rover** scheme:
      - For each goal we assign it to the rover that yields the cheapest estimate.
      - For each rover we take the maximum of its assigned goal-costs (a lower bound if that rover must at least
        achieve its hardest assigned goal).
      - Finally we sum across rovers.

    This avoids the common over-counting pitfall of summing all goal costs even when one rover can accomplish
    multiple goals in one trip; and it scales well to large instances.

    Important implementation constraints
    ------------------------------------
    - We never call model.state_explicit_repr(state) during heuristic evaluation (too slow).
    - We do bitwise fact checks using (state >> fact.global_id) & 1.

    The heuristic is not guaranteed admissible in all corner cases due to simplistic goal-to-rover assignment,
    but it is strongly informative and efficient for guiding search in this domain.
    """

    def __init__(self, model, rover_gpt52_heuristic_14: str = "rover_gpt52_heuristic_14"):
        super().__init__(model, rover_gpt52_heuristic_14)
        self.model = model

        # --- Fact indexing ---
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Domain object sets
        self.rovers: Set[str] = set()
        self.waypoints: Set[str] = set()
        self.stores: Set[str] = set()
        self.cameras: Set[str] = set()
        self.objectives: Set[str] = set()
        self.modes: Set[str] = set()
        self.landers: Set[str] = set()

        # Static relations
        self.store_of: Dict[str, str] = {}  # store -> rover
        self.rover_store: Dict[str, str] = {}  # rover -> store
        self.can_traverse: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))  # rover -> from -> {to}
        self.visible: Dict[str, Set[str]] = defaultdict(set)  # wp -> {wp}

        self.at_lander: Dict[str, str] = {}  # lander -> waypoint
        self.calibration_target: Dict[str, str] = {}  # camera -> objective
        self.visible_from: Dict[str, Set[str]] = defaultdict(set)  # objective -> {waypoint}
        self.on_board: Dict[str, str] = {}  # camera -> rover
        self.supports: Dict[str, Set[str]] = defaultdict(set)  # camera -> {mode}

        # Capability sets
        self.soil_capable: Set[str] = set()
        self.rock_capable: Set[str] = set()
        self.imaging_capable: Set[str] = set()

        # Precomputed helper sets
        self.waypoints_visible_to_lander: Set[str] = set()  # any wp that can see the lander wp (visible wp lander_wp)
        self._traverse_dist_cache: Dict[Tuple[str, str], Dict[str, int]] = {}  # (rover, start_wp) -> {wp:dist}

        # Goal facts (as global ids) and parsed goal requirements
        self.goal_fact_ids: List[int] = []
        self.goal_soil: List[str] = []  # communicated_soil_data(waypoint)
        self.goal_rock: List[str] = []  # communicated_rock_data(waypoint)
        self.goal_image: List[Tuple[str, str]] = []  # (objective, mode)

        self._build_indices_and_static_data()

    # ---------- Parsing helpers ----------

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
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

    def _fid(self, pred: str, *args: str) -> Optional[int]:
        return self._fact_by_key.get((pred, tuple(args)))

    def _is_true(self, state: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state >> fact_id) & 1) == 1

    # ---------- Model preprocessing ----------

    def _build_indices_and_static_data(self) -> None:
        # Index all facts by (predicate,args)
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_by_key[(pred, tuple(args))] = f.global_id

            # Harvest object names by predicate signature (cheap and robust without type lists)
            if pred == "at" and len(args) == 2:
                self.rovers.add(args[0]); self.waypoints.add(args[1])
            elif pred == "visible" and len(args) == 2:
                self.waypoints.add(args[0]); self.waypoints.add(args[1])
            elif pred == "can_traverse" and len(args) == 3:
                self.rovers.add(args[0]); self.waypoints.add(args[1]); self.waypoints.add(args[2])
            elif pred in ("at_soil_sample", "at_rock_sample", "visited") and len(args) == 1:
                self.waypoints.add(args[0])
            elif pred == "store_of" and len(args) == 2:
                self.stores.add(args[0]); self.rovers.add(args[1])
            elif pred == "on_board" and len(args) == 2:
                self.cameras.add(args[0]); self.rovers.add(args[1])
            elif pred == "supports" and len(args) == 2:
                self.cameras.add(args[0]); self.modes.add(args[1])
            elif pred == "visible_from" and len(args) == 2:
                self.objectives.add(args[0]); self.waypoints.add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self.cameras.add(args[0]); self.objectives.add(args[1])
            elif pred == "at_lander" and len(args) == 2:
                self.landers.add(args[0]); self.waypoints.add(args[1])

        # Build static adjacency and relations from fact keys (presence in fact universe)
        for (pred, args), _gid in self._fact_by_key.items():
            if pred == "visible" and len(args) == 2:
                self.visible[args[0]].add(args[1])
            elif pred == "can_traverse" and len(args) == 3:
                r, a, b = args
                self.can_traverse[r][a].add(b)
            elif pred == "store_of" and len(args) == 2:
                s, r = args
                self.store_of[s] = r
                self.rover_store[r] = s
            elif pred == "at_lander" and len(args) == 2:
                l, w = args
                self.at_lander[l] = w
            elif pred == "calibration_target" and len(args) == 2:
                c, o = args
                self.calibration_target[c] = o
            elif pred == "visible_from" and len(args) == 2:
                o, w = args
                self.visible_from[o].add(w)
            elif pred == "on_board" and len(args) == 2:
                c, r = args
                self.on_board[c] = r
            elif pred == "supports" and len(args) == 2:
                c, m = args
                self.supports[c].add(m)

        # Capability sets are state-dependent predicates but we can identify candidate rovers from fact universe
        # and then check truth in state.

        # Precompute goal list
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for nm in goal_fact_names:
            pred, args = self._parse_fact_name(nm)
            if pred is None:
                continue
            gid = self._fact_by_key.get((pred, tuple(args)))
            if gid is not None:
                self.goal_fact_ids.append(gid)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soil.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rock.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_image.append((args[0], args[1]))

        # Lander visibility target set (depends only on at_lander and visible)
        if self.at_lander:
            # If multiple landers exist, any works (goals require communicate with some lander).
            lander_wps = set(self.at_lander.values())
            for w in self.waypoints:
                for lw in lander_wps:
                    if lw in self.visible.get(w, set()):
                        self.waypoints_visible_to_lander.add(w)

    # ---------- Distance computation ----------

    def _dist_map(self, rover: str, start_wp: str) -> Dict[str, int]:
        key = (rover, start_wp)
        cached = self._traverse_dist_cache.get(key)
        if cached is not None:
            return cached

        # BFS on can_traverse edges for that rover (navigate requires also visible, but in instances
        # traverse edges are usually consistent with visibility; we keep it relaxed and only use can_traverse).
        dist: Dict[str, int] = {start_wp: 0}
        q = deque([start_wp])
        adj = self.can_traverse.get(rover, {})
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, ()):  # neighbors
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        self._traverse_dist_cache[key] = dist
        return dist

    def _min_dist(self, rover: str, start_wp: str, targets: Set[str]) -> Optional[int]:
        if not targets:
            return None
        dist = self._dist_map(rover, start_wp)
        best = None
        for t in targets:
            d = dist.get(t)
            if d is None:
                continue
            if best is None or d < best:
                best = d
        return best

    # ---------- Heuristic core ----------

    def __call__(self, node) -> int:
        state = node.state

        # Gather dynamic rover positions and capabilities
        rover_at: Dict[str, str] = {}
        for r in self.rovers:
            # find its current waypoint by checking at[r,wp] facts via scanning known waypoints
            # We avoid explicit repr; instead probe all waypoints. This is O(|rovers|*|waypoints|),
            # but in practice waypoints per rover in these benchmarks is manageable and fact lookup is bitwise.
            # To reduce cost, we early break when found.
            for w in self.waypoints:
                if self._is_true(state, self._fid("at", r, w)):
                    rover_at[r] = w
                    break

        if not rover_at:
            return 0

        soil_capable = {r for r in rover_at if self._is_true(state, self._fid("equipped_for_soil_analysis", r))}
        rock_capable = {r for r in rover_at if self._is_true(state, self._fid("equipped_for_rock_analysis", r))}
        imaging_capable = {r for r in rover_at if self._is_true(state, self._fid("equipped_for_imaging", r))}

        # Store status per rover
        store_empty: Set[str] = set()
        store_full: Set[str] = set()
        for r in rover_at:
            s = self.rover_store.get(r)
            if s is None:
                continue
            if self._is_true(state, self._fid("empty", s)):
                store_empty.add(r)
            if self._is_true(state, self._fid("full", s)):
                store_full.add(r)

        # Helper: minimal cost for a rover to reach communication position
        def comm_cost(r: str, from_wp: str) -> Optional[int]:
            d = self._min_dist(r, from_wp, self.waypoints_visible_to_lander)
            if d is None:
                return None
            # communicating action itself counts as 1 primitive action
            return d + 1

        # Estimate each goal and assign to best rover
        per_rover_assigned: Dict[str, List[int]] = defaultdict(list)

        # --- Soil goals ---
        for p in self.goal_soil:
            if self._is_true(state, self._fid("communicated_soil_data", p)):
                continue

            best_r = None
            best_cost = None

            # If already have analysis for some rover, can skip sampling part
            for r in rover_at:
                have = self._is_true(state, self._fid("have_soil_analysis", r, p))
                if have:
                    ccomm = comm_cost(r, rover_at[r])
                    if ccomm is None:
                        continue
                    if best_cost is None or ccomm < best_cost:
                        best_cost, best_r = ccomm, r

            # Otherwise: need to navigate soil rover to p, ensure empty, sample, then communicate
            for r in soil_capable:
                start = rover_at[r]
                dist_to_p = self._dist_map(r, start).get(p)
                if dist_to_p is None:
                    continue

                # empty_store: if already empty => 0; else if full => 1 drop; unknown => assume 0 (relaxed LB)
                empty_cost = 0
                if r in store_full and r not in store_empty:
                    empty_cost = 1

                # sample_soil is 1 action
                sample_cost = 1

                # communicate from some comm waypoint
                ccomm = comm_cost(r, p)
                if ccomm is None:
                    continue

                total = dist_to_p + empty_cost + sample_cost + ccomm
                if best_cost is None or total < best_cost:
                    best_cost, best_r = total, r

            if best_cost is None:
                continue
            per_rover_assigned[best_r].append(best_cost)

        # --- Rock goals ---
        for p in self.goal_rock:
            if self._is_true(state, self._fid("communicated_rock_data", p)):
                continue

            best_r = None
            best_cost = None

            for r in rover_at:
                have = self._is_true(state, self._fid("have_rock_analysis", r, p))
                if have:
                    ccomm = comm_cost(r, rover_at[r])
                    if ccomm is None:
                        continue
                    if best_cost is None or ccomm < best_cost:
                        best_cost, best_r = ccomm, r

            for r in rock_capable:
                start = rover_at[r]
                dist_to_p = self._dist_map(r, start).get(p)
                if dist_to_p is None:
                    continue

                empty_cost = 0
                if r in store_full and r not in store_empty:
                    empty_cost = 1

                sample_cost = 1
                ccomm = comm_cost(r, p)
                if ccomm is None:
                    continue

                total = dist_to_p + empty_cost + sample_cost + ccomm
                if best_cost is None or total < best_cost:
                    best_cost, best_r = total, r

            if best_cost is None:
                continue
            per_rover_assigned[best_r].append(best_cost)

        # --- Image goals ---
        for (o, m) in self.goal_image:
            if self._is_true(state, self._fid("communicated_image_data", o, m)):
                continue

            best_r = None
            best_cost = None

            # If have_image already, only communication needed
            for r in rover_at:
                if self._is_true(state, self._fid("have_image", r, o, m)):
                    ccomm = comm_cost(r, rover_at[r])
                    if ccomm is None:
                        continue
                    if best_cost is None or ccomm < best_cost:
                        best_cost, best_r = ccomm, r

            # Otherwise, need take_image (and likely calibrate)
            # Choose a feasible (camera on rover, supports mode, calibration target reachable) combination.
            target_wps = self.visible_from.get(o, set())
            if not target_wps:
                # No waypoint can see objective => skip
                if best_cost is not None:
                    per_rover_assigned[best_r].append(best_cost)
                continue

            for r in imaging_capable:
                start = rover_at[r]
                # cameras on this rover that support mode
                cams = [c for c, rr in self.on_board.items() if rr == r and m in self.supports.get(c, set())]
                if not cams:
                    continue

                distmap = self._dist_map(r, start)

                # Communication after image: from any comm-visible waypoint
                # (we approximate as from the image-taking waypoint)
                for w in target_wps:
                    dw = distmap.get(w)
                    if dw is None:
                        continue

                    # Need calibrate + take_image.
                    # Calibrate requires reaching some waypoint wcal where visible_from(target_obj,wcal)
                    # for the camera's calibration_target.
                    for c in cams:
                        cal_obj = self.calibration_target.get(c)
                        if cal_obj is None:
                            continue
                        cal_wps = self.visible_from.get(cal_obj, set())
                        if not cal_wps:
                            continue

                        dcal = self._min_dist(r, start, cal_wps)
                        if dcal is None:
                            continue

                        # Lower bound skeleton:
                        # navigate to cal wp (dcal) + calibrate (1)
                        # then navigate to image wp (dw) + take_image (1)
                        # NOTE: this overcounts navigation because after calibration we are not necessarily at start.
                        # For a lower bound we should *under*count, so we take min(dcal + 1 + min_dist(cal_wp,w),
                        # but we approximate by max(dcal, dw) + 2 (can't do both without at least the larger travel).
                        # This is a safe-ish lower bound on moves in a graph from a single start.
                        nav_lb = max(dcal, dw)
                        img_actions = 2  # calibrate + take_image

                        ccomm = comm_cost(r, w)
                        if ccomm is None:
                            continue

                        total = nav_lb + img_actions + ccomm
                        if best_cost is None or total < best_cost:
                            best_cost, best_r = total, r

            if best_cost is None:
                continue
            per_rover_assigned[best_r].append(best_cost)

        # Aggregate: sum of max assigned per rover (stronger than max overall; avoids summing all goals)
        h = 0
        for r, costs in per_rover_assigned.items():
            if costs:
                h += max(costs)

        return int(h)
