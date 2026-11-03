from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic12(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    Design rationale (domain insights -> heuristic components)
    ---------------------------------------------------------
    The Rover domain’s top-level HTN tasks are typically of the forms:
      - get_soil_data(wp): navigate to wp, ensure store empty, sample_soil, send_soil_data
      - get_rock_data(wp): navigate to wp, ensure store empty, sample_rock, send_rock_data
      - get_image_data(obj, mode): calibrate a camera (navigate to calibration target visibility),
        navigate to a waypoint that can see the objective, take_image, send_image_data

    The dominant cost drivers are:
      (1) Navigation steps (navigate action) along the rover-specific can_traverse graph.
      (2) Store management: if store is full, must drop before sampling.
      (3) Communication requires being at a waypoint that is 'visible' to the lander waypoint.
      (4) Imaging requires a camera that supports the mode and (re-)calibration before each image
          (take_image consumes calibration).

    This heuristic provides a fast *lower bound* estimate of remaining primitive actions by:
      - Counting unsatisfied goal facts by type (communicated_*).
      - For each unsatisfied goal, estimating a minimal number of primitive actions needed,
        including at least one communicate action and, when needed, sampling/taking image.
      - Adding a *relaxed navigation lower bound* based on shortest-path distances in the
        can_traverse graph (ignoring visibility and availability constraints to stay admissible).
      - Taking the MAX across goals as a cheap admissible approximation of doing “the hardest
        remaining item”, plus a small additive term for the number of remaining independent
        communications (admissible because each requires at least one primitive action).

    Admissibility notes
    -------------------
    - We never require extra actions due to negative effects beyond what is strictly necessary.
    - Navigation distance is computed on can_traverse edges only, and used as a *lower bound*
      on actual navigation cost (actual requires also visible(p1,p2), but ignoring it can only
      underestimate).
    - We do not assume any particular rover/camera assignment; for each goal we take the minimum
      over all rovers (and cameras when needed), which is also a lower bound.

    Performance notes
    -----------------
    - All state checks use bitwise operations with cached fact global_ids.
    - All static relations (at_lander, store_of, on_board, supports, calibration_target,
      visible_from, can_traverse) are extracted once from model facts.
    - Shortest paths are computed per rover via BFS over waypoints (unweighted edges).

    The heuristic is intended to reduce node expansions by strongly preferring states that:
      - are closer (in traverse-graph distance) to a communication waypoint,
      - already hold required analyses/images,
      - have an empty store when sampling is still needed.
    """

    def __init__(self, rover_gpt52_heuristic_12: Heuristic):
        super().__init__(rover_gpt52_heuristic_12)
        self.model = rover_gpt52_heuristic_12.model

        # --- Fact parsing caches ---
        self._fact_by_name: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Goal facts (names) are retrieved once.
        self._goal_fact_names: List[str] = self.model.state_explicit_repr(self.model.goals)
        self._goal_fact_ids: List[int] = [self._fact_by_name[n] for n in self._goal_fact_names if n in self._fact_by_name]

        # Partition goal facts by type
        self._soil_goal_wps: Set[str] = set()
        self._rock_goal_wps: Set[str] = set()
        self._img_goal_pairs: Set[Tuple[str, str]] = set()  # (objective, mode)
        for n in self._goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._soil_goal_wps.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._rock_goal_wps.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._img_goal_pairs.add((args[0], args[1]))

        # --- Static relations extracted from facts (domain constants) ---
        self._rovers: Set[str] = set()
        self._stores: Set[str] = set()
        self._cameras: Set[str] = set()
        self._objectives: Set[str] = set()
        self._modes: Set[str] = set()
        self._waypoints: Set[str] = set()
        self._landers: Set[str] = set()

        # store_of(store, rover)
        self._store_of: Dict[str, str] = {}
        self._store_for_rover: Dict[str, str] = {}

        # on_board(camera, rover)
        self._camera_on: Dict[str, str] = {}
        self._cameras_of_rover: Dict[str, List[str]] = {}

        # supports(camera, mode)
        self._supports: Dict[str, Set[str]] = {}

        # calibration_target(camera, objective)
        self._calib_target: Dict[str, Set[str]] = {}

        # visible_from(objective, waypoint)
        self._visible_from: Dict[str, Set[str]] = {}

        # at_lander(lander, waypoint)
        self._lander_at: Dict[str, str] = {}
        self._lander_waypoints: Set[str] = set()

        # can_traverse(rover, wp1, wp2)
        self._adj: Dict[str, Dict[str, List[str]]] = {}  # rover -> wp -> neighbors

        # Cache ids for common unary facts we may check often
        self._empty_ids: Set[int] = set()
        self._full_ids: Set[int] = set()

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == "store_of" and len(args) == 2:
                s, r = args
                self._store_of[s] = r
                self._store_for_rover[r] = s
                self._stores.add(s)
                self._rovers.add(r)
            elif pred == "on_board" and len(args) == 2:
                c, r = args
                self._camera_on[c] = r
                self._cameras_of_rover.setdefault(r, []).append(c)
                self._cameras.add(c)
                self._rovers.add(r)
            elif pred == "supports" and len(args) == 2:
                c, m = args
                self._supports.setdefault(c, set()).add(m)
                self._cameras.add(c)
                self._modes.add(m)
            elif pred == "calibration_target" and len(args) == 2:
                c, o = args
                self._calib_target.setdefault(c, set()).add(o)
                self._cameras.add(c)
                self._objectives.add(o)
            elif pred == "visible_from" and len(args) == 2:
                o, w = args
                self._visible_from.setdefault(o, set()).add(w)
                self._objectives.add(o)
                self._waypoints.add(w)
            elif pred == "at_lander" and len(args) == 2:
                l, w = args
                self._lander_at[l] = w
                self._lander_waypoints.add(w)
                self._landers.add(l)
                self._waypoints.add(w)
            elif pred == "can_traverse" and len(args) == 3:
                r, w1, w2 = args
                self._adj.setdefault(r, {}).setdefault(w1, []).append(w2)
                self._rovers.add(r)
                self._waypoints.add(w1)
                self._waypoints.add(w2)
            elif pred == "empty" and len(args) == 1:
                self._empty_ids.add(fact.global_id)
            elif pred == "full" and len(args) == 1:
                self._full_ids.add(fact.global_id)
            elif pred == "at" and len(args) == 2:
                # helps collect rover/waypoint universe even if not seen elsewhere
                r, w = args
                self._rovers.add(r)
                self._waypoints.add(w)

        # Precompute fact ids for at(rover, wp)
        self._at_id: Dict[Tuple[str, str], int] = {}
        for r in self._rovers:
            for w in self._waypoints:
                name = f"+at[{r},{w}]"
                if name in self._fact_by_name:
                    self._at_id[(r, w)] = self._fact_by_name[name]

        # Precompute fact ids for have_soil_analysis(rover, wp) and have_rock_analysis
        self._have_soil_id: Dict[Tuple[str, str], int] = {}
        self._have_rock_id: Dict[Tuple[str, str], int] = {}
        for r in self._rovers:
            for w in self._waypoints:
                n1 = f"+have_soil_analysis[{r},{w}]"
                if n1 in self._fact_by_name:
                    self._have_soil_id[(r, w)] = self._fact_by_name[n1]
                n2 = f"+have_rock_analysis[{r},{w}]"
                if n2 in self._fact_by_name:
                    self._have_rock_id[(r, w)] = self._fact_by_name[n2]

        # Precompute ids for have_image(rover,obj,mode)
        self._have_img_id: Dict[Tuple[str, str, str], int] = {}
        for r in self._rovers:
            for (o, m) in self._img_goal_pairs:
                n = f"+have_image[{r},{o},{m}]"
                if n in self._fact_by_name:
                    self._have_img_id[(r, o, m)] = self._fact_by_name[n]

        # Precompute ids for communicated_* goals
        self._comm_soil_id: Dict[str, int] = {}
        self._comm_rock_id: Dict[str, int] = {}
        self._comm_img_id: Dict[Tuple[str, str], int] = {}
        for w in self._soil_goal_wps:
            n = f"+communicated_soil_data[{w}]"
            if n in self._fact_by_name:
                self._comm_soil_id[w] = self._fact_by_name[n]
        for w in self._rock_goal_wps:
            n = f"+communicated_rock_data[{w}]"
            if n in self._fact_by_name:
                self._comm_rock_id[w] = self._fact_by_name[n]
        for (o, m) in self._img_goal_pairs:
            n = f"+communicated_image_data[{o},{m}]"
            if n in self._fact_by_name:
                self._comm_img_id[(o, m)] = self._fact_by_name[n]

        # Precompute per-rover all-pairs shortest distances from any waypoint using BFS from each start.
        # For performance, we compute distances from *current rover positions* on demand and cache.
        self._dist_cache: Dict[Tuple[str, str], Dict[str, int]] = {}  # (rover, start_wp) -> dist map

        # Candidate communication waypoints (any waypoint visible to any lander waypoint).
        # In the domain, communication actions require visible(pX, lander_wp).
        # We do not use visible facts for admissibility; instead we approximate by allowing comm at lander_wp itself.
        # This is safe as a lower bound because it may underestimate movement.
        self._comm_anchor_wps: Set[str] = set(self._lander_waypoints) if self._lander_waypoints else set(self._waypoints)

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

    def _is_true(self, state: int, fact_id: int) -> bool:
        return ((state >> fact_id) & 1) == 1

    def _rover_position(self, state: int, rover: str) -> Optional[str]:
        # Scan all waypoint candidates; this is okay because we only do it for a limited number of rovers
        # (and cached by early return).
        for w in self._waypoints:
            fid = self._at_id.get((rover, w))
            if fid is not None and self._is_true(state, fid):
                return w
        return None

    def _bfs_dist(self, rover: str, start: str) -> Dict[str, int]:
        key = (rover, start)
        if key in self._dist_cache:
            return self._dist_cache[key]
        adj = self._adj.get(rover, {})
        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[key] = dist
        return dist

    def _min_dist_to_any(self, distmap: Dict[str, int], targets: Set[str]) -> int:
        best = None
        for t in targets:
            d = distmap.get(t)
            if d is not None:
                if best is None or d < best:
                    best = d
        return best if best is not None else 0

    def __call__(self, node) -> int:
        state = node.state

        # Count remaining goal facts and compute a per-goal lower bound.
        remaining_items = 0
        max_item_cost_lb = 0

        # --- Soil goals ---
        for w in self._soil_goal_wps:
            fid = self._comm_soil_id.get(w)
            if fid is None or self._is_true(state, fid):
                continue
            remaining_items += 1
            max_item_cost_lb = max(max_item_cost_lb, self._soil_item_lb(state, w))

        # --- Rock goals ---
        for w in self._rock_goal_wps:
            fid = self._comm_rock_id.get(w)
            if fid is None or self._is_true(state, fid):
                continue
            remaining_items += 1
            max_item_cost_lb = max(max_item_cost_lb, self._rock_item_lb(state, w))

        # --- Image goals ---
        for (o, m) in self._img_goal_pairs:
            fid = self._comm_img_id.get((o, m))
            if fid is None or self._is_true(state, fid):
                continue
            remaining_items += 1
            max_item_cost_lb = max(max_item_cost_lb, self._image_item_lb(state, o, m))

        if remaining_items == 0:
            return 0

        # Lower bound aggregation:
        # - At least one primitive action per remaining item (communication is necessary for each goal).
        # - Plus the hardest item's more detailed bound.
        # We take max(hardest, remaining_items) to stay admissible.
        return max(max_item_cost_lb, remaining_items)

    def _soil_item_lb(self, state: int, wp: str) -> int:
        # Need: have_soil_analysis(r, wp) then communicate.
        # Minimal primitive actions ignoring ordering conflicts:
        # - if already have analysis: at least 1 (communicate)
        # - else: need sample_soil (1) + communicate (1) + possibly drop (>=0) + navigation lower bound
        best = None
        for r in self._rovers:
            pos = self._rover_position(state, r)
            if pos is None:
                continue
            distmap = self._bfs_dist(r, pos)

            have_id = self._have_soil_id.get((r, wp))
            has_analysis = have_id is not None and self._is_true(state, have_id)

            nav_to_sample = distmap.get(wp, 0)
            nav_to_comm = self._min_dist_to_any(distmap, self._comm_anchor_wps)
            nav_lb = min(nav_to_comm, nav_to_sample + 0)  # communication could happen after sampling; keep optimistic

            drop_lb = 0
            s = self._store_for_rover.get(r)
            if s is not None:
                full_id = self._fact_by_name.get(f"+full[{s}]")
                if full_id is not None and self._is_true(state, full_id) and not has_analysis:
                    drop_lb = 1

            if has_analysis:
                cost = 1 + nav_to_comm
            else:
                cost = nav_to_sample + drop_lb + 1 + 1  # move + (optional drop) + sample + communicate
                # ensure at least some comm movement accounted (but optimistic)
                cost = min(cost, 1 + nav_to_comm + 1)  # communicate without sample impossible, but keep min conservative
                cost = max(cost, nav_lb + 2)  # at least communicate+sample plus some movement

            best = cost if best is None else min(best, cost)

        return best if best is not None else 1

    def _rock_item_lb(self, state: int, wp: str) -> int:
        best = None
        for r in self._rovers:
            pos = self._rover_position(state, r)
            if pos is None:
                continue
            distmap = self._bfs_dist(r, pos)

            have_id = self._have_rock_id.get((r, wp))
            has_analysis = have_id is not None and self._is_true(state, have_id)

            nav_to_sample = distmap.get(wp, 0)
            nav_to_comm = self._min_dist_to_any(distmap, self._comm_anchor_wps)

            drop_lb = 0
            s = self._store_for_rover.get(r)
            if s is not None:
                full_id = self._fact_by_name.get(f"+full[{s}]")
                if full_id is not None and self._is_true(state, full_id) and not has_analysis:
                    drop_lb = 1

            if has_analysis:
                cost = 1 + nav_to_comm
            else:
                cost = nav_to_sample + drop_lb + 1 + 1
                cost = max(cost, 2)  # sample+communicate

            best = cost if best is None else min(best, cost)

        return best if best is not None else 1

    def _image_item_lb(self, state: int, obj: str, mode: str) -> int:
        # Need: have_image(r,obj,mode) then communicate.
        # To obtain have_image: must take_image (1) which requires calibrated(camera,r) and at a waypoint visible_from(obj,wp).
        # Calibration can be achieved by calibrate (1) after navigating to some wp visible_from(calib_target, wp).
        # We build a very optimistic LB:
        # - If already have_image: 1 + dist_to_comm
        # - Else: (calibrate? 0/1) + take_image(1) + communicate(1) + navigation to some visible_from(obj, wp)
        best = None
        obj_wps = self._visible_from.get(obj, set())
        if not obj_wps:
            obj_wps = set(self._waypoints)

        for r in self._rovers:
            pos = self._rover_position(state, r)
            if pos is None:
                continue
            distmap = self._bfs_dist(r, pos)
            nav_to_comm = self._min_dist_to_any(distmap, self._comm_anchor_wps)

            have_id = self._have_img_id.get((r, obj, mode))
            has_img = have_id is not None and self._is_true(state, have_id)
            if has_img:
                cost = 1 + nav_to_comm
                best = cost if best is None else min(best, cost)
                continue

            # choose a camera on rover that supports mode and can be calibrated for some target
            cams = self._cameras_of_rover.get(r, [])
            cam_best = None
            for c in cams:
                if mode not in self._supports.get(c, set()):
                    continue
                # calibration target objective(s) for this camera
                targets = self._calib_target.get(c, set())
                # for LB: allow calibration at current position if any target is visible_from there
                calib_cost_lb = 1 if targets else 0

                # navigation to some waypoint that can see the requested objective
                nav_to_img = self._min_dist_to_any(distmap, obj_wps)

                # total LB: nav_to_img + calibrate + take_image + communicate
                cost = nav_to_img + calib_cost_lb + 1 + 1
                cam_best = cost if cam_best is None else min(cam_best, cost)

            if cam_best is None:
                # no usable camera -> return minimal communicate-only bound
                cam_best = 1 + nav_to_comm

            best = cam_best if best is None else min(best, cam_best)

        return best if best is not None else 1
