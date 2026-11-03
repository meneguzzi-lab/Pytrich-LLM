from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

# Pytrich import (works in the evaluation environment)
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic2(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design goals
    ------------
    This heuristic is meant to be *fast* and to provide a *useful lower bound* for
    hierarchical search in Pytrich.

    Key rover-domain observations exploited
    --------------------------------------
    1) All top-level goals are of the form:
         communicated_soil_data(waypoint)
         communicated_rock_data(waypoint)
         communicated_image_data(objective, mode)
       So every remaining goal requires, at minimum, one communication action.

    2) To communicate soil/rock data for waypoint p, the rover must have the analysis
       (have_soil_analysis / have_rock_analysis), which in turn requires visiting p and
       sampling (plus possibly emptying store).

    3) To communicate image data (o,m), a rover must have the image. Taking an image
       requires being at some waypoint w with visible_from(o,w) and having a suitable
       camera on-board (supports mode) that is calibrated; calibration itself requires
       a waypoint where the calibration target objective is visible.

    4) Navigation cost dominates. In these instances, traversability edges are given by
       can_traverse(rover, a, b). We can precompute shortest path distances per rover
       over these edges and use them as an admissible lower bound on the number of
       navigate steps needed.

    Heuristic definition (lower bound)
    ---------------------------------
    We compute a sum of independent *per-goal* lower bounds (then cap with a max-based
    component for safety against overcounting). Each per-goal bound is:

      soil goal at waypoint p:
        if already communicated: 0
        else:
          min over rovers r that can do soil:
            dist(r_at, p) + 1(sample_soil) + minCommCost
        where minCommCost is 1 if from p some lander-visible comm waypoint exists,
        or 1 if already at such place etc. We conservatively use 1 when possible,
        otherwise 2 as a coarse bound (navigate-to-comm + communicate).

      rock goal similarly.

      image goal (o,m):
        if already communicated: 0
        else:
          min over rovers r with imaging and some camera c on r supporting m:
            [optional calibrate] + dist(to imaging waypoint) + 1(take_image) + comm
        For admissibility, we treat calibrate cost as 0 if already calibrated for
        some suitable camera, else at least 1 (calibrate action), ignoring navigation
        to calibration point (relaxation).

    Finally, we combine:
      h = max( max_per_goal, ceil(sum_per_goal * alpha) ) with alpha in (0,1]
    Using alpha<1 reduces risk of overcounting due to goal interactions.

    This heuristic is not intended to be admissible for optimality proofs in general,
    but it is a strong, domain-guiding estimate that works well in practice for
    reducing node expansions. If strict admissibility is required, set alpha=0 and
    rely solely on max_per_goal.
    """

    def __init__(self, model, rover_gpt52_heuristic_2: str = "rover_gpt52_heuristic_2"):
        super().__init__(model, rover_gpt52_heuristic_2)
        self.model = model

        # --- Pre-index facts by predicate for fast state bit checks ---
        self.facts_by_pred: Dict[str, List] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self.facts_by_pred.setdefault(pred, []).append((f, args))

        # --- Extract objects we care about ---
        self.rovers: Set[str] = set()
        self.waypoints: Set[str] = set()
        self.landers: Set[str] = set()
        self.cameras: Set[str] = set()
        self.objectives: Set[str] = set()
        self.modes: Set[str] = set()

        # The model doesn't expose typing directly; infer from fact arguments.
        # We opportunistically gather from common predicates.
        for pred, entries in self.facts_by_pred.items():
            for _, args in entries:
                if pred == "at" and len(args) == 2:
                    self.rovers.add(args[0])
                    self.waypoints.add(args[1])
                elif pred == "at_lander" and len(args) == 2:
                    self.landers.add(args[0])
                    self.waypoints.add(args[1])
                elif pred == "on_board" and len(args) == 2:
                    self.cameras.add(args[0])
                    self.rovers.add(args[1])
                elif pred == "visible" and len(args) == 2:
                    self.waypoints.add(args[0])
                    self.waypoints.add(args[1])
                elif pred == "visible_from" and len(args) == 2:
                    self.objectives.add(args[0])
                    self.waypoints.add(args[1])
                elif pred == "supports" and len(args) == 2:
                    self.cameras.add(args[0])
                    self.modes.add(args[1])
                elif pred == "calibration_target" and len(args) == 2:
                    self.cameras.add(args[0])
                    self.objectives.add(args[1])

        # --- Capability maps (static from init facts; treated as invariant) ---
        self.equipped_soil: Set[str] = set()
        self.equipped_rock: Set[str] = set()
        self.equipped_img: Set[str] = set()
        for f, args in self.facts_by_pred.get("equipped_for_soil_analysis", []):
            if len(args) == 1:
                self.equipped_soil.add(args[0])
        for f, args in self.facts_by_pred.get("equipped_for_rock_analysis", []):
            if len(args) == 1:
                self.equipped_rock.add(args[0])
        for f, args in self.facts_by_pred.get("equipped_for_imaging", []):
            if len(args) == 1:
                self.equipped_img.add(args[0])

        # --- Cameras on each rover + supported modes + calibration target objective ---
        self.cameras_on_rover: Dict[str, Set[str]] = {r: set() for r in self.rovers}
        for f, args in self.facts_by_pred.get("on_board", []):
            if len(args) == 2:
                cam, rov = args
                self.cameras_on_rover.setdefault(rov, set()).add(cam)

        self.camera_supports: Dict[str, Set[str]] = {}
        for f, args in self.facts_by_pred.get("supports", []):
            if len(args) == 2:
                cam, mode = args
                self.camera_supports.setdefault(cam, set()).add(mode)

        self.camera_calib_target: Dict[str, str] = {}
        for f, args in self.facts_by_pred.get("calibration_target", []):
            if len(args) == 2:
                cam, obj = args
                self.camera_calib_target[cam] = obj

        # --- Lander positions (static) ---
        self.lander_at: Dict[str, str] = {}
        for f, args in self.facts_by_pred.get("at_lander", []):
            if len(args) == 2:
                self.lander_at[args[0]] = args[1]

        # --- Visibility graph between waypoints (static) ---
        self.visible_wp: Dict[str, Set[str]] = {w: set() for w in self.waypoints}
        for f, args in self.facts_by_pred.get("visible", []):
            if len(args) == 2:
                a, b = args
                self.visible_wp.setdefault(a, set()).add(b)

        # --- Communication-relevant waypoint sets ---
        # A waypoint w is a valid comm-point if exists lander l at wl with visible(w,wl).
        self.comm_points: Set[str] = set()
        for l, wl in self.lander_at.items():
            for w, neigh in self.visible_wp.items():
                if wl in neigh:
                    self.comm_points.add(w)

        # --- Precompute per-rover navigation graphs from can_traverse ---
        # We use directed edges as given.
        self.traverse_graph: Dict[str, Dict[str, List[str]]] = {r: {} for r in self.rovers}
        for f, args in self.facts_by_pred.get("can_traverse", []):
            if len(args) == 3:
                r, a, b = args
                self.traverse_graph.setdefault(r, {}).setdefault(a, []).append(b)

        # Precompute all-pairs shortest path distances per rover with BFS from each waypoint.
        self._dist_cache: Dict[Tuple[str, str, str], int] = {}
        self._build_all_pairs_distances()

        # Goals parsed once
        self.goal_items = self._parse_goals()

        # Weighting: alpha<1 reduces overcounting; set to 0 for strict max-only bound.
        self.alpha = 0.6

    # -------------------------- core API --------------------------

    def __call__(self, node) -> int:
        state = node.state

        # Quick goal check: if already at goal state (and tn empty), heuristic 0
        # (We cannot reliably check tn here; model.goal_reached uses tn, but node may include it.)
        if (state & self.model.goals) == self.model.goals:
            return 0

        per_goal_costs: List[int] = []

        # Precompute rover positions from current state (dynamic)
        rover_pos = self._current_rover_positions(state)

        # Precompute calibrated cameras in state
        calibrated_pairs = self._current_calibrated_pairs(state)  # (cam, rover)

        # For each goal, compute a cheap-to-evaluate lower bound
        for g in self.goal_items:
            kind = g[0]
            if kind == "soil":
                wp = g[1]
                if self._is_true(state, "communicated_soil_data", [wp]):
                    per_goal_costs.append(0)
                else:
                    per_goal_costs.append(self._lb_soil(state, wp, rover_pos))
            elif kind == "rock":
                wp = g[1]
                if self._is_true(state, "communicated_rock_data", [wp]):
                    per_goal_costs.append(0)
                else:
                    per_goal_costs.append(self._lb_rock(state, wp, rover_pos))
            elif kind == "image":
                obj, mode = g[1], g[2]
                if self._is_true(state, "communicated_image_data", [obj, mode]):
                    per_goal_costs.append(0)
                else:
                    per_goal_costs.append(self._lb_image(state, obj, mode, rover_pos, calibrated_pairs))

        if not per_goal_costs:
            return 0

        max_part = max(per_goal_costs)
        sum_part = sum(per_goal_costs)
        blended = int((sum_part * self.alpha) + 0.999999)
        return max(max_part, blended)

    # -------------------------- lower bounds --------------------------

    def _lb_soil(self, state: int, wp: str, rover_pos: Dict[str, str]) -> int:
        # If any rover already has analysis for wp, skip sampling
        already_have = False
        for r in self.rovers:
            if self._is_true(state, "have_soil_analysis", [r, wp]):
                already_have = True
                break

        best = 10**9
        for r in self.equipped_soil:
            start = rover_pos.get(r)
            if start is None:
                continue
            nav = 0 if already_have else self._dist(r, start, wp)
            if nav >= 10**8:
                continue
            sample_cost = 0 if already_have else 1
            comm_cost = self._lb_comm_from(wp, r, rover_pos)
            best = min(best, nav + sample_cost + comm_cost)

        return 0 if best == 10**9 else best

    def _lb_rock(self, state: int, wp: str, rover_pos: Dict[str, str]) -> int:
        already_have = False
        for r in self.rovers:
            if self._is_true(state, "have_rock_analysis", [r, wp]):
                already_have = True
                break

        best = 10**9
        for r in self.equipped_rock:
            start = rover_pos.get(r)
            if start is None:
                continue
            nav = 0 if already_have else self._dist(r, start, wp)
            if nav >= 10**8:
                continue
            sample_cost = 0 if already_have else 1
            comm_cost = self._lb_comm_from(wp, r, rover_pos)
            best = min(best, nav + sample_cost + comm_cost)

        return 0 if best == 10**9 else best

    def _lb_image(
        self,
        state: int,
        obj: str,
        mode: str,
        rover_pos: Dict[str, str],
        calibrated_pairs: Set[Tuple[str, str]],
    ) -> int:
        # If any rover already has the image, only communication is needed.
        for r in self.rovers:
            if self._is_true(state, "have_image", [r, obj, mode]):
                # Need to communicate from some comm point; use rover's current pos
                start = rover_pos.get(r)
                if start is None:
                    continue
                return self._lb_comm_from(start, r, rover_pos, treat_from_is_wp=True)

        # Otherwise, must take image.
        # Candidate imaging waypoints are those with visible_from(obj, w)
        img_wps = self._visible_from_waypoints(obj)
        if not img_wps:
            return 10**6

        best = 10**9
        for r in self.equipped_img:
            start = rover_pos.get(r)
            if start is None:
                continue

            # Find at least one camera on rover supporting mode
            cams = self.cameras_on_rover.get(r, set())
            possible_cams = [c for c in cams if mode in self.camera_supports.get(c, set())]
            if not possible_cams:
                continue

            # Calibration lower bound: if any possible cam is already calibrated on that rover -> 0 else 1
            need_calib = 1
            for c in possible_cams:
                if (c, r) in calibrated_pairs:
                    need_calib = 0
                    break

            # Navigation to some imaging waypoint
            for w in img_wps:
                nav = self._dist(r, start, w)
                if nav >= 10**8:
                    continue
                take = 1
                comm_cost = self._lb_comm_from(w, r, rover_pos, treat_from_is_wp=True)
                best = min(best, need_calib + nav + take + comm_cost)

        return 0 if best == 10**9 else best

    def _lb_comm_from(self, from_wp: str, rover: str, rover_pos: Dict[str, str], treat_from_is_wp: bool = True) -> int:
        """Lower bound on cost to communicate some data.

        Communication actions require rover at some waypoint w1 such that visible(w1, lander_wp).
        - If already at a comm point: at least 1 (communicate action)
        - Else: at least dist(to nearest comm point) + 1

        We intentionally ignore channel_free/available as they are usually true and/or cheap.
        """
        if not treat_from_is_wp:
            # Not used currently
            pass

        # If from_wp itself is a comm point, just 1
        if from_wp in self.comm_points:
            return 1

        # Otherwise move to nearest comm point
        best_nav = 10**9
        for cp in self.comm_points:
            d = self._dist(rover, from_wp, cp)
            if d < best_nav:
                best_nav = d

        if best_nav >= 10**8:
            # If unreachable, still give a coarse positive heuristic
            return 2
        return best_nav + 1

    # -------------------------- utilities --------------------------

    def _parse_goals(self):
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        items = []
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "communicated_soil_data" and len(args) == 1:
                items.append(("soil", args[0]))
            elif pred == "communicated_rock_data" and len(args) == 1:
                items.append(("rock", args[0]))
            elif pred == "communicated_image_data" and len(args) == 2:
                items.append(("image", args[0], args[1]))
        return items

    def _current_rover_positions(self, state: int) -> Dict[str, str]:
        pos = {}
        for f, args in self.facts_by_pred.get("at", []):
            if len(args) != 2:
                continue
            r, w = args
            if r in self.rovers and ((state >> f.global_id) & 1):
                pos[r] = w
        return pos

    def _current_calibrated_pairs(self, state: int) -> Set[Tuple[str, str]]:
        res = set()
        for f, args in self.facts_by_pred.get("calibrated", []):
            if len(args) != 2:
                continue
            cam, r = args
            if ((state >> f.global_id) & 1):
                res.add((cam, r))
        return res

    def _visible_from_waypoints(self, obj: str) -> List[str]:
        wps = []
        for f, args in self.facts_by_pred.get("visible_from", []):
            if len(args) == 2 and args[0] == obj:
                wps.append(args[1])
        return wps

    def _dist(self, rover: str, a: str, b: str) -> int:
        if a == b:
            return 0
        return self._dist_cache.get((rover, a, b), 10**8)

    def _build_all_pairs_distances(self):
        # BFS from each waypoint for each rover
        for r in self.rovers:
            graph = self.traverse_graph.get(r, {})
            if not graph:
                continue
            # nodes are waypoints mentioned in graph keys/values
            nodes = set(graph.keys())
            for vs in graph.values():
                nodes.update(vs)
            for src in nodes:
                dist = {src: 0}
                q = deque([src])
                while q:
                    u = q.popleft()
                    for v in graph.get(u, []):
                        if v not in dist:
                            dist[v] = dist[u] + 1
                            q.append(v)
                for tgt, d in dist.items():
                    self._dist_cache[(r, src, tgt)] = d

    def _is_true(self, state: int, pred: str, args: List[str]) -> bool:
        # Use pre-indexed facts to find the matching grounded fact and then bit-test.
        for f, fargs in self.facts_by_pred.get(pred, []):
            if fargs == args:
                return ((state >> f.global_id) & 1) == 1
        return False

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
