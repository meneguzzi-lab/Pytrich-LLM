from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic4(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic features)
    --------------------------------------------------------
    The Rover domain goals are *communicated_* facts. Achieving any such goal typically requires:
      - For soil/rock: navigate to the sample waypoint, (maybe) empty store, sample, then navigate to a
        waypoint that is visible to the lander and communicate.
      - For image: calibrate camera (navigate to a calibration-target-visible waypoint + calibrate),
        navigate to an objective-visible waypoint, take_image, then navigate to a lander-visible waypoint
        and communicate.

    The dominant cost driver is navigation (many waypoints; sparse can_traverse graph; visibility constraint).
    This heuristic therefore:
      1) Precomputes shortest-path distances on the can_traverse graph per rover (unit edge cost).
      2) Precomputes "communication-capable" waypoints: those that are visible to the lander waypoint.
      3) For each missing communicated goal, estimates a *lower bound* on remaining primitive actions:
           - minimal navigation steps + mandatory action count (sample/communicate/take_image/calibrate)
         using optimistic assumptions (ignore store fullness problems; ignore camera calibration persistence;
         pick best rover).

    Admissibility
    -------------
    The heuristic is intended as a lower bound by:
      - Taking the *minimum* over rovers for each goal.
      - Using *shortest-path* distances.
      - Ignoring negative interactions/resources (store full, calibration resets, availability/channel_free).
      - Summing per-goal lower bounds is generally NOT admissible with shared actions; hence we use
        a conservative aggregation: take the maximum per-goal estimate (a lower bound on finishing all).

    Task-network awareness
    ----------------------
    Pytrich evaluates both state and remaining task network. This heuristic also gives a small, safe
    lower bound based on the number of remaining primitive operators in the task network (each costs >=1).
    This does not overestimate because each remaining primitive operator must be executed at least once.

    Efficiency
    ----------
    All heavy computations are done once in __init__:
      - parse facts into indices
      - build rover-specific adjacency and BFS distances
      - build lander waypoint and comm-set
    Per state call is then mostly bit checks and dictionary lookups.
    """

    def __init__(self, model, rover_gpt52_heuristic_4: str = "rover_gpt52_heuristic_4"):
        super().__init__(model, rover_gpt52_heuristic_4)
        self.model = model

        # --- Fact indexing helpers ---
        self._fact_by_pred: Dict[str, List[Tuple[int, List[str]]]] = {}
        self._fact_id_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_by_pred.setdefault(pred, []).append((f.global_id, args))
            self._fact_id_by_key[(pred, tuple(args))] = f.global_id

        # --- Extract static structures ---
        self.rovers: List[str] = self._collect_objects_from_facts("available", 1)
        if not self.rovers:
            # fallback: infer rovers from at[rover,wp]
            self.rovers = self._collect_objects_from_facts("at", 2, obj_index=0)

        # Locations: at[rover, waypoint]
        self._at_fact_ids: Dict[Tuple[str, str], int] = {}
        for gid, args in self._fact_by_pred.get("at", []):
            if len(args) == 2:
                self._at_fact_ids[(args[0], args[1])] = gid

        # can_traverse[rover, from, to]
        self._adj: Dict[str, Dict[str, List[str]]] = {r: {} for r in self.rovers}
        for _gid, args in self._fact_by_pred.get("can_traverse", []):
            if len(args) != 3:
                continue
            r, a, b = args
            if r in self._adj:
                self._adj[r].setdefault(a, []).append(b)

        # lander position(s): at_lander[lander, waypoint]
        self.lander_waypoints: List[str] = []
        for _gid, args in self._fact_by_pred.get("at_lander", []):
            if len(args) == 2:
                self.lander_waypoints.append(args[1])
        # Most instances have exactly one lander
        self.lander_wp: Optional[str] = self.lander_waypoints[0] if self.lander_waypoints else None

        # visibility between waypoints: visible[w1,w2]
        self._visible_pairs: set[Tuple[str, str]] = set()
        for _gid, args in self._fact_by_pred.get("visible", []):
            if len(args) == 2:
                self._visible_pairs.add((args[0], args[1]))

        # Communication-capable waypoints: any w such that visible[w, lander_wp]
        self.comm_waypoints: List[str] = []
        if self.lander_wp is not None:
            self.comm_waypoints = [w1 for (w1, w2) in self._visible_pairs if w2 == self.lander_wp]

        # objective visibility: visible_from[obj, waypoint]
        self._visible_from: Dict[str, List[str]] = {}
        for _gid, args in self._fact_by_pred.get("visible_from", []):
            if len(args) == 2:
                self._visible_from.setdefault(args[0], []).append(args[1])

        # camera on rover: on_board[camera, rover]
        self._cameras_on_rover: Dict[str, List[str]] = {r: [] for r in self.rovers}
        for _gid, args in self._fact_by_pred.get("on_board", []):
            if len(args) == 2:
                cam, r = args
                if r in self._cameras_on_rover:
                    self._cameras_on_rover[r].append(cam)

        # calibration_target[camera, objective]
        self._calib_target: Dict[str, str] = {}
        for _gid, args in self._fact_by_pred.get("calibration_target", []):
            if len(args) == 2:
                self._calib_target[args[0]] = args[1]

        # supports[camera, mode]
        self._supports: Dict[str, set[str]] = {}
        for _gid, args in self._fact_by_pred.get("supports", []):
            if len(args) == 2:
                self._supports.setdefault(args[0], set()).add(args[1])

        # Equipment
        self._soil_capable = set(self._collect_objects_from_facts("equipped_for_soil_analysis", 1))
        self._rock_capable = set(self._collect_objects_from_facts("equipped_for_rock_analysis", 1))
        self._img_capable = set(self._collect_objects_from_facts("equipped_for_imaging", 1))

        # --- Precompute BFS distances per rover from every waypoint it can reach ---
        self._dist: Dict[str, Dict[str, Dict[str, int]]] = {r: {} for r in self.rovers}
        for r in self.rovers:
            # collect all nodes seen in adjacency
            nodes = set(self._adj.get(r, {}).keys())
            for a in self._adj.get(r, {}):
                for b in self._adj[r][a]:
                    nodes.add(b)
            for src in nodes:
                self._dist[r][src] = self._bfs_distances(r, src)

        # Cache goal facts (explicit names -> parsed)
        self._goal_facts: List[Tuple[str, List[str], int]] = []  # (pred,args, gid)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            gid = self._fact_id_by_key.get((pred, tuple(args)))
            if gid is not None:
                self._goal_facts.append((pred, args, gid))

    # ----------------- Core heuristic -----------------

    def __call__(self, node) -> int:
        state = node.state
        tn = getattr(node, "task_network", [])

        # Lower bound from remaining primitive tasks in the task network.
        # (Each operator costs at least 1 to execute.)
        primitive_lb = 0
        for t in tn:
            # Operators have 'pos_precons' attribute in this framework
            if hasattr(t, "pos_precons") and hasattr(t, "add_effects"):
                primitive_lb += 1

        # If goals already satisfied, heuristic is just remaining primitive tasks lower bound.
        # (Often zero at goal.)
        # Note: model.goal_reached also checks tn empty; we only check goal facts here.
        all_goals_true = True
        for _pred, _args, gid in self._goal_facts:
            if ((state >> gid) & 1) == 0:
                all_goals_true = False
                break
        if all_goals_true:
            return primitive_lb

        # Compute max per missing communicated goal (safe aggregation).
        max_goal_cost = 0
        for pred, args, gid in self._goal_facts:
            if ((state >> gid) & 1) == 1:
                continue

            if pred == "communicated_soil_data" and len(args) == 1:
                wp = args[0]
                est = self._estimate_soil(wp, state)
            elif pred == "communicated_rock_data" and len(args) == 1:
                wp = args[0]
                est = self._estimate_rock(wp, state)
            elif pred == "communicated_image_data" and len(args) == 2:
                obj, mode = args
                est = self._estimate_image(obj, mode, state)
            else:
                # Unknown goal type: fall back to 1
                est = 1

            if est > max_goal_cost:
                max_goal_cost = est

        return max(max_goal_cost, primitive_lb)

    # ----------------- Goal estimators -----------------

    def _estimate_soil(self, sample_wp: str, state: int) -> int:
        # Need: be at sample_wp, sample_soil, then communicate at a comm waypoint.
        # Lower bound actions (excluding navigation): sample_soil (1) + communicate (1)
        base_actions = 2

        best = 10**9
        for r in self.rovers:
            if r not in self._soil_capable:
                continue
            rpos = self._rover_position(r, state)
            if rpos is None:
                continue
            d1 = self._dist_lookup(r, rpos, sample_wp)
            if d1 is None:
                continue

            # after sampling at sample_wp, go to some comm waypoint
            d2 = self._min_dist_to_set(r, sample_wp, self.comm_waypoints)
            if d2 is None:
                continue

            best = min(best, d1 + d2 + base_actions)

        return 1 if best == 10**9 else best

    def _estimate_rock(self, sample_wp: str, state: int) -> int:
        base_actions = 2  # sample_rock + communicate

        best = 10**9
        for r in self.rovers:
            if r not in self._rock_capable:
                continue
            rpos = self._rover_position(r, state)
            if rpos is None:
                continue
            d1 = self._dist_lookup(r, rpos, sample_wp)
            if d1 is None:
                continue
            d2 = self._min_dist_to_set(r, sample_wp, self.comm_waypoints)
            if d2 is None:
                continue
            best = min(best, d1 + d2 + base_actions)

        return 1 if best == 10**9 else best

    def _estimate_image(self, obj: str, mode: str, state: int) -> int:
        # Need: calibrate (navigate + calibrate action), navigate to image waypoint, take_image, communicate.
        # We lower-bound by choosing best rover+camera that supports mode and has calibration target.
        # Actions excluding navigation: calibrate(1) + take_image(1) + communicate(1) = 3
        base_actions = 3

        # Candidate waypoints to take image of obj
        view_wps = self._visible_from.get(obj, [])
        if not view_wps:
            return 1

        best = 10**9
        for r in self.rovers:
            if r not in self._img_capable:
                continue
            rpos = self._rover_position(r, state)
            if rpos is None:
                continue

            for cam in self._cameras_on_rover.get(r, []):
                if mode not in self._supports.get(cam, set()):
                    continue
                calib_obj = self._calib_target.get(cam)
                if calib_obj is None:
                    continue
                calib_wps = self._visible_from.get(calib_obj, [])
                if not calib_wps:
                    continue

                # nav: rover -> some calib_wp -> some view_wp -> some comm_wp
                d_cal = self._min_dist_to_set(r, rpos, calib_wps)
                if d_cal is None:
                    continue

                # optimistic: from chosen calib waypoint to view waypoint (take min pair distance)
                d_cal_to_view = self._min_pair_dist(r, calib_wps, view_wps)
                if d_cal_to_view is None:
                    continue

                # from view waypoint to some comm waypoint
                d_view_to_comm = self._min_dist_from_set_to_set(r, view_wps, self.comm_waypoints)
                if d_view_to_comm is None:
                    continue

                best = min(best, d_cal + d_cal_to_view + d_view_to_comm + base_actions)

        return 1 if best == 10**9 else best

    # ----------------- Bitwise / parsing utilities -----------------

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

    def _collect_objects_from_facts(self, pred: str, arity: int, obj_index: int = 0) -> List[str]:
        out = []
        for _gid, args in self._fact_by_pred.get(pred, []):
            if len(args) == arity:
                out.append(args[obj_index])
        return list(dict.fromkeys(out))

    def _rover_position(self, rover: str, state: int) -> Optional[str]:
        # Scan known at facts for this rover (fast enough; per rover few facts)
        # We avoid model.state_explicit_repr.
        for (r, wp), gid in self._at_fact_ids.items():
            if r != rover:
                continue
            if ((state >> gid) & 1) == 1:
                return wp
        return None

    # ----------------- Distance helpers -----------------

    def _bfs_distances(self, rover: str, src: str) -> Dict[str, int]:
        dist = {src: 0}
        q = deque([src])
        adj = self._adj.get(rover, {})
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _dist_lookup(self, rover: str, a: str, b: str) -> Optional[int]:
        return self._dist.get(rover, {}).get(a, {}).get(b)

    def _min_dist_to_set(self, rover: str, src: str, targets: List[str]) -> Optional[int]:
        if not targets:
            return None
        dmap = self._dist.get(rover, {}).get(src)
        if not dmap:
            return None
        best = None
        for t in targets:
            d = dmap.get(t)
            if d is None:
                continue
            best = d if best is None else min(best, d)
        return best

    def _min_pair_dist(self, rover: str, sources: List[str], targets: List[str]) -> Optional[int]:
        best = None
        for s in sources:
            dmap = self._dist.get(rover, {}).get(s)
            if not dmap:
                continue
            for t in targets:
                d = dmap.get(t)
                if d is None:
                    continue
                best = d if best is None else min(best, d)
        return best

    def _min_dist_from_set_to_set(self, rover: str, sources: List[str], targets: List[str]) -> Optional[int]:
        # min over s in sources: dist(s, targets)
        best = None
        for s in sources:
            d = self._min_dist_to_set(rover, s, targets)
            if d is None:
                continue
            best = d if best is None else min(best, d)
        return best
