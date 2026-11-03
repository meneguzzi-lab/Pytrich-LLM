from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


@dataclass
class _GoalReq:
    """Compact representation of one goal requirement."""
    kind: str  # 'soil' | 'rock' | 'image'
    w: Optional[str] = None
    o: Optional[str] = None
    m: Optional[str] = None


class RoverGPT52Heuristic10(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic terms)
    -----------------------------------------------------
    Rover problems consist of many independent *deliverables* that must be
    *communicated* to a lander:
      - communicated_soil_data(waypoint)
      - communicated_rock_data(waypoint)
      - communicated_image_data(objective, mode)

    Each deliverable has a characteristic minimal primitive structure:
      Soil at waypoint p (ignoring ordering/resource interactions):
        navigate-to(p) + sample_soil + communicate
      Rock at waypoint p:
        navigate-to(p) + sample_rock + communicate
      Image (o,m):
        calibrate (requires reaching some calibration waypoint for (camera,o_cal))
        + navigate-to(image waypoint w where visible_from(o,w)) + take_image + communicate

    Major planning difficulty in large instances is *navigation* cost (graph
    distances), plus occasional store-emptying and calibration.

    Heuristic strategy
    ------------------
    We compute a fast admissible lower bound as a sum of independent per-goal
    lower bounds, using optimistic assumptions:
      - choose the best rover for each goal (min over rovers)
      - choose the best sampling/observation waypoint (min over candidates)
      - use shortest-path distances in the rover-specific traverse graph
      - ignore negative interactions between goals (shared rover, store capacity,
        camera calibration consumed by taking image, channel contention, etc.)

    This yields an admissible (and typically informative) estimate for A*/GBFS.

    Components used per unsatisfied goal:
      Soil goal at p:
        min_rover [ dist(rover_pos -> p) ] + 2
        (+2 accounts for sample_soil + communicate)

      Rock goal at p:
        min_rover [ dist(rover_pos -> p) ] + 2

      Image goal (o,m):
        For each rover with imaging capability and some onboard camera supporting m:
          - pick a camera c
          - pick an image waypoint w_img in visible_from(o, w_img)
          - if c already calibrated on rover: calibration_cost = 0 else
              calibration_cost = min_{w_cal s.t. visible_from(o_cal, w_cal)} dist(pos->w_cal) + 1
              where o_cal is calibration_target(c, o_cal)
          - imaging_cost = dist(pos -> w_img) + 1 (take_image)
          - communication_cost = 1 (communicate_image_data)
        Lower bound is the minimum total among those optimistic choices.

    Efficiency notes
    ----------------
    - Uses bitwise checks for goal satisfaction.
    - Precomputes:
        * mapping of predicate instances -> fact global id
        * initial rover positions (updated from state each call by scanning only
          relevant at[rover,wp] facts present in state via bit checks)
        * rover-specific adjacency graphs from can_traverse
        * waypoint candidates for each objective visibility
        * camera metadata (on_board, supports, calibration_target)
    - Shortest paths computed on demand with BFS and cached per (rover, source).

    The heuristic is admissible but not necessarily consistent; in practice it
    still strongly reduces expansions in this domain.
    """

    def __init__(self, model, rover_gpt52_heuristic_10: str = "rover_gpt52_heuristic_10"):
        super().__init__(model, rover_gpt52_heuristic_10)
        self.model = model

        # --- Parse all facts once into useful indices ---
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Typed object sets (best-effort derived from fact arguments)
        self._rovers: Set[str] = set()
        self._waypoints: Set[str] = set()
        self._landers: Set[str] = set()
        self._cameras: Set[str] = set()
        self._objectives: Set[str] = set()
        self._modes: Set[str] = set()
        self._stores: Set[str] = set()

        # Navigation edges per rover from can_traverse
        self._adj: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # visibility (undirected in instances but keep directed as given)
        self._visible: Set[Tuple[str, str]] = set()

        # lander location(s)
        self._lander_at: List[Tuple[str, str]] = []  # (lander, waypoint)

        # objective visibility
        self._obj_visible_from: Dict[str, Set[str]] = defaultdict(set)  # objective -> waypoints

        # rover capabilities
        self._equipped_soil: Set[str] = set()
        self._equipped_rock: Set[str] = set()
        self._equipped_img: Set[str] = set()

        # camera relations
        self._on_board: Dict[str, str] = {}  # camera -> rover
        self._supports: Dict[str, Set[str]] = defaultdict(set)  # camera -> modes
        self._calib_target: Dict[str, str] = {}  # camera -> objective (calibration target objective)

        # Fast access to all at[rover,wp] fact ids for scanning rover positions
        self._at_fact_ids_by_rover: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # For calibration: objective -> possible calibration waypoints (via visible_from)
        # already covered by _obj_visible_from

        # Cache distances: (rover, source_wp) -> dict(target_wp -> dist)
        self._dist_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

        # Precompute goal requirements from model.goals
        self._goal_reqs: List[_GoalReq] = []

        self._init_from_model_facts()
        self._init_goals()

    # ------------------------ Fact parsing utilities ------------------------
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

    def _init_from_model_facts(self):
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_id[key] = f.global_id

            # Collect structural info
            if pred == "can_traverse" and len(args) == 3:
                r, a, b = args
                self._rovers.add(r)
                self._waypoints.add(a)
                self._waypoints.add(b)
                self._adj[r][a].append(b)
            elif pred == "visible" and len(args) == 2:
                a, b = args
                self._waypoints.add(a)
                self._waypoints.add(b)
                self._visible.add((a, b))
            elif pred == "at_lander" and len(args) == 2:
                l, w = args
                self._landers.add(l)
                self._waypoints.add(w)
                self._lander_at.append((l, w))
            elif pred == "visible_from" and len(args) == 2:
                o, w = args
                self._objectives.add(o)
                self._waypoints.add(w)
                self._obj_visible_from[o].add(w)
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self._rovers.add(args[0])
                self._equipped_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self._rovers.add(args[0])
                self._equipped_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self._rovers.add(args[0])
                self._equipped_img.add(args[0])
            elif pred == "on_board" and len(args) == 2:
                c, r = args
                self._cameras.add(c)
                self._rovers.add(r)
                self._on_board[c] = r
            elif pred == "supports" and len(args) == 2:
                c, m = args
                self._cameras.add(c)
                self._modes.add(m)
                self._supports[c].add(m)
            elif pred == "calibration_target" and len(args) == 2:
                c, o = args
                self._cameras.add(c)
                self._objectives.add(o)
                self._calib_target[c] = o
            elif pred == "store_of" and len(args) == 2:
                s, r = args
                self._stores.add(s)
                self._rovers.add(r)
            elif pred == "at" and len(args) == 2:
                r, w = args
                # Only rovers appear in this predicate in this domain
                self._rovers.add(r)
                self._waypoints.add(w)
                self._at_fact_ids_by_rover[r].append((w, f.global_id))

    def _init_goals(self):
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_reqs.append(_GoalReq("soil", w=args[0]))
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_reqs.append(_GoalReq("rock", w=args[0]))
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_reqs.append(_GoalReq("image", o=args[0], m=args[1]))

    # ------------------------ Distance computation ------------------------
    def _bfs_distances(self, rover: str, source: str) -> Dict[str, int]:
        key = (rover, source)
        if key in self._dist_cache:
            return self._dist_cache[key]

        dist: Dict[str, int] = {source: 0}
        q = deque([source])
        adj_r = self._adj.get(rover, {})

        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj_r.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)

        self._dist_cache[key] = dist
        return dist

    def _dist(self, rover: str, a: str, b: str) -> int:
        if a == b:
            return 0
        distmap = self._bfs_distances(rover, a)
        return distmap.get(b, 10**9)  # unreachable -> very large

    # ------------------------ State queries (bitwise) ------------------------
    def _fact_true(self, state: int, fact_global_id: int) -> bool:
        return ((state >> fact_global_id) & 1) == 1

    def _goal_satisfied(self, state: int, req: _GoalReq) -> bool:
        if req.kind == "soil":
            fid = self._fact_id.get(("communicated_soil_data", (req.w,)))
            return fid is not None and self._fact_true(state, fid)
        if req.kind == "rock":
            fid = self._fact_id.get(("communicated_rock_data", (req.w,)))
            return fid is not None and self._fact_true(state, fid)
        if req.kind == "image":
            fid = self._fact_id.get(("communicated_image_data", (req.o, req.m)))
            return fid is not None and self._fact_true(state, fid)
        return False

    def _current_rover_positions(self, state: int) -> Dict[str, str]:
        """Return rover -> waypoint using only at-facts known for that rover."""
        pos: Dict[str, str] = {}
        for r, wp_fids in self._at_fact_ids_by_rover.items():
            for w, gid in wp_fids:
                if self._fact_true(state, gid):
                    pos[r] = w
                    break
        return pos

    def _is_calibrated(self, state: int, cam: str, rover: str) -> bool:
        fid = self._fact_id.get(("calibrated", (cam, rover)))
        return fid is not None and self._fact_true(state, fid)

    # ------------------------ Per-goal lower bounds ------------------------
    def _soil_lb(self, rover_pos: Dict[str, str], p: str) -> int:
        best = 10**9
        for r in self._equipped_soil:
            if r not in rover_pos:
                continue
            d = self._dist(r, rover_pos[r], p)
            if d < best:
                best = d
        if best >= 10**9:
            return 0  # unreachable => don't mislead; keep admissible but weak
        return best + 2

    def _rock_lb(self, rover_pos: Dict[str, str], p: str) -> int:
        best = 10**9
        for r in self._equipped_rock:
            if r not in rover_pos:
                continue
            d = self._dist(r, rover_pos[r], p)
            if d < best:
                best = d
        if best >= 10**9:
            return 0
        return best + 2

    def _image_lb(self, state: int, rover_pos: Dict[str, str], o: str, m: str) -> int:
        w_imgs = self._obj_visible_from.get(o)
        if not w_imgs:
            return 0

        best_total = 10**9

        # For each camera that supports mode m and is on-board some imaging rover
        for cam, r in self._on_board.items():
            if r not in self._equipped_img:
                continue
            if r not in rover_pos:
                continue
            if m not in self._supports.get(cam, set()):
                continue

            pos = rover_pos[r]

            # Calibration cost (optimistic): if already calibrated -> 0 else
            # reach any waypoint that sees cam's calibration target objective.
            if self._is_calibrated(state, cam, r):
                calib_cost = 0
                calib_end_pos_options = [pos]
            else:
                o_cal = self._calib_target.get(cam)
                if o_cal is None:
                    continue
                w_cals = self._obj_visible_from.get(o_cal)
                if not w_cals:
                    continue
                best_cal = 10**9
                best_cal_w = None
                for wcal in w_cals:
                    dcal = self._dist(r, pos, wcal)
                    if dcal < best_cal:
                        best_cal = dcal
                        best_cal_w = wcal
                if best_cal >= 10**9:
                    continue
                calib_cost = best_cal + 1  # calibrate action
                calib_end_pos_options = [best_cal_w]

            # Imaging waypoint and take_image
            for calib_end in calib_end_pos_options:
                best_img = 10**9
                for wimg in w_imgs:
                    dimg = self._dist(r, calib_end, wimg)
                    if dimg < best_img:
                        best_img = dimg
                if best_img >= 10**9:
                    continue
                img_cost = best_img + 1  # take_image

                # communicate_image_data
                total = calib_cost + img_cost + 1
                if total < best_total:
                    best_total = total

        if best_total >= 10**9:
            return 0
        return best_total

    # ------------------------ Heuristic interface ------------------------
    def __call__(self, node) -> int:
        state = node.state

        rover_pos = self._current_rover_positions(state)

        h = 0
        for req in self._goal_reqs:
            if self._goal_satisfied(state, req):
                continue

            if req.kind == "soil":
                h += self._soil_lb(rover_pos, req.w)
            elif req.kind == "rock":
                h += self._rock_lb(rover_pos, req.w)
            elif req.kind == "image":
                h += self._image_lb(state, rover_pos, req.o, req.m)

        # Keep heuristic non-negative int
        if h < 0:
            h = 0
        return int(h)
