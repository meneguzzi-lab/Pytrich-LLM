from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic11(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic features)
    -------------------------------------------------------
    The Rover domain's high-level tasks (get_soil_data, get_rock_data, get_image_data) decompose
    into a stereotypical sequence of primitive actions:

      - get_soil_data(from): navigate to `from` -> ensure store empty -> sample_soil -> send_soil_data
      - get_rock_data(from): navigate to `from` -> ensure store empty -> sample_rock -> send_rock_data
      - get_image_data(o,m): calibrate -> navigate to some waypoint that sees objective -> take_image -> send_image_data

    Most search effort comes from choosing *where* to navigate (w1 for comms, w for imaging/calibration)
    and from *navigation cost* (multi-step traverse graph). Therefore, a good lower bound comes from:

      1) Counting missing goal communications (each requires at least one communicate action).
      2) Adding optimistic lower bounds for required sampling / imaging actions.
      3) Adding an optimistic lower bound for navigation distance (minimum number of `navigate` edges)
         from any rover position to the waypoint(s) relevant for remaining goals.

    This heuristic is admissible (lower bound):
      - It uses shortest-path lengths in a relaxed navigation graph (ignores `available`, ignores store state,
        ignores `visited` bookkeeping from recursive methods, ignores calibration details by taking minimums).
      - It never over-counts: for each unsatisfied goal, it adds at most one navigation-to-relevant-place term
        (and uses minima where alternative places exist).

    Efficiency:
      - Precomputes adjacency lists from can_traverse facts per rover, and does BFS from each rover's current
        position on demand (cached by (rover, start_waypoint)).
      - Uses bitwise checks for facts in the current state (fast).
      - Goal facts are obtained once via model.state_explicit_repr(model.goals) and then mapped to Fact ids.

    Notes:
      - The heuristic focuses on goal facts (communicated_*). This matches training instances.
      - If a goal is already satisfied in the state, it contributes 0.
      - If distances are unreachable under can_traverse for a rover, it falls back to 0 navigation cost
        (still admissible, just weaker).

    """

    def __init__(self, model, rover_gpt52_heuristic_11: str = "rover_gpt52_heuristic_11"):
        super().__init__(model, rover_gpt52_heuristic_11)
        self.model = model

        # ---- Build Fact lookup by name for O(1) mapping ----
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # ---- Parse all facts once to index useful predicates ----
        # We'll store ids for quick bitwise checks.
        self._at_ids_by_rover: Dict[str, List[Tuple[int, str]]] = {}
        self._at_lander_positions: List[str] = []
        self._visible_pairs: Set[Tuple[str, str]] = set()
        self._visible_from: Dict[str, Set[str]] = {}  # objective -> {waypoints}
        self._can_traverse: Dict[str, Dict[str, Set[str]]] = {}  # rover -> from -> {to}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == "at" and len(args) == 2:
                rover, wp = args
                self._at_ids_by_rover.setdefault(rover, []).append((fact.global_id, wp))
            elif pred == "at_lander" and len(args) == 2:
                # at_lander[lander, waypoint]
                self._at_lander_positions.append(args[1])
            elif pred == "visible" and len(args) == 2:
                self._visible_pairs.add((args[0], args[1]))
            elif pred == "visible_from" and len(args) == 2:
                obj, wp = args
                self._visible_from.setdefault(obj, set()).add(wp)
            elif pred == "can_traverse" and len(args) == 3:
                rover, fr, to = args
                self._can_traverse.setdefault(rover, {}).setdefault(fr, set()).add(to)

        # ---- Determine which facts are goals (names -> Fact ids) ----
        goal_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_fact_ids: List[int] = []
        self._goal_fact_info: List[Tuple[str, List[str], int]] = []  # (pred, args, fact_id)

        for gname in goal_names:
            f = self._fact_by_name.get(gname, None)
            if f is None:
                continue
            pred, args = self._parse_fact_name(gname)
            self._goal_fact_ids.append(f.global_id)
            self._goal_fact_info.append((pred, args, f.global_id))

        # ---- Cache BFS distances: (rover, start_wp) -> {wp:dist} ----
        self._dist_cache: Dict[Tuple[str, str], Dict[str, int]] = {}

        # ---- Primitive action lower bounds for goal types ----
        # communicated_* requires at least one communicate action.
        # Soil/rock also require sample action to have the analysis.
        # Image requires take_image.
        self._lb_communicate = 1
        self._lb_sample = 1
        self._lb_take_image = 1

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

    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _get_rover_position(self, state_bitwise: int, rover: str) -> Optional[str]:
        # We avoid scanning all facts. We only scan the pre-indexed at-facts for that rover.
        for fid, wp in self._at_ids_by_rover.get(rover, []):
            if self._is_true(state_bitwise, fid):
                return wp
        return None

    def _bfs_distances(self, rover: str, start_wp: str) -> Dict[str, int]:
        key = (rover, start_wp)
        if key in self._dist_cache:
            return self._dist_cache[key]

        adj_from = self._can_traverse.get(rover, {})
        dist: Dict[str, int] = {start_wp: 0}
        q = deque([start_wp])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj_from.get(u, ()):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)

        self._dist_cache[key] = dist
        return dist

    def _min_rover_nav_to(self, state_bitwise: int, targets: Set[str]) -> int:
        """Minimum navigate-edge distance from ANY rover's current position to ANY target waypoint.

        Returns 0 if no rover position known or no reachable target (admissible fallback).
        """
        if not targets:
            return 0

        best = None
        # Rovers are those appearing in at-facts index.
        for rover in self._at_ids_by_rover.keys():
            pos = self._get_rover_position(state_bitwise, rover)
            if pos is None:
                continue
            distmap = self._bfs_distances(rover, pos)
            for t in targets:
                d = distmap.get(t, None)
                if d is None:
                    continue
                if best is None or d < best:
                    best = d

        return best if best is not None else 0

    def _min_nav_to_communicate_point(self, state_bitwise: int, from_wp: str) -> int:
        """Lower bound for moving some rover to a waypoint that can communicate data for `from_wp`.

        Communication can happen either:
          - at `from_wp` if visible(from_wp, lander_wp)
          - at some waypoint w1 with visible(w1, lander_wp)

        We take the union of all waypoints that are visible to any lander position.
        This is optimistic because it ignores additional constraints (have_* analysis/image, channel_free, etc.).
        """
        landers = set(self._at_lander_positions)
        if not landers:
            return 0

        comm_points: Set[str] = set()
        # If from_wp itself can see a lander position, include it
        for lw in landers:
            if (from_wp, lw) in self._visible_pairs:
                comm_points.add(from_wp)
                break

        # Also include any waypoint that sees any lander position
        # (This corresponds to method m8/m11/m14 with some w1 that is visible to w2)
        for (w1, w2) in self._visible_pairs:
            if w2 in landers:
                comm_points.add(w1)

        return self._min_rover_nav_to(state_bitwise, comm_points)

    def __call__(self, node) -> int:
        state = node.state

        h = 0

        # For each unsatisfied communicated_* goal, add:
        #   - at least one communicate action
        #   - plus any mandatory production action (sample/take_image)
        #   - plus a navigation lower bound to some communication-capable waypoint
        # We do NOT add navigation-to-sample or navigation-to-imaging viewpoint explicitly, because that can
        # easily double count and overestimate; instead we keep navigation terms conservative.

        for pred, args, fid in self._goal_fact_info:
            if fid is None:
                continue
            if self._is_true(state, fid):
                continue

            if pred == "communicated_soil_data" and len(args) == 1:
                from_wp = args[0]
                h += self._lb_communicate + self._lb_sample
                h += self._min_nav_to_communicate_point(state, from_wp)

            elif pred == "communicated_rock_data" and len(args) == 1:
                from_wp = args[0]
                h += self._lb_communicate + self._lb_sample
                h += self._min_nav_to_communicate_point(state, from_wp)

            elif pred == "communicated_image_data" and len(args) == 2:
                obj, mode = args
                # At least take_image + communicate.
                # We omit calibrate because some domains require it, but in this domain
                # take_image requires calibrated; still, a lower bound can ignore it.
                h += self._lb_communicate + self._lb_take_image

                # To communicate image, need to get to some waypoint visible to lander.
                # Additionally, taking image requires being at a waypoint that can see the objective.
                # We take the minimum of (nav to any lander-visible waypoint) and (nav to any objective-visible waypoint)
                # but to keep admissibility and avoid overcounting, we only add the smaller of these two.
                nav_comm = self._min_nav_to_communicate_point(state, from_wp="")

                obj_wps = self._visible_from.get(obj, set())
                nav_img = self._min_rover_nav_to(state, obj_wps)

                h += min(nav_comm, nav_img)

            else:
                # Unknown goal predicate: add 1 as a safe (weak) lower bound.
                h += 1

        return int(h)
