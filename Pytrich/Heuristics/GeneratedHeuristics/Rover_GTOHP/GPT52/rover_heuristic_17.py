from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

try:
    # Pytrich typical import style
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback (some installations expose it differently)
    from heuristic import Heuristic


class RoverGPT52Heuristic17(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights):
    ---------------------------------
    Rover goals are *communicated_* facts (soil/rock/image). Achieving any such goal
    requires (at least) a communication action at some waypoint that is visible to the lander.

    The domain methods expand each high-level "get_*_data" task into:
      - navigate to a sampling/imaging waypoint
      - possibly empty store (drop)
      - sample / take_image (with calibration for images)
      - navigate to a communication waypoint visible to the lander
      - communicate_* (soil/rock/image)

    Key cost drivers are therefore:
      (1) navigation distance in the rover's traversability graph
      (2) extra steps for store handling (drop) when store is full
      (3) extra steps for imaging calibration (calibrate)

    This heuristic provides a *fast lower bound* (admissible-ish in practice, but not
    formally guaranteed due to ignoring some negative effects / ordering nuances):

    - For each unsatisfied communicated_* goal, compute a minimal estimate:
        min_rover( dist(rover_pos -> some suitable "source" waypoint) +
                   dist(source -> some communication waypoint visible to lander) )
      plus constant action costs for sample/communicate (and for imaging, calibrate+take_image+communicate).

    - Communication waypoint constraint:
        a rover can communicate at waypoint w if there exists a lander waypoint wl with
        +at_lander[lander,wl] and +visible[w,wl]. In the given problems there is typically
        one lander, but we handle multiple.

    - Source waypoint for soil/rock goals is the goal's waypoint p itself (because
      communicated_soil_data[p] needs have_soil_analysis[x,p], which requires sampling at p).

    - Source waypoint for image goal (o,m): any waypoint w with +visible_from[o,w].

    - Distances are computed on a relaxed, rover-agnostic adjacency built from +can_traverse
      (we ignore the rover parameter) because in most benchmarks can_traverse is consistent
      and this yields a safe optimistic distance.

    Efficiency considerations:
    --------------------------
    - Uses only bitwise checks for state facts.
    - Precomputes:
        * mapping from fact global_id -> parsed (pred, args)
        * adjacency graph of waypoints
        * all-pairs shortest paths are NOT computed; instead we do BFS from a small set of
          start nodes each call (one per rover position plus a few targets) and cache BFS results
          per start waypoint.
    - Goal facts are extracted from model.goals via state_explicit_repr (allowed once per init).

    Notes:
    ------
    This heuristic is intended to reduce node expansions by strongly preferring states that
    move rovers closer to required sampling/imaging and especially closer to lander-visible
    communication waypoints.
    """

    def __init__(self, model, rover_gpt52_heuristic_17: str = "rover_gpt52_heuristic_17"):
        super().__init__(model, rover_gpt52_heuristic_17)
        self.model = model

        # --- Parse all fact names once ---
        self._fact_pred: List[Optional[str]] = [None] * len(self.model.facts)
        self._fact_args: List[List[str]] = [[] for _ in range(len(self.model.facts))]
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            self._fact_pred[f.global_id] = pred
            self._fact_args[f.global_id] = args

        # Build indices for fast state queries
        # (predicate, tuple(args)) -> global_id
        self._fact_index: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for f in self.model.facts:
            pred, args = self._fact_pred[f.global_id], self._fact_args[f.global_id]
            if pred is None:
                continue
            self._fact_index[(pred, tuple(args))] = f.global_id

        # --- Extract static objects ---
        self._waypoints: Set[str] = set()
        for f in self.model.facts:
            pred = self._fact_pred[f.global_id]
            if pred == "visible":
                a = self._fact_args[f.global_id]
                if len(a) == 2:
                    self._waypoints.add(a[0])
                    self._waypoints.add(a[1])
            elif pred == "at":
                a = self._fact_args[f.global_id]
                if len(a) == 2:
                    self._waypoints.add(a[1])
            elif pred == "at_lander":
                a = self._fact_args[f.global_id]
                if len(a) == 2:
                    self._waypoints.add(a[1])
            elif pred == "can_traverse":
                a = self._fact_args[f.global_id]
                if len(a) == 3:
                    self._waypoints.add(a[1])
                    self._waypoints.add(a[2])

        # --- Precompute relaxed traversal adjacency (ignore rover parameter) ---
        self._adj: Dict[str, Set[str]] = {w: set() for w in self._waypoints}
        for f in self.model.facts:
            if self._fact_pred[f.global_id] == "can_traverse":
                a = self._fact_args[f.global_id]
                if len(a) == 3:
                    _rover, w1, w2 = a
                    if w1 in self._adj:
                        self._adj[w1].add(w2)
                    else:
                        self._adj[w1] = {w2}

        # --- Goal parsing ---
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_communicated_soil: List[str] = []
        self._goal_communicated_rock: List[str] = []
        self._goal_communicated_image: List[Tuple[str, str]] = []

        # We'll also keep goal fact global_ids for quick satisfaction checks
        self._goal_fact_ids: List[int] = []

        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            gid = self._fact_index.get((pred, tuple(args)))
            if gid is not None:
                self._goal_fact_ids.append(gid)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_communicated_soil.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_communicated_rock.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_communicated_image.append((args[0], args[1]))

        # --- Precompute visible_from map: objective -> list(waypoints) ---
        self._visible_from: Dict[str, List[str]] = defaultdict(list)
        for f in self.model.facts:
            if self._fact_pred[f.global_id] == "visible_from":
                a = self._fact_args[f.global_id]
                if len(a) == 2:
                    obj, wp = a
                    self._visible_from[obj].append(wp)

        # --- Cache BFS distances per start waypoint ---
        self._bfs_cache: Dict[str, Dict[str, int]] = {}

    # ------------------------- Core heuristic API -------------------------
    def __call__(self, state_bitwise: int, task_network: List[Any] = None) -> int:
        """Return heuristic estimate for remaining effort.

        The planner will call this frequently; keep it fast.
        """
        # Quick goal test (cheap lower bound 0)
        # (We do not check task_network emptiness here; heuristic guides search anyway.)
        unsat = 0
        for gid in self._goal_fact_ids:
            if ((state_bitwise >> gid) & 1) == 0:
                unsat += 1
        if unsat == 0:
            return 0

        # Extract current rover positions (from +at[rover,wp])
        rover_positions = self._get_rover_positions(state_bitwise)
        if not rover_positions:
            # Fallback: no rover position known -> weak heuristic
            return unsat

        # Compute set of communication waypoints (visible to any lander location)
        comm_wps = self._get_communication_waypoints(state_bitwise)
        if not comm_wps:
            # If nothing visible to lander, still return something proportional to unsatisfied goals
            return 2 * unsat

        # Prepare BFS caches for rover positions and sometimes for targets
        # (done lazily in _dist)

        h_val = 0

        # Soil goals: communicated_soil_data[p]
        for p in self._goal_communicated_soil:
            gid = self._fact_index.get(("communicated_soil_data", (p,)))
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                continue
            # minimal: navigate to p + sample + navigate to comm + communicate
            # sample_soil + communicate_soil_data* => at least 2 primitive actions
            best = self._best_rover_route_cost(rover_positions, src_candidates=[p], comm_wps=comm_wps)
            # Add minimal action overhead (sample + communicate)
            h_val += best + 2

        # Rock goals
        for p in self._goal_communicated_rock:
            gid = self._fact_index.get(("communicated_rock_data", (p,)))
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                continue
            best = self._best_rover_route_cost(rover_positions, src_candidates=[p], comm_wps=comm_wps)
            h_val += best + 2

        # Image goals: communicated_image_data[o,m]
        for (o, m) in self._goal_communicated_image:
            gid = self._fact_index.get(("communicated_image_data", (o, m)))
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                continue
            srcs = self._visible_from.get(o, [])
            if not srcs:
                # Impossible/unknown; give small penalty
                h_val += 3
                continue
            best = self._best_rover_route_cost(rover_positions, src_candidates=srcs, comm_wps=comm_wps)
            # Minimal primitive actions: calibrate + take_image + communicate = 3
            # (navigation handled by best)
            h_val += best + 3

        # Light normalization: avoid returning 0 when unsatisfied > 0
        if h_val <= 0:
            return unsat
        return int(h_val)

    # ------------------------- Helper methods -------------------------
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

    def _get_rover_positions(self, state_bitwise: int) -> Dict[str, str]:
        """Return rover -> waypoint from +at[rover,wp] facts that are true."""
        res: Dict[str, str] = {}
        # Iterate only over facts (fast; number of facts is manageable)
        for f in self.model.facts:
            gid = f.global_id
            if self._fact_pred[gid] != "at":
                continue
            if ((state_bitwise >> gid) & 1) == 0:
                continue
            args = self._fact_args[gid]
            if len(args) == 2:
                rover, wp = args
                res[rover] = wp
        return res

    def _get_communication_waypoints(self, state_bitwise: int) -> Set[str]:
        """Waypoints from which a lander is visible: {w | exists l,wl: at_lander(l,wl) & visible(w,wl)}"""
        lander_wps: Set[str] = set()
        for f in self.model.facts:
            gid = f.global_id
            if self._fact_pred[gid] != "at_lander":
                continue
            if ((state_bitwise >> gid) & 1) == 0:
                continue
            args = self._fact_args[gid]
            if len(args) == 2:
                _l, wl = args
                lander_wps.add(wl)

        if not lander_wps:
            return set()

        comm: Set[str] = set()
        for f in self.model.facts:
            gid = f.global_id
            if self._fact_pred[gid] != "visible":
                continue
            if ((state_bitwise >> gid) & 1) == 0:
                continue
            a = self._fact_args[gid]
            if len(a) != 2:
                continue
            w1, w2 = a
            if w2 in lander_wps:
                comm.add(w1)
        return comm

    def _bfs_from(self, start: str) -> Dict[str, int]:
        """Unweighted shortest path distances from start over relaxed adjacency."""
        if start in self._bfs_cache:
            return self._bfs_cache[start]

        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in self._adj.get(u, ()):  # may be empty
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        self._bfs_cache[start] = dist
        return dist

    def _dist(self, a: str, b: str) -> int:
        """Relaxed distance between waypoints; returns large number if unreachable."""
        if a == b:
            return 0
        dist_map = self._bfs_from(a)
        return dist_map.get(b, 10**6)

    def _best_rover_route_cost(
        self,
        rover_positions: Dict[str, str],
        src_candidates: List[str],
        comm_wps: Set[str],
    ) -> int:
        """Compute min over rovers and choices of source and comm waypoint:

        min_{r,src,comm} dist(pos_r, src) + dist(src, comm)

        This captures the two navigation phases in the methods (go to data site, then go to comm site).
        """
        best = 10**9
        # Pre-filter comm_wps list for iteration speed
        comm_list = list(comm_wps)

        for _r, pos in rover_positions.items():
            # distances from rover position once
            dist_from_pos = self._bfs_from(pos)
            for src in src_candidates:
                d1 = dist_from_pos.get(src, 10**6)
                if d1 >= best:
                    continue
                # distances from src
                dist_from_src = self._bfs_from(src)
                # find nearest comm
                d2_best = 10**6
                for cw in comm_list:
                    d2 = dist_from_src.get(cw, 10**6)
                    if d2 < d2_best:
                        d2_best = d2
                        if d2_best == 0:
                            break
                total = d1 + d2_best
                if total < best:
                    best = total

        if best >= 10**6:
            # unreachable in relaxed graph
            return 10
        return best
