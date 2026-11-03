from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

try:
    # Pytrich typical import path
    from Pytrich.Heuristics.heuristic_base import Heuristic
except Exception:
    # Fallback (some installations expose it differently)
    from heuristics.heuristic_base import Heuristic


class RoverGPT52Heuristic8(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    RATIONALE / DOMAIN INSIGHTS
    --------------------------
    In this domain, all top-level goals are of the form:
      - communicated_soil_data(waypoint)
      - communicated_rock_data(waypoint)
      - communicated_image_data(objective, mode)

    Achieving any communicated_* goal requires (at least):
      - having the corresponding data on some rover:
          have_soil_analysis(rover, wp) OR have_rock_analysis(rover, wp) OR have_image(rover, obj, mode)
      - being positioned at a waypoint from which the lander is visible (directly or via a visible waypoint)
      - executing a communicate_* operator.

    The main cost driver is navigation. Navigation is constrained by can_traverse edges,
    and each move also needs visible(p1,p2). In all provided instances, the navigation
    graph is sparse but connected; on large instances, navigation dominates.

    Heuristic strategy (fast, informative, domain-specific):
      1) Extract remaining *goal facts* from model.goals (bitset) without enumerating the state.
      2) For each remaining goal, compute a cheap lower bound of required primitive actions:
           - If communicated_* already true: 0.
           - Else at minimum, we must perform 1 communication action.
           - Plus a lower bound on navigation steps needed to get some suitable rover
             to some suitable communication waypoint.
           - Plus a small constant if data acquisition is still needed.
      3) Aggregate using max over goals (admissible lower bound under unit-ish costs),
         plus a very small tie-break term based on number of unsatisfied goals.

    Admissibility notes:
      - We use shortest-path distances in a *relaxed* movement graph:
          edge exists if can_traverse(x,a,b) and visible(a,b) appear in the static facts.
        This yields a true lower bound on number of navigate actions for that rover.
      - For each goal we take the MIN over rovers and relevant waypoints, hence still a lower bound.
      - We do NOT sum per-goal costs (would overcount shared navigation / shared communication position).
        We take MAX across goals, which is a standard admissible aggregation.

    Efficiency notes:
      - Precomputes:
          * which fact ids correspond to at(rover,wp), at_lander(lander,wp), visible(wp,wp)
          * per-rover movement adjacency and all-pairs distances (BFS from each waypoint)
          * mapping from goal fact -> its type and parameters
      - During evaluation, only uses bit operations to query truth of a small set of facts.

    This heuristic is designed to reduce node expansions by strongly preferring states
    that (a) already have data, and (b) are closer (in navigation steps) to some waypoint
    that can communicate with the lander.
    """

    def __init__(self, model, rover_gpt52_heuristic_8=None):
        super().__init__(model)
        self.model = model

        # --- Parse facts and build indices ---
        self.facts = list(model.facts)

        # Fact ID buckets
        self.at_fact_id: Dict[Tuple[str, str], int] = {}  # (rover, wp) -> fact_id
        self.at_lander_wps: Set[str] = set()
        self.visible_pairs: Set[Tuple[str, str]] = set()
        self.can_traverse_pairs: Dict[str, Set[Tuple[str, str]]] = {}  # rover -> {(a,b),...}

        # Data possession facts
        self.have_soil_fact_id: Dict[Tuple[str, str], int] = {}   # (rover, wp)
        self.have_rock_fact_id: Dict[Tuple[str, str], int] = {}
        self.have_image_fact_id: Dict[Tuple[str, str, str], int] = {}  # (rover, obj, mode)

        # Communication goal facts by their argument tuples
        self.comm_soil_goal_id: Dict[str, int] = {}   # wp -> fact_id
        self.comm_rock_goal_id: Dict[str, int] = {}
        self.comm_image_goal_id: Dict[Tuple[str, str], int] = {}  # (obj, mode) -> fact_id

        # Object sets
        self.rovers: Set[str] = set()
        self.waypoints: Set[str] = set()

        # Parse all fact names once
        for f in self.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue

            if pred == "at" and len(args) == 2:
                r, w = args
                self.rovers.add(r)
                self.waypoints.add(w)
                self.at_fact_id[(r, w)] = f.global_id

            elif pred == "at_lander" and len(args) == 2:
                _l, w = args
                self.waypoints.add(w)
                self.at_lander_wps.add(w)

            elif pred == "visible" and len(args) == 2:
                a, b = args
                self.waypoints.add(a)
                self.waypoints.add(b)
                self.visible_pairs.add((a, b))

            elif pred == "can_traverse" and len(args) == 3:
                r, a, b = args
                self.rovers.add(r)
                self.waypoints.add(a)
                self.waypoints.add(b)
                self.can_traverse_pairs.setdefault(r, set()).add((a, b))

            elif pred == "have_soil_analysis" and len(args) == 2:
                r, w = args
                self.rovers.add(r)
                self.waypoints.add(w)
                self.have_soil_fact_id[(r, w)] = f.global_id

            elif pred == "have_rock_analysis" and len(args) == 2:
                r, w = args
                self.rovers.add(r)
                self.waypoints.add(w)
                self.have_rock_fact_id[(r, w)] = f.global_id

            elif pred == "have_image" and len(args) == 3:
                r, o, m = args
                self.rovers.add(r)
                self.have_image_fact_id[(r, o, m)] = f.global_id

            elif pred == "communicated_soil_data" and len(args) == 1:
                (w,) = args
                self.waypoints.add(w)
                self.comm_soil_goal_id[w] = f.global_id

            elif pred == "communicated_rock_data" and len(args) == 1:
                (w,) = args
                self.waypoints.add(w)
                self.comm_rock_goal_id[w] = f.global_id

            elif pred == "communicated_image_data" and len(args) == 2:
                o, m = args
                self.comm_image_goal_id[(o, m)] = f.global_id

        # --- Precompute per-rover movement adjacency constrained by both can_traverse and visible ---
        self.move_adj: Dict[str, Dict[str, List[str]]] = {}  # rover -> from_wp -> [to_wp]
        for r in self.rovers:
            adj: Dict[str, List[str]] = {w: [] for w in self.waypoints}
            for (a, b) in self.can_traverse_pairs.get(r, set()):
                if (a, b) in self.visible_pairs:
                    adj.setdefault(a, []).append(b)
            self.move_adj[r] = adj

        # --- Precompute all-pairs shortest path distances per rover (BFS from each waypoint) ---
        # Distances measured in number of navigate actions.
        self.dist: Dict[str, Dict[str, Dict[str, int]]] = {}  # rover -> src -> dst -> dist
        for r in self.rovers:
            self.dist[r] = {}
            for src in self.waypoints:
                self.dist[r][src] = self._bfs_dist(self.move_adj[r], src)

        # --- Precompute for each waypoint whether it can communicate with lander ---
        # A rover can communicate from w1 if exists lander at w2 with visible(w1,w2).
        self.comm_wps: Set[str] = set()
        for w1 in self.waypoints:
            for w2 in self.at_lander_wps:
                if (w1, w2) in self.visible_pairs:
                    self.comm_wps.add(w1)
                    break

        # Cache goal fact ids list for quick scanning
        self.goal_fact_ids: List[int] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(name)
            if pred in ("communicated_soil_data", "communicated_rock_data", "communicated_image_data"):
                # Find global id via dictionaries built above
                if pred == "communicated_soil_data" and len(args) == 1 and args[0] in self.comm_soil_goal_id:
                    self.goal_fact_ids.append(self.comm_soil_goal_id[args[0]])
                elif pred == "communicated_rock_data" and len(args) == 1 and args[0] in self.comm_rock_goal_id:
                    self.goal_fact_ids.append(self.comm_rock_goal_id[args[0]])
                elif pred == "communicated_image_data" and len(args) == 2 and (args[0], args[1]) in self.comm_image_goal_id:
                    self.goal_fact_ids.append(self.comm_image_goal_id[(args[0], args[1])])

        # Build reverse lookup from fact_id to (pred,args) only for communication goals
        self.comm_goal_by_id: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
        for w, fid in self.comm_soil_goal_id.items():
            self.comm_goal_by_id[fid] = ("communicated_soil_data", (w,))
        for w, fid in self.comm_rock_goal_id.items():
            self.comm_goal_by_id[fid] = ("communicated_rock_data", (w,))
        for (o, m), fid in self.comm_image_goal_id.items():
            self.comm_goal_by_id[fid] = ("communicated_image_data", (o, m))

    # ----------------- Core heuristic API -----------------

    def __call__(self, node) -> int:
        """Return heuristic value for a search node.

        The node is expected to have .state (bitset int) and possibly .task_network.
        We ignore the task network content here and focus on remaining goal facts,
        because the provided benchmark instances use goals that correspond directly
        to communicated_* facts.
        """
        state = node.state

        # Quick goal satisfaction check (bitwise subset)
        if (state & self.model.goals) == self.model.goals:
            return 0

        # Compute current rover positions (only once per call)
        rover_pos = self._extract_rover_positions(state)

        # Evaluate each remaining communicated_* goal and take the maximum lower bound
        max_lb = 0
        unsat = 0

        # Iterate only over goal facts we care about
        for fid in self.goal_fact_ids:
            if self._is_true(state, fid):
                continue
            unsat += 1
            pred, args = self.comm_goal_by_id.get(fid, (None, ()))
            if pred is None:
                continue

            if pred == "communicated_soil_data":
                (wp,) = args
                lb = self._lb_communicate_sample(state, rover_pos, sample_type="soil", sample_wp=wp)
            elif pred == "communicated_rock_data":
                (wp,) = args
                lb = self._lb_communicate_sample(state, rover_pos, sample_type="rock", sample_wp=wp)
            else:
                o, m = args
                lb = self._lb_communicate_image(state, rover_pos, obj=o, mode=m)

            if lb > max_lb:
                max_lb = lb

        # Tie-break: very small pressure toward finishing multiple remaining goals.
        # Keep it admissible by adding 0 (or at most 0 in integer arithmetic).
        # We'll instead just return max_lb; search's secondary ordering can handle ties.
        return int(max_lb)

    # ----------------- Lower-bound components -----------------

    def _lb_communicate_sample(self, state: int, rover_pos: Dict[str, str], sample_type: str, sample_wp: str) -> int:
        """Lower bound for a communicated soil/rock goal at sample_wp."""
        # At least one communication action is needed if not yet communicated.
        base_comm = 1

        # If some rover already has the analysis for that waypoint, we just need to move to a comm waypoint.
        if sample_type == "soil":
            has_fact = self.have_soil_fact_id
        else:
            has_fact = self.have_rock_fact_id

        best = None
        for r, rwp in rover_pos.items():
            # If rover has data already
            data_fid = has_fact.get((r, sample_wp))
            if data_fid is not None and self._is_true(state, data_fid):
                nav = self._min_dist_to_any_comm_wp(r, rwp)
                cost = base_comm + nav
                best = cost if best is None else min(best, cost)

        if best is not None:
            return best

        # Otherwise, must (at least) navigate to sample waypoint, sample (1), then navigate to comm wp, then communicate.
        # We ignore store empty/full constraints (relaxation), so still a lower bound.
        best2 = None
        for r, rwp in rover_pos.items():
            d_to_sample = self._dist(r, rwp, sample_wp)
            if d_to_sample is None:
                continue
            d_sample_to_comm = self._min_dist_to_any_comm_wp(r, sample_wp)
            if d_sample_to_comm is None:
                continue
            # minimal: navigate to sample + sample action + navigate to comm + communicate
            cost = d_to_sample + 1 + d_sample_to_comm + base_comm
            best2 = cost if best2 is None else min(best2, cost)

        # If unreachable (shouldn't happen in benchmarks), return a large number
        return best2 if best2 is not None else 10**6

    def _lb_communicate_image(self, state: int, rover_pos: Dict[str, str], obj: str, mode: str) -> int:
        """Lower bound for a communicated image goal (obj, mode)."""
        base_comm = 1

        # If already have_image on some rover: just move to comm waypoint and communicate
        best = None
        for r, rwp in rover_pos.items():
            img_fid = self.have_image_fact_id.get((r, obj, mode))
            if img_fid is not None and self._is_true(state, img_fid):
                nav = self._min_dist_to_any_comm_wp(r, rwp)
                cost = base_comm + nav
                best = cost if best is None else min(best, cost)
        if best is not None:
            return best

        # Otherwise: must at least take_image once (1) somewhere visible_from(obj, wp).
        # In the relaxed lower bound we ignore calibration and just require reaching some photo waypoint.
        # Then navigate to comm waypoint and communicate.
        photo_wps = self._objective_visible_waypoints(obj)
        if not photo_wps:
            return 10**6

        best2 = None
        for r, rwp in rover_pos.items():
            # choose best photo waypoint
            for pw in photo_wps:
                d_to_photo = self._dist(r, rwp, pw)
                if d_to_photo is None:
                    continue
                d_photo_to_comm = self._min_dist_to_any_comm_wp(r, pw)
                if d_photo_to_comm is None:
                    continue
                cost = d_to_photo + 1 + d_photo_to_comm + base_comm
                best2 = cost if best2 is None else min(best2, cost)

        return best2 if best2 is not None else 10**6

    # ----------------- Helpers (bitwise + precomputed graph) -----------------

    def _is_true(self, state: int, fact_id: int) -> bool:
        return ((state >> fact_id) & 1) == 1

    def _extract_rover_positions(self, state: int) -> Dict[str, str]:
        """Return mapping rover->current waypoint using bitwise checks.

        Assumes exactly one at(rover,wp) is true per rover in valid states.
        """
        rover_pos: Dict[str, str] = {}
        # Iterate over known at-facts (fast enough: |rovers|*|wps| is too big; instead iterate dict items)
        for (r, w), fid in self.at_fact_id.items():
            if r in rover_pos:
                continue
            if self._is_true(state, fid):
                rover_pos[r] = w
        return rover_pos

    def _dist(self, rover: str, src: str, dst: str) -> Optional[int]:
        return self.dist.get(rover, {}).get(src, {}).get(dst, None)

    def _min_dist_to_any_comm_wp(self, rover: str, src: str) -> Optional[int]:
        if not self.comm_wps:
            return None
        best = None
        dmap = self.dist.get(rover, {}).get(src, {})
        for cw in self.comm_wps:
            d = dmap.get(cw, None)
            if d is None:
                continue
            best = d if best is None else min(best, d)
        return best

    def _objective_visible_waypoints(self, obj: str) -> List[str]:
        """Collect waypoints w with visible_from(obj,w) from the fact list.

        This is static and could be cached; we cache per objective lazily.
        """
        if not hasattr(self, "_obj_vis_cache"):
            self._obj_vis_cache = {}
        cache = self._obj_vis_cache
        if obj in cache:
            return cache[obj]

        wps: List[str] = []
        for f in self.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "visible_from" and len(args) == 2 and args[0] == obj:
                wps.append(args[1])
        cache[obj] = wps
        return wps

    def _bfs_dist(self, adj: Dict[str, List[str]], src: str) -> Dict[str, int]:
        dist: Dict[str, int] = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

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
