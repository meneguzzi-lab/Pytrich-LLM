from __future__ import annotations

from collections import deque
from typing import Dict, List, Tuple, Optional

# Pytrich imports (available in the evaluation environment)
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic3(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic features)
    --------------------------------------------------------
    Rover goal achievement in this HDDL domain always follows one of three pipelines:

      Soil pipeline (for a waypoint p):
        do_navigate1(rover, p) -> empty_store(store, rover) -> sample_soil -> send_soil_data
      Rock pipeline (for a waypoint p):
        do_navigate1(rover, p) -> empty_store -> sample_rock -> send_rock_data
      Image pipeline (for objective o, mode m):
        do_calibrate(rover, camera) -> do_navigate1(rover, w) -> take_image -> send_image_data

    Key planning challenges captured:
      1) Navigation dominates cost: do_navigate1/2 decomposes into sequences of navigate actions.
         We lower-bound this by shortest-path distance on the can_traverse graph (ignoring visited
         bookkeeping and visibility constraints).
      2) Communication requires being at some waypoint visible to the lander waypoint.
         We precompute a set of "comm points" (waypoints w with visible(w, lander_wp)).
         A lower bound includes moving from data location to some comm point (or direct comm if possible).
      3) Store capacity: sampling requires an empty store; if the store is full, at least one drop is needed.
         We add +1 when a required sampling is pending and no store is currently empty (relaxed LB).
      4) Imaging requires a camera supporting the mode and a calibration target; calibration consumes a step.
         We add a small constant lower bound for pending image goals (at least take_image+communicate,
         and +1 if no calibrated camera is currently available).

    Heuristic form:
      h(state) = sum over unsatisfied goals of per-goal lower bounds

    This is NOT guaranteed admissible (because of summation/double counting shared moves), but it is
    intentionally domain-guided to reduce node expansions in practice.

    Performance considerations:
      - Never uses explicit state string lists for checking truth; all checks are bitwise.
      - Pre-indexes fact ids for key predicates and parses model facts once.
      - Precomputes all-pairs shortest paths over waypoints per rover using BFS on can_traverse graph
        (unit cost edges). This is feasible in typical rover benchmarks.

    Notes about grounded fact format:
      Facts are like '+at[rover0,waypoint1]'. We parse these into (pred, args).
    """

    def __init__(self, model, rover_gpt52_heuristic_3: str = "rover_gpt52_heuristic_3"):
        super().__init__(model, rover_gpt52_heuristic_3)
        self.model = model

        # ---------- Parse all facts once ----------
        self._fact_by_name: Dict[str, int] = {}
        # predicate -> list of (args_tuple, global_id)
        self._facts_by_pred: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}

        for f in self.model.facts:
            self._fact_by_name[f.name] = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((tuple(args), f.global_id))

        # ---------- Collect objects from facts (robust without relying on model typing API) ----------
        self.rovers = self._collect_unary_arg0("available") | self._collect_unary_arg0("equipped_for_soil_analysis") | \
                      self._collect_unary_arg0("equipped_for_rock_analysis") | self._collect_unary_arg0("equipped_for_imaging")
        self.waypoints = self._collect_waypoints()
        self.landers = self._collect_unary_arg0("channel_free") | self._collect_unary_arg0("at_lander")

        # ---------- Lander waypoint & comm points ----------
        self.lander_at: Dict[str, str] = {}  # lander -> waypoint
        for (args, _) in self._facts_by_pred.get("at_lander", []):
            l, w = args
            self.lander_at[l] = w

        # For simplicity, use the first lander if multiple exist.
        self.default_lander: Optional[str] = next(iter(self.lander_at.keys()), None)
        self.lander_wp: Optional[str] = self.lander_at.get(self.default_lander) if self.default_lander else None

        # comm_points: waypoints w such that visible(w, lander_wp)
        self.comm_points: List[str] = []
        if self.lander_wp is not None:
            for (args, _) in self._facts_by_pred.get("visible", []):
                w1, w2 = args
                if w2 == self.lander_wp:
                    self.comm_points.append(w1)
        self.comm_points = list(sorted(set(self.comm_points)))

        # ---------- Store mapping and empty/full fact ids ----------
        # store_of[s, r]
        self.store_of: Dict[str, str] = {}  # rover -> store
        for (args, _) in self._facts_by_pred.get("store_of", []):
            s, r = args
            # If multiple stores per rover appear, keep first.
            self.store_of.setdefault(r, s)

        # Facts ids for empty/full by store
        self.empty_fact: Dict[str, int] = {}
        self.full_fact: Dict[str, int] = {}
        for (args, gid) in self._facts_by_pred.get("empty", []):
            (s,) = args
            self.empty_fact[s] = gid
        for (args, gid) in self._facts_by_pred.get("full", []):
            (s,) = args
            self.full_fact[s] = gid

        # ---------- Equipment sets ----------
        self.soil_capable = self._collect_unary_arg0("equipped_for_soil_analysis")
        self.rock_capable = self._collect_unary_arg0("equipped_for_rock_analysis")
        self.img_capable = self._collect_unary_arg0("equipped_for_imaging")

        # ---------- Cameras and supports ----------
        # on_board[c, r]
        self.cameras_on_rover: Dict[str, List[str]] = {}
        for (args, _) in self._facts_by_pred.get("on_board", []):
            c, r = args
            self.cameras_on_rover.setdefault(r, []).append(c)
        for r in list(self.cameras_on_rover.keys()):
            self.cameras_on_rover[r] = list(sorted(set(self.cameras_on_rover[r])))

        # supports[c, m]
        self.camera_supports: Dict[str, set] = {}
        for (args, _) in self._facts_by_pred.get("supports", []):
            c, m = args
            self.camera_supports.setdefault(c, set()).add(m)

        # calibration_target[c, o]
        self.calib_target: Dict[str, str] = {}
        for (args, _) in self._facts_by_pred.get("calibration_target", []):
            c, o = args
            self.calib_target[c] = o

        # visible_from[o, w]
        self.visible_from: Dict[str, List[str]] = {}
        for (args, _) in self._facts_by_pred.get("visible_from", []):
            o, w = args
            self.visible_from.setdefault(o, []).append(w)
        for o in list(self.visible_from.keys()):
            self.visible_from[o] = list(sorted(set(self.visible_from[o])))

        # calibrated[c, r] fact ids
        self.calibrated_fact: Dict[Tuple[str, str], int] = {}
        for (args, gid) in self._facts_by_pred.get("calibrated", []):
            c, r = args
            self.calibrated_fact[(c, r)] = gid

        # ---------- Position fact ids (at rover waypoint) ----------
        # at[r, w]
        self.at_fact: Dict[Tuple[str, str], int] = {}
        for (args, gid) in self._facts_by_pred.get("at", []):
            x, w = args
            self.at_fact[(x, w)] = gid

        # ---------- Goal facts list ----------
        self.goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self.goal_facts: List[int] = []
        self.goal_atoms: List[Tuple[str, Tuple[str, ...], int]] = []  # (pred, args, gid)
        for name in self.goal_fact_names:
            gid = self._fact_by_name.get(name)
            if gid is None:
                continue
            pred, args = self._parse_fact_name(name)
            self.goal_facts.append(gid)
            self.goal_atoms.append((pred, tuple(args), gid))

        # ---------- Precompute rover waypoint graph distances ----------
        # adjacency per rover: waypoint -> neighbors
        self.rover_adj: Dict[str, Dict[str, List[str]]] = {}
        for r in self.rovers:
            self.rover_adj[r] = {w: [] for w in self.waypoints}

        for (args, _) in self._facts_by_pred.get("can_traverse", []):
            r, w1, w2 = args
            if r in self.rover_adj and w1 in self.rover_adj[r]:
                self.rover_adj[r][w1].append(w2)

        # distances per rover: (from_wp -> dict(to_wp->dist))
        self.rover_dists: Dict[str, Dict[str, Dict[str, int]]] = {}
        for r in self.rovers:
            self.rover_dists[r] = {}
            for w in self.waypoints:
                self.rover_dists[r][w] = self._bfs_dist(self.rover_adj[r], w)

    # ---------------- Heuristic API ----------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        # If all goal facts true, heuristic 0 (even if TN not empty; search algorithm may handle TN separately)
        if self.model.goals <= state_bitwise:
            return 0

        # Current rover positions
        rover_pos: Dict[str, str] = {}
        for r in self.rovers:
            wp = self._get_rover_position(state_bitwise, r)
            if wp is not None:
                rover_pos[r] = wp

        # Store emptiness per rover (best-effort)
        rover_store_empty: Dict[str, bool] = {}
        rover_store_full: Dict[str, bool] = {}
        for r, s in self.store_of.items():
            e_id = self.empty_fact.get(s)
            f_id = self.full_fact.get(s)
            rover_store_empty[r] = bool(e_id is not None and ((state_bitwise >> e_id) & 1))
            rover_store_full[r] = bool(f_id is not None and ((state_bitwise >> f_id) & 1))

        # Calibrated cameras on rover
        rover_has_calibrated: Dict[str, bool] = {}
        for r, cams in self.cameras_on_rover.items():
            rover_has_calibrated[r] = any(
                ((state_bitwise >> self.calibrated_fact[(c, r)]) & 1)
                for c in cams
                if (c, r) in self.calibrated_fact
            )

        h = 0

        for pred, args, gid in self.goal_atoms:
            if ((state_bitwise >> gid) & 1):
                continue

            if pred == "communicated_soil_data":
                (p,) = args
                h += self._lb_communicate_soil(state_bitwise, rover_pos, rover_store_empty, rover_store_full, p)
            elif pred == "communicated_rock_data":
                (p,) = args
                h += self._lb_communicate_rock(state_bitwise, rover_pos, rover_store_empty, rover_store_full, p)
            elif pred == "communicated_image_data":
                o, m = args
                h += self._lb_communicate_image(state_bitwise, rover_pos, rover_has_calibrated, o, m)
            else:
                # Fallback: unsatisfied goal contributes 1
                h += 1

        return int(h)

    # ---------------- Lower bound components ----------------
    def _lb_communicate_soil(self, state: int, rover_pos: Dict[str, str], empty: Dict[str, bool], full: Dict[str, bool], p: str) -> int:
        # If already have communicated soil -> 0 handled earlier
        # If some rover already has soil analysis at p, we just need to move to comm and communicate.
        # Otherwise: need navigate to p, maybe drop, sample, then comm.

        # Identify rovers that can do soil
        candidates = [r for r in self.rovers if r in self.soil_capable and r in rover_pos]
        if not candidates:
            return 10  # dead-ish

        have_fact_ids = []
        for (a, gid) in self._facts_by_pred.get("have_soil_analysis", []):
            r, wp = a
            if wp == p:
                have_fact_ids.append((r, gid))

        best = 10**9
        for r in candidates:
            rp = rover_pos[r]
            d_to_p = self._dist(r, rp, p)

            # Already have analysis?
            has_analysis = False
            for rr, fid in have_fact_ids:
                if rr == r and ((state >> fid) & 1):
                    has_analysis = True
                    break

            if has_analysis:
                # Need go to some comm point and communicate (>=1 action)
                best = min(best, self._dist_to_comm(r, rp) + 1)
                continue

            # Need sample: at least sample action + comm action (+ optional drop)
            drop_lb = 1 if (r in full and full[r] and not empty.get(r, False)) else 0
            # need to get to p then sample then go to comm and communicate
            lb = d_to_p + drop_lb + 1 + self._dist_to_comm(r, p) + 1
            best = min(best, lb)

        return 0 if best == 10**9 else best

    def _lb_communicate_rock(self, state: int, rover_pos: Dict[str, str], empty: Dict[str, bool], full: Dict[str, bool], p: str) -> int:
        candidates = [r for r in self.rovers if r in self.rock_capable and r in rover_pos]
        if not candidates:
            return 10

        have_fact_ids = []
        for (a, gid) in self._facts_by_pred.get("have_rock_analysis", []):
            r, wp = a
            if wp == p:
                have_fact_ids.append((r, gid))

        best = 10**9
        for r in candidates:
            rp = rover_pos[r]
            d_to_p = self._dist(r, rp, p)

            has_analysis = False
            for rr, fid in have_fact_ids:
                if rr == r and ((state >> fid) & 1):
                    has_analysis = True
                    break

            if has_analysis:
                best = min(best, self._dist_to_comm(r, rp) + 1)
                continue

            drop_lb = 1 if (r in full and full[r] and not empty.get(r, False)) else 0
            lb = d_to_p + drop_lb + 1 + self._dist_to_comm(r, p) + 1
            best = min(best, lb)

        return 0 if best == 10**9 else best

    def _lb_communicate_image(self, state: int, rover_pos: Dict[str, str], rover_has_calibrated: Dict[str, bool], o: str, m: str) -> int:
        # Need have_image[r,o,m] then communicate from comm point.
        # To take image: must be at some waypoint w with visible_from(o,w), and camera on rover supports m.
        # Calibration required unless already calibrated.

        # Find candidate waypoints for taking image
        ws = self.visible_from.get(o, [])
        if not ws:
            return 10

        # Find rovers with imaging capability and some camera supporting mode
        candidates = []
        for r in self.rovers:
            if r not in self.img_capable or r not in rover_pos:
                continue
            cams = self.cameras_on_rover.get(r, [])
            if any(m in self.camera_supports.get(c, set()) for c in cams):
                candidates.append(r)
        if not candidates:
            return 10

        # Check if already have_image
        have_img_facts = []
        for (a, gid) in self._facts_by_pred.get("have_image", []):
            r, oo, mm = a
            if oo == o and mm == m:
                have_img_facts.append((r, gid))

        best = 10**9
        for r in candidates:
            rp = rover_pos[r]

            has_img = False
            for rr, fid in have_img_facts:
                if rr == r and ((state >> fid) & 1):
                    has_img = True
                    break

            if has_img:
                best = min(best, self._dist_to_comm(r, rp) + 1)
                continue

            # Need to take image: calibrate (>=1) if none calibrated, then move to some w, take image (1), move to comm, communicate (1)
            calib_lb = 0 if rover_has_calibrated.get(r, False) else 1
            d_to_best_w = min(self._dist(r, rp, w) for w in ws)
            lb = calib_lb + d_to_best_w + 1 + self._dist_to_comm(r, ws[0] if d_to_best_w == 10**6 else self._argmin_wp(r, rp, ws)) + 1
            best = min(best, lb)

        return 0 if best == 10**9 else best

    # ---------------- Utility methods ----------------
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

    def _collect_unary_arg0(self, pred: str) -> set:
        s = set()
        for (args, _) in self._facts_by_pred.get(pred, []):
            if len(args) == 1:
                s.add(args[0])
            elif len(args) >= 1:
                s.add(args[0])
        return s

    def _collect_waypoints(self) -> set:
        wps = set()
        # Use all occurrences in visible, can_traverse, at_lander, visible_from, at
        for (args, _) in self._facts_by_pred.get("visible", []):
            if len(args) == 2:
                wps.add(args[0]); wps.add(args[1])
        for (args, _) in self._facts_by_pred.get("can_traverse", []):
            if len(args) == 3:
                wps.add(args[1]); wps.add(args[2])
        for (args, _) in self._facts_by_pred.get("at_lander", []):
            if len(args) == 2:
                wps.add(args[1])
        for (args, _) in self._facts_by_pred.get("visible_from", []):
            if len(args) == 2:
                wps.add(args[1])
        for (args, _) in self._facts_by_pred.get("at", []):
            if len(args) == 2:
                wps.add(args[1])
        return wps

    def _get_rover_position(self, state: int, rover: str) -> Optional[str]:
        # Scan over waypoints using preindexed at_fact ids (fast-ish)
        # Prefer using at_fact mapping rather than parsing state.
        for w in self.waypoints:
            fid = self.at_fact.get((rover, w))
            if fid is None:
                continue
            if (state >> fid) & 1:
                return w
        return None

    def _bfs_dist(self, adj: Dict[str, List[str]], start: str) -> Dict[str, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _dist(self, rover: str, w1: str, w2: str) -> int:
        if w1 == w2:
            return 0
        d = self.rover_dists.get(rover, {}).get(w1, {}).get(w2)
        if d is None:
            return 10**6
        return d

    def _dist_to_comm(self, rover: str, from_wp: str) -> int:
        if self.lander_wp is None or not self.comm_points:
            return 10**6
        best = 10**6
        for cp in self.comm_points:
            best = min(best, self._dist(rover, from_wp, cp))
        return best

    def _argmin_wp(self, rover: str, from_wp: str, wps: List[str]) -> str:
        best_w = wps[0]
        best_d = 10**9
        for w in wps:
            d = self._dist(rover, from_wp, w)
            if d < best_d:
                best_d = d
                best_w = w
        return best_w
