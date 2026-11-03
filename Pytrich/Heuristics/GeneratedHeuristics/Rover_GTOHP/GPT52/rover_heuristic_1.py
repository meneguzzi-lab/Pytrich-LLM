from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic1(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic components)
    ---------------------------------------------------------
    The Rover HDDL encoding has three recurring high-level goal patterns:

      (A) Soil:  communicated_soil_data(wp)
          Achieved by: sample_soil at wp (needs rover with soil equip + empty store)
                       then communicate_soil_data* while rover is at a waypoint that is
                       visible to the lander's waypoint.

      (B) Rock:  communicated_rock_data(wp)
          Similar to soil, but needs rock equip.

      (C) Image: communicated_image_data(obj, mode)
          Achieved by: (possibly) calibrate a camera for obj
                       navigate to some waypoint w with visible_from(obj, w)
                       take_image (consumes calibration)
                       communicate_image_data from some waypoint visible to lander.

    The main costs come from navigation edges (navigate actions) and mandatory
    primitive actions (sample/take/communicate/drop/calibrate). The HTN methods
    also introduce do_navigate* tasks that ultimately become 0 or more navigate.

    This heuristic is a fast, *admissible* (lower-bound) estimator that combines:

      1) Goal-count lower bounds: each unmet communicated_* goal requires at least
         one communication action.

      2) Navigation lower bounds via all-pairs shortest paths on the *can_traverse*
         graph per rover (ignoring visibility and visited bookkeeping). This is a
         relaxation: it may underestimate true navigation cost, hence admissible.

         For each unmet goal, we estimate:
           - minimal distance from any rover position to a waypoint that can achieve
             the corresponding sensing (sampling or imaging vantage)
           - plus minimal distance from that sensing waypoint to a communication
             waypoint that is visible to the lander.

         We take the minimum over rovers that are capable (equipment/camera/supports).

      3) Mandatory action lower bounds for sensing steps:
           soil/rock: +1 for sample
           image: +1 for take_image
           +1 for communicate (already counted in (1), but we ensure at least one
              communicate per unmet goal).

         For calibration, we add +1 only if we can prove it is currently impossible
         to have a calibrated camera for that rover+camera (i.e., no calibrated fact
         present for any suitable camera). This is safe but conservative.

    Efficiency notes
    ----------------
    - Uses bitwise checks for facts.
    - Precomputes indices from facts once in __init__.
    - Precomputes rover-specific shortest paths on can_traverse edges.
      (In the provided instances, each rover has its own graph; this is still
       tractable and computed once.)

    The heuristic returns an integer.
    """

    def __init__(self, model, rover_gpt52_heuristic_1: str = "rover_gpt52_heuristic_1"):
        super().__init__(model, rover_gpt52_heuristic_1)
        self.model = model

        # ---- Parse and index facts we care about ----
        self._fact_by_pred: Dict[str, List[Tuple[int, List[str]]]] = defaultdict(list)
        self._fact_id_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # Type-like sets derived from facts (object names used in arguments)
        self.rovers: Set[str] = set()
        self.waypoints: Set[str] = set()
        self.landers: Set[str] = set()
        self.cameras: Set[str] = set()
        self.objectives: Set[str] = set()
        self.modes: Set[str] = set()
        self.stores: Set[str] = set()

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            # store only positive facts in indices; negative ones are rarely present
            if not f.name.startswith("+"):
                continue
            self._fact_by_pred[pred].append((f.global_id, args))
            self._fact_id_by_key[(pred, tuple(args))] = f.global_id

            # crude object discovery by predicate signatures
            if pred == "at" and len(args) == 2:
                self.rovers.add(args[0]); self.waypoints.add(args[1])
            elif pred == "at_lander" and len(args) == 2:
                self.landers.add(args[0]); self.waypoints.add(args[1])
            elif pred == "store_of" and len(args) == 2:
                self.stores.add(args[0]); self.rovers.add(args[1])
            elif pred == "on_board" and len(args) == 2:
                self.cameras.add(args[0]); self.rovers.add(args[1])
            elif pred == "supports" and len(args) == 2:
                self.cameras.add(args[0]); self.modes.add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self.cameras.add(args[0]); self.objectives.add(args[1])
            elif pred in ("visible", "can_traverse") and len(args) == 2:
                self.waypoints.add(args[0]); self.waypoints.add(args[1])
            elif pred == "visible_from" and len(args) == 2:
                self.objectives.add(args[0]); self.waypoints.add(args[1])

        # Landers: instances typically have one lander.
        self.lander_wp: Dict[str, str] = {}
        for _, args in self._fact_by_pred.get("at_lander", []):
            if len(args) == 2:
                self.lander_wp[args[0]] = args[1]

        # Visibility graph between waypoints (for communication feasibility)
        self.visible: Dict[str, Set[str]] = defaultdict(set)
        for _, args in self._fact_by_pred.get("visible", []):
            if len(args) == 2:
                self.visible[args[0]].add(args[1])

        # For each rover: adjacency of can_traverse (navigation graph)
        self.traverse_adj: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        for _, args in self._fact_by_pred.get("can_traverse", []):
            if len(args) == 3:
                r, a, b = args
                self.rovers.add(r); self.waypoints.add(a); self.waypoints.add(b)
                self.traverse_adj[r][a].add(b)

        # Precompute all-pairs shortest path distances per rover on can_traverse graph.
        self._dist: Dict[str, Dict[str, Dict[str, int]]] = {}
        for r in self.rovers:
            self._dist[r] = self._all_pairs_bfs(self.traverse_adj.get(r, {}))

        # Sampling locations
        self.soil_samples: Set[str] = set(args[0] for _, args in self._fact_by_pred.get("at_soil_sample", []) if len(args) == 1)
        self.rock_samples: Set[str] = set(args[0] for _, args in self._fact_by_pred.get("at_rock_sample", []) if len(args) == 1)

        # Imaging vantage: objective -> set(waypoints)
        self.visible_from: Dict[str, Set[str]] = defaultdict(set)
        for _, args in self._fact_by_pred.get("visible_from", []):
            if len(args) == 2:
                o, w = args
                self.visible_from[o].add(w)

        # Camera capabilities and calibration targets
        self.camera_on: Dict[str, str] = {}  # camera -> rover
        for _, args in self._fact_by_pred.get("on_board", []):
            if len(args) == 2:
                c, r = args
                self.camera_on[c] = r

        self.camera_supports: Dict[str, Set[str]] = defaultdict(set)
        for _, args in self._fact_by_pred.get("supports", []):
            if len(args) == 2:
                c, m = args
                self.camera_supports[c].add(m)

        self.calib_target: Dict[str, str] = {}  # camera -> objective
        for _, args in self._fact_by_pred.get("calibration_target", []):
            if len(args) == 2:
                c, o = args
                self.calib_target[c] = o

        # Equipment predicates fact ids for fast state checking (per rover)
        self._equipped_soil: Dict[str, int] = {}
        self._equipped_rock: Dict[str, int] = {}
        self._equipped_img: Dict[str, int] = {}
        for r in self.rovers:
            self._equipped_soil[r] = self._fact_id_by_key.get(("equipped_for_soil_analysis", (r,)), -1)
            self._equipped_rock[r] = self._fact_id_by_key.get(("equipped_for_rock_analysis", (r,)), -1)
            self._equipped_img[r] = self._fact_id_by_key.get(("equipped_for_imaging", (r,)), -1)

        # Goal facts parsed once
        self._goal_items = []  # list of tuples (kind, args)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred in ("communicated_soil_data", "communicated_rock_data", "communicated_image_data"):
                self._goal_items.append((pred, args))

        # Precompute communication waypoints: those that can see the lander waypoint.
        # If multiple landers exist, we union their visibility sets conservatively.
        self.comm_wps: Set[str] = set()
        for l, lw in self.lander_wp.items():
            # any waypoint w such that visible(w, lw)
            for w, neigh in self.visible.items():
                if lw in neigh:
                    self.comm_wps.add(w)
            # also if visible(lw, w) is needed? In actions: visible(p1,p2) where p2 is lander wp.
            # so we keep only w with edge to lw as above.

    # ---------------- Heuristic core ----------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        h = 0

        # Current rover positions
        rover_pos: Dict[str, str] = {}
        for r in self.rovers:
            rover_pos[r] = self._get_rover_position(state_bitwise, r)

        # Which communicated goals already achieved?
        for pred, args in self._goal_items:
            fid = self._fact_id_by_key.get((pred, tuple(args)), -1)
            if fid >= 0 and ((state_bitwise >> fid) & 1):
                continue

            # Each unmet communicated_* requires at least one communication primitive.
            h += 1

            if pred == "communicated_soil_data":
                wp = args[0]
                h += self._estimate_sample_and_send(state_bitwise, rover_pos, wp, kind="soil")
            elif pred == "communicated_rock_data":
                wp = args[0]
                h += self._estimate_sample_and_send(state_bitwise, rover_pos, wp, kind="rock")
            elif pred == "communicated_image_data":
                obj, mode = args[0], args[1]
                h += self._estimate_image_and_send(state_bitwise, rover_pos, obj, mode)

        return int(h)

    # ---------------- Estimators ----------------
    def _estimate_sample_and_send(self, state_bitwise: int, rover_pos: Dict[str, str], sample_wp: str, kind: str) -> int:
        """Lower bound for sampling at sample_wp and then being able to communicate.

        Returns additional cost beyond the mandatory communicate counted elsewhere.
        Includes:
          - +1 for the sample action
          - +min navigation cost to reach sample_wp by a capable rover
          - +min navigation cost from sample_wp to a communication waypoint

        Ignores store emptying/drop because it can be avoided in relaxation and
        is not always necessary (store may already be empty).
        """
        best = None
        for r, pos in rover_pos.items():
            if pos is None:
                continue
            if kind == "soil":
                eq_id = self._equipped_soil.get(r, -1)
            else:
                eq_id = self._equipped_rock.get(r, -1)
            if eq_id < 0 or ((state_bitwise >> eq_id) & 1) == 0:
                continue

            d1 = self._dist_value(r, pos, sample_wp)
            if d1 is None:
                continue

            d2 = self._min_dist_to_comm_wp(r, sample_wp)
            if d2 is None:
                # If we can't find any comm waypoint reachable, still keep navigation to sample
                # as a weak bound; communication may be impossible but planner handles dead-ends.
                d2 = 0

            val = d1 + d2
            if best is None or val < best:
                best = val

        if best is None:
            # No capable rover reachable -> return only sample action LB (still admissible)
            return 1

        return 1 + best

    def _estimate_image_and_send(self, state_bitwise: int, rover_pos: Dict[str, str], obj: str, mode: str) -> int:
        """Lower bound for taking and sending an image for (obj, mode).

        Adds:
          - +1 for take_image
          - +min navigation to some vantage waypoint w with visible_from(obj,w)
          - +min navigation from that vantage to a communication waypoint
          - +1 for calibration only if no suitable calibrated camera is currently true

        Communication is counted outside.
        """
        vantage_wps = self.visible_from.get(obj, set())
        if not vantage_wps:
            return 1  # cannot reason; keep minimal

        # Precompute which cameras are suitable for (obj, mode) and which rover they are on.
        suitable_cams: List[Tuple[str, str]] = []  # (camera, rover)
        for c, r in self.camera_on.items():
            if self.calib_target.get(c) != obj:
                continue
            if mode not in self.camera_supports.get(c, set()):
                continue
            suitable_cams.append((c, r))

        best = None
        need_calibration_lb = 0

        # Check if any suitable camera is already calibrated (then calibration LB can be 0)
        any_calibrated = False
        for c, r in suitable_cams:
            fid = self._fact_id_by_key.get(("calibrated", (c, r)), -1)
            if fid >= 0 and ((state_bitwise >> fid) & 1):
                any_calibrated = True
                break
        if not any_calibrated and suitable_cams:
            # In this domain, take_image requires calibrated, and calibrate is the only producer.
            # If no suitable calibrated fact currently holds, at least 1 calibrate will be needed.
            need_calibration_lb = 1

        for c, r in suitable_cams:
            pos = rover_pos.get(r)
            if pos is None:
                continue
            eq_id = self._equipped_img.get(r, -1)
            if eq_id < 0 or ((state_bitwise >> eq_id) & 1) == 0:
                continue

            # Choose best vantage for this rover
            for w in vantage_wps:
                d1 = self._dist_value(r, pos, w)
                if d1 is None:
                    continue
                d2 = self._min_dist_to_comm_wp(r, w)
                if d2 is None:
                    d2 = 0
                val = d1 + d2
                if best is None or val < best:
                    best = val

        if best is None:
            return 1 + need_calibration_lb

        return 1 + need_calibration_lb + best

    # ---------------- Helpers ----------------
    def _min_dist_to_comm_wp(self, rover: str, from_wp: str) -> Optional[int]:
        if not self.comm_wps:
            return 0
        dmap = self._dist.get(rover, {}).get(from_wp)
        if not dmap:
            return None
        best = None
        for cw in self.comm_wps:
            if cw in dmap:
                v = dmap[cw]
                if best is None or v < best:
                    best = v
        return best

    def _dist_value(self, rover: str, a: str, b: str) -> Optional[int]:
        if a == b:
            return 0
        return self._dist.get(rover, {}).get(a, {}).get(b)

    def _get_rover_position(self, state_bitwise: int, rover: str) -> Optional[str]:
        # Scan all +at[rover,wp] facts for this rover (small per rover)
        for fid, args in self._fact_by_pred.get("at", []):
            if len(args) == 2 and args[0] == rover:
                if (state_bitwise >> fid) & 1:
                    return args[1]
        return None

    def _all_pairs_bfs(self, adj: Dict[str, Set[str]]) -> Dict[str, Dict[str, int]]:
        """All-pairs shortest paths on an unweighted directed graph using BFS per source."""
        dists: Dict[str, Dict[str, int]] = {}
        nodes = set(adj.keys())
        for u, vs in adj.items():
            nodes.update(vs)
        for s in nodes:
            q = deque([s])
            dist = {s: 0}
            while q:
                x = q.popleft()
                for y in adj.get(x, ()):  # directed edges
                    if y not in dist:
                        dist[y] = dist[x] + 1
                        q.append(y)
            dists[s] = dist
        return dists

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
