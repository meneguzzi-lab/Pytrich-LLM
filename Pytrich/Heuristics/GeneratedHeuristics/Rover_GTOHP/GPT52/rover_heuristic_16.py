from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT52Heuristic16(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain (Pytrich).

    RATIONALE / DOMAIN INSIGHTS
    ---------------------------
    The top-level goals in the provided instances are always *communication goals*:
      - communicated_soil_data(waypoint)
      - communicated_rock_data(waypoint)
      - communicated_image_data(objective, mode)

    In this domain, every communicated_* fact requires (at least) the following causal chain:
      Soil:
        sample_soil -> have_soil_analysis -> communicate_soil_data(1/2) -> communicated_soil_data
      Rock:
        sample_rock -> have_rock_analysis -> communicate_rock_data(1/2) -> communicated_rock_data
      Image:
        (calibrate -> take_image -> have_image) -> communicate_image_data -> communicated_image_data

    Additionally, communication requires the rover to be at some waypoint that is visible to the lander waypoint.
    Navigation is typically the dominant cost; however exact shortest paths per rover are expensive to compute online.

    HEURISTIC DESIGN
    ----------------
    We use a fast additive lower bound composed of:
      (A) Goal-achievement landmark counts:
          For each unsatisfied goal we add the minimal number of *remaining primitive actions* still needed,
          based on which intermediate facts already hold in the current state.
          This yields a consistent, admissible lower bound because it only counts actions that must occur.

      (B) Lightweight navigation-to-communicate lower bound:
          For each unsatisfied communicated_* goal, at least one navigation step may be needed to reach a
          waypoint that can communicate with the lander (i.e., visible to lander). We precompute the set of
          such "comm points" from static facts (visible, at_lander). If currently no rover is at any comm point,
          we add +1 (at least one navigate) as a global lower bound (not per-goal, to stay admissible).

    Notes:
      - We do not use model.state_explicit_repr(state) (too slow). All checks are bitwise by fact.global_id.
      - We avoid heavy graph search; precomputations are O(|facts|) once in __init__.
      - The heuristic naturally scales to large instances by relying on constant-time bit tests.

    Admissibility:
      - Component (A) counts required remaining actions per goal chain; it never overestimates.
      - Component (B) adds at most 1 for navigation if no rover is already at any possible comm point; this is
        a necessary condition to communicate anything, hence also a lower bound.

    """

    def __init__(self, model, rover_gpt52_heuristic_16: str = "rover_gpt52_heuristic_16"):
        super().__init__(model, rover_gpt52_heuristic_16)
        self.model = model

        # --- Fact caches (by signature) ---
        # Maps (predicate, tuple(args)) -> global_id
        self._fact_gid: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        # For fast lookup of specific communicated goals
        self._goal_comm_soil: List[str] = []  # waypoint
        self._goal_comm_rock: List[str] = []  # waypoint
        self._goal_comm_image: List[Tuple[str, str]] = []  # (objective, mode)

        # Support structures for image goals
        # (objective, mode) -> list of (rover, camera, waypoint) candidates such that
        # supports(camera,mode), on_board(camera,rover), visible_from(objective,waypoint)
        # (calibration is ignored in this candidate list; handled via action counting)
        self._img_candidates: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}

        # Store_of mapping: rover -> store
        self._store_of: Dict[str, str] = {}

        # Comm points: waypoints w such that exists lander l at w2 and visible(w,w2)
        self._comm_points: Set[str] = set()

        # Sets for quick state checks
        self._rovers: Set[str] = set()

        self._build_static_caches()
        self._parse_goals()

    # -------------------- Parsing helpers --------------------
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

    def _lit_gid(self, predicate: str, args: List[str]) -> Optional[int]:
        return self._fact_gid.get((predicate, tuple(args)))

    def _is_true_gid(self, state: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    # -------------------- Precomputation --------------------
    def _build_static_caches(self) -> None:
        # Index all facts by parsed signature
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._fact_gid[(pred, tuple(args))] = f.global_id

        # Derive rover set and store_of mapping from facts
        for (pred, args), gid in self._fact_gid.items():
            if pred == "store_of" and len(args) == 2:
                store, rover = args
                self._store_of[rover] = store
            elif pred == "at" and len(args) == 2:
                rover, _wp = args
                # In this domain, 'at' first arg is rover
                self._rovers.add(rover)

        # Precompute comm points from visible + at_lander
        lander_at: List[str] = []
        for (pred, args), gid in self._fact_gid.items():
            if pred == "at_lander" and len(args) == 2:
                _lander, w2 = args
                lander_at.append(w2)

        if lander_at:
            lander_wps = set(lander_at)
            for (pred, args), gid in self._fact_gid.items():
                if pred == "visible" and len(args) == 2:
                    w1, w2 = args
                    if w2 in lander_wps:
                        self._comm_points.add(w1)

        # Precompute image candidate tuples for all (objective,mode) pairs that might be goals.
        # We build a reusable index:
        #   visible_from(objective, waypoint)
        #   supports(camera, mode)
        #   on_board(camera, rover)
        vis_from: Dict[str, List[str]] = {}
        supports: Dict[str, Set[str]] = {}
        on_board: Dict[str, str] = {}

        for (pred, args), gid in self._fact_gid.items():
            if pred == "visible_from" and len(args) == 2:
                obj, wp = args
                vis_from.setdefault(obj, []).append(wp)
            elif pred == "supports" and len(args) == 2:
                cam, mode = args
                supports.setdefault(cam, set()).add(mode)
            elif pred == "on_board" and len(args) == 2:
                cam, rov = args
                on_board[cam] = rov

        # Build candidates lazily in _parse_goals using these indices stored temporarily
        self._static_vis_from = vis_from
        self._static_supports = supports
        self._static_on_board = on_board

    def _parse_goals(self) -> None:
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g in goal_fact_names:
            pred, args = self._parse_fact_name(g)
            if pred is None:
                continue
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goal_comm_soil.append(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goal_comm_rock.append(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goal_comm_image.append((args[0], args[1]))

        # Build image candidates only for (objective,mode) that appear in goals.
        for (obj, mode) in self._goal_comm_image:
            cands: List[Tuple[str, str, str]] = []
            for cam, modes in self._static_supports.items():
                if mode not in modes:
                    continue
                rov = self._static_on_board.get(cam)
                if rov is None:
                    continue
                for wp in self._static_vis_from.get(obj, []):
                    cands.append((rov, cam, wp))
            self._img_candidates[(obj, mode)] = cands

    # -------------------- Heuristic core --------------------
    def __call__(self, state: int, task_network=None) -> int:
        # If goals already achieved (ignoring TN), heuristic is 0.
        if (state & self.model.goals) == self.model.goals:
            return 0

        h = 0

        # (A) Per-goal remaining-action lower bounds
        # Soil goals
        for wp in self._goal_comm_soil:
            gid_goal = self._lit_gid("communicated_soil_data", [wp])
            if self._is_true_gid(state, gid_goal):
                continue

            gid_have = None
            # have_soil_analysis has a rover argument; we lower-bound by "exists rover".
            # If any rover has analysis at wp, then only communication remains.
            have_any = False
            for rov in self._rovers:
                gid_have = self._lit_gid("have_soil_analysis", [rov, wp])
                if self._is_true_gid(state, gid_have):
                    have_any = True
                    break

            if have_any:
                # Need at least one communicate action
                h += 1
            else:
                # Need at least: sample_soil + communicate_soil_data
                # (empty_store / navigate could be needed but are ignored to keep admissible.)
                h += 2

        # Rock goals
        for wp in self._goal_comm_rock:
            gid_goal = self._lit_gid("communicated_rock_data", [wp])
            if self._is_true_gid(state, gid_goal):
                continue

            have_any = False
            for rov in self._rovers:
                gid_have = self._lit_gid("have_rock_analysis", [rov, wp])
                if self._is_true_gid(state, gid_have):
                    have_any = True
                    break

            if have_any:
                h += 1
            else:
                h += 2  # sample_rock + communicate_rock_data

        # Image goals
        for (obj, mode) in self._goal_comm_image:
            gid_goal = self._lit_gid("communicated_image_data", [obj, mode])
            if self._is_true_gid(state, gid_goal):
                continue

            # If any rover already has the image, only communication remains
            have_img = False
            for rov in self._rovers:
                gid_have_img = self._lit_gid("have_image", [rov, obj, mode])
                if self._is_true_gid(state, gid_have_img):
                    have_img = True
                    break

            if have_img:
                h += 1
                continue

            # Otherwise need at least take_image + communicate.
            # Calibration is often required, but could already be calibrated; we add +0/+1 accordingly.
            # We lower-bound calibration need by checking if there exists any (rover,camera) that is calibrated.
            # If none, then at least one calibrate action must happen before any take_image.
            any_calibrated = False
            for (rov, cam, _wp) in self._img_candidates.get((obj, mode), []):
                gid_cal = self._lit_gid("calibrated", [cam, rov])
                if self._is_true_gid(state, gid_cal):
                    any_calibrated = True
                    break

            # Minimal remaining actions for image chain:
            #   if already calibrated somewhere: take_image + communicate = 2
            #   else: calibrate + take_image + communicate = 3
            h += 2 if any_calibrated else 3

        # (B) Global navigation-to-comm-point lower bound (at most +1)
        # If there is at least one unsatisfied communicated_* goal, and currently no rover is at any comm_point,
        # then at least one navigate action is needed before any communication can occur.
        if h > 0 and self._comm_points and (not self._any_rover_at_comm_point(state)):
            h += 1

        return h

    def _any_rover_at_comm_point(self, state: int) -> bool:
        # Checks if exists rover r, waypoint w in comm_points such that at(r,w) true.
        for rov in self._rovers:
            for wp in self._comm_points:
                gid_at = self._lit_gid("at", [rov, wp])
                if self._is_true_gid(state, gid_at):
                    return True
        return False
