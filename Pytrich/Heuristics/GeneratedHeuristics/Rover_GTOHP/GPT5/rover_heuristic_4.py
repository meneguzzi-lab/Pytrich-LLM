from typing import Dict, Set, Tuple, List, Optional

try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback base class if environment injects Heuristic differently
    class Heuristic:
        def __init__(self, model, name: str = "heuristic"):
            self.model = model
            self.name = name
        def __call__(self, state_bitwise: int, task_network=None) -> int:
            return self.evaluate(state_bitwise, task_network or [])
        def evaluate(self, state_bitwise: int, task_network=None) -> int:
            return 0


class RoverGPT5Heuristic4(Heuristic):
    """
    Domain-dependent heuristic for HDDL Rover domain.

    Design rationale (admissible, scalable lower bound):
    - Every goal communicated_* requires at least one communication action (additive, per-goal, necessary).
    - Soil/Rock goals additionally require a sample at the target waypoint if no rover already has the analysis for that waypoint.
      We therefore add +1 sample per such waypoint not yet analyzed by any rover. (No navigation or store-empty penalties are added
      to keep it a safe lower bound.)
    - Image goals require having the image: if not already available, at least one take_image is required per (objective, mode).
      Also, each take_image consumes calibration; hence at least one calibrate is needed per new image, except when a suitable
      camera is already calibrated. To remain admissible across multiple image goals, we aggregate calibrations by mode:
        calibrate_needed = sum_m max(0, pending_images_with_mode_m - currently_calibrated_cameras_supporting_m)
      This ensures we never overcount calibrations even if several pending images could use the same currently calibrated camera.
    - We deliberately avoid summing navigation costs per goal, because navigations can often be shared across multiple goals in
      optimal solutions (e.g., sample several waypoints, then navigate once to a lander-visible waypoint and communicate all).
      Adding per-goal navigation would risk overestimating, thus breaking admissibility. The chosen lower bound stays informative
      by focusing on strictly necessary primitive actions that cannot be shared away: one communicate per goal and one domain-
      specific producer action per data item when it is not already available (sample/take_image and the minimal number of
      calibrations).

    Efficiency:
    - All static structures (objects, mappings from grounded facts, goal structures, visibility, cameras, supports, etc.) are
      pre-indexed once in __init__.
    - State queries use fast bitwise membership checks via global_id shifting, never expanding the explicit state at runtime.

    This heuristic is robust across scales (small to very large instances) and respects the Pytrich bitset model.
    """

    def __init__(self, model, rover_gpt5_heuristic_4: bool = True):
        super().__init__(model, name="RoverGPT5Heuristic4")
        self._init_done = False
        # Storage for indices
        self._facts_by_pred: Dict[str, List] = {}
        self._comm_soil_goal_map: Dict[str, int] = {}       # wp -> fact.global_id
        self._comm_rock_goal_map: Dict[str, int] = {}       # wp -> fact.global_id
        self._comm_image_goal_map: Dict[Tuple[str, str], int] = {}  # (obj,mode) -> fact.global_id

        self._have_soil_map: Dict[Tuple[str, str], int] = {}  # (rover, wp) -> fact.global_id
        self._have_rock_map: Dict[Tuple[str, str], int] = {}
        self._have_image_map: Dict[Tuple[str, str, str], int] = {}  # (rover, obj, mode) -> fact.global_id

        self._equipped_soil: Set[str] = set()  # rovers
        self._equipped_rock: Set[str] = set()
        self._equipped_imaging: Set[str] = set()

        self._on_board: Dict[str, str] = {}  # camera -> rover
        self._supports: Dict[str, Set[str]] = {}  # camera -> set(modes)
        self._calibrated_fact: Dict[Tuple[str, str], int] = {}  # (camera, rover) -> fact.global_id

        self._rovers: Set[str] = set()
        self._waypoints: Set[str] = set()
        self._objectives: Set[str] = set()
        self._modes: Set[str] = set()

        self._at_fact_by_rover: Dict[str, List[int]] = {}  # rover -> list of fact.global_id for all at[rover,*]
        self._at_facts_detail: Dict[int, Tuple[str, str]] = {}  # fact.global_id -> (rover, waypoint)

        self._visible_pairs: Set[Tuple[str, str]] = set()  # (w1,w2)
        self._visible_from: Dict[str, Set[str]] = {}  # objective -> set(waypoints)
        self._lander_wps: Set[str] = set()  # lander locations
        self._vtl_wps: Set[str] = set()  # waypoints visible to any lander location

        self._goals_comm_soil: Set[str] = set()  # waypoint
        self._goals_comm_rock: Set[str] = set()
        self._goals_comm_image: Set[Tuple[str, str]] = set()  # (objective, mode)

        self._build_indices()
        self._init_done = True

    # -------------------------------
    # Core API
    # -------------------------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        return self.evaluate(state_bitwise, task_network or [])

    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        # Quick goal check
        if self.model.goal_reached(state_bitwise, task_network or []):
            return 0

        # Count pending goal facts
        comm_needed = 0
        soil_sample_needed = 0
        rock_sample_needed = 0

        # Communication soil
        for wp in self._goals_comm_soil:
            fgid = self._comm_soil_goal_map.get(wp, None)
            if fgid is None:
                continue
            if not self._is_true(state_bitwise, fgid):
                comm_needed += 1
                # If no rover currently has soil analysis for this wp, sampling is necessary at least once
                if not self._exists_have_soil(state_bitwise, wp):
                    soil_sample_needed += 1

        # Communication rock
        for wp in self._goals_comm_rock:
            fgid = self._comm_rock_goal_map.get(wp, None)
            if fgid is None:
                continue
            if not self._is_true(state_bitwise, fgid):
                comm_needed += 1
                if not self._exists_have_rock(state_bitwise, wp):
                    rock_sample_needed += 1

        # Communication images and image production/calibration aggregation
        pending_images: List[Tuple[str, str]] = []  # (objective, mode)
        for (obj, mode) in self._goals_comm_image:
            fgid = self._comm_image_goal_map.get((obj, mode), None)
            if fgid is None:
                continue
            if not self._is_true(state_bitwise, fgid):
                comm_needed += 1
                # If have_image already exists for some rover, no production is needed for this pair
                if not self._exists_have_image(state_bitwise, obj, mode):
                    pending_images.append((obj, mode))

        # take_image needed per pending image
        take_image_needed = len(pending_images)

        # Aggregate calibration lower bound per mode:
        # For each mode m, count pending images P_m. Also count number of currently calibrated cameras supporting m: C_m.
        # Minimal calibrations needed >= max(0, P_m - C_m), summed over modes (each calibrated camera can produce only one image
        # before becoming uncalibrated).
        calibrate_needed = 0
        if take_image_needed > 0:
            # Count pending per mode
            p_by_mode: Dict[str, int] = {}
            for (_, m) in pending_images:
                p_by_mode[m] = p_by_mode.get(m, 0) + 1

            # Count currently calibrated cameras by mode support
            c_by_mode: Dict[str, int] = {m: 0 for m in p_by_mode.keys()}
            for (cam, rov), fgid in self._calibrated_fact.items():
                # If calibrated now and supports any of the pending modes, count it
                if self._is_true(state_bitwise, fgid):
                    for m in self._supports.get(cam, set()):
                        if m in c_by_mode:
                            c_by_mode[m] += 1
            for m, pcount in p_by_mode.items():
                ccount = c_by_mode.get(m, 0)
                need = pcount - ccount
                if need > 0:
                    calibrate_needed += need

        # Final admissible lower bound (unit costs):
        # - One communication per pending goal fact
        # - One sample per soil/rock goal not already analyzed
        # - One take_image per pending image
        # - Aggregate minimal calibrations across modes
        h_val = comm_needed + soil_sample_needed + rock_sample_needed + take_image_needed + calibrate_needed
        return max(0, h_val)

    # -------------------------------
    # Helpers
    # -------------------------------
    def _is_true(self, state_bitwise: int, global_id: int) -> bool:
        return ((state_bitwise >> global_id) & 1) == 1

    def _exists_have_soil(self, state_bitwise: int, wp: str) -> bool:
        # Any rover currently has have_soil_analysis for this waypoint?
        for rov in self._rovers:
            fgid = self._have_soil_map.get((rov, wp))
            if fgid is not None and self._is_true(state_bitwise, fgid):
                return True
        return False

    def _exists_have_rock(self, state_bitwise: int, wp: str) -> bool:
        for rov in self._rovers:
            fgid = self._have_rock_map.get((rov, wp))
            if fgid is not None and self._is_true(state_bitwise, fgid):
                return True
        return False

    def _exists_have_image(self, state_bitwise: int, obj: str, mode: str) -> bool:
        for rov in self._rovers:
            fgid = self._have_image_map.get((rov, obj, mode))
            if fgid is not None and self._is_true(state_bitwise, fgid):
                return True
        return False

    # -------------------------------
    # Model indexing
    # -------------------------------
    def _build_indices(self):
        # Index all facts by predicate and extract useful structures
        self._facts_by_pred = {}
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            self._facts_by_pred.setdefault(pred, []).append(fact)

        # Extract static sets and maps
        self._index_rovers_waypoints_objectives_modes()
        self._index_visibility_and_lander()
        self._index_equipment_and_storage()
        self._index_cameras_supports_and_calibration()
        self._index_have_and_comm_facts()
        self._index_at_facts()
        self._index_goals()

    def _index_rovers_waypoints_objectives_modes(self):
        # Deduce rovers and waypoints from 'at' and 'can_traverse' and other predicates
        for pred in ["at", "can_traverse", "store_of", "available"]:
            for f in self._facts_by_pred.get(pred, []):
                p, args = self._parse_fact_name(f.name)
                if p == "at":
                    if len(args) == 2:
                        self._rovers.add(args[0])
                        self._waypoints.add(args[1])
                elif p == "can_traverse":
                    if len(args) == 3:
                        self._rovers.add(args[0])
                        self._waypoints.add(args[1])
                        self._waypoints.add(args[2])
                elif p == "store_of" and len(args) == 2:
                    self._rovers.add(args[1])
                elif p == "available" and len(args) == 1:
                    self._rovers.add(args[0])

        for f in self._facts_by_pred.get("visible_from", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self._objectives.add(args[0])
                self._waypoints.add(args[1])

        for f in self._facts_by_pred.get("supports", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self._modes.add(args[1])

    def _index_visibility_and_lander(self):
        # visible pairs
        init = self.model.initial_state
        self._visible_pairs = set()
        for f in self._facts_by_pred.get("visible", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    self._waypoints.add(args[0])
                    self._waypoints.add(args[1])
                    self._visible_pairs.add((args[0], args[1]))
        # lander locations
        self._lander_wps = set()
        for f in self._facts_by_pred.get("at_lander", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    self._lander_wps.add(args[1])
        # waypoints visible to at least one lander location
        self._vtl_wps = set()
        for w in self._waypoints:
            for lw in self._lander_wps:
                if (w, lw) in self._visible_pairs:
                    self._vtl_wps.add(w)
                    break
        # visible_from
        self._visible_from = {}
        for f in self._facts_by_pred.get("visible_from", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    self._visible_from.setdefault(args[0], set()).add(args[1])

    def _index_equipment_and_storage(self):
        init = self.model.initial_state
        for f in self._facts_by_pred.get("equipped_for_soil_analysis", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if args:
                    self._equipped_soil.add(args[0])
        for f in self._facts_by_pred.get("equipped_for_rock_analysis", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if args:
                    self._equipped_rock.add(args[0])
        for f in self._facts_by_pred.get("equipped_for_imaging", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if args:
                    self._equipped_imaging.add(args[0])

    def _index_cameras_supports_and_calibration(self):
        init = self.model.initial_state
        # on_board camera->rover
        for f in self._facts_by_pred.get("on_board", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    cam, rov = args[0], args[1]
                    self._on_board[cam] = rov
        # supports camera->modes
        self._supports = {}
        for f in self._facts_by_pred.get("supports", []):
            if self._is_true(init, f.global_id):
                _, args = self._parse_fact_name(f.name)
                if len(args) == 2:
                    cam, mode = args[0], args[1]
                    self._supports.setdefault(cam, set()).add(mode)
        # calibrated facts: (camera, rover) -> fact id
        self._calibrated_fact = {}
        for f in self._facts_by_pred.get("calibrated", []):
            # calibrated can change dynamically; don't filter by init
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, rov = args[0], args[1]
                self._calibrated_fact[(cam, rov)] = f.global_id

    def _index_have_and_comm_facts(self):
        # have_soil_analysis, have_rock_analysis, have_image
        for f in self._facts_by_pred.get("have_soil_analysis", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                rov, wp = args[0], args[1]
                self._have_soil_map[(rov, wp)] = f.global_id
        for f in self._facts_by_pred.get("have_rock_analysis", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                rov, wp = args[0], args[1]
                self._have_rock_map[(rov, wp)] = f.global_id
        for f in self._facts_by_pred.get("have_image", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 3:
                rov, obj, mode = args[0], args[1], args[2]
                self._have_image_map[(rov, obj, mode)] = f.global_id

        # communicated_soil_data, communicated_rock_data, communicated_image_data
        for f in self._facts_by_pred.get("communicated_soil_data", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self._comm_soil_goal_map[args[0]] = f.global_id
        for f in self._facts_by_pred.get("communicated_rock_data", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self._comm_rock_goal_map[args[0]] = f.global_id
        for f in self._facts_by_pred.get("communicated_image_data", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                obj, mode = args[0], args[1]
                self._comm_image_goal_map[(obj, mode)] = f.global_id

    def _index_at_facts(self):
        # For getting rover current positions quickly (by checking which at[rover,*] fact is true)
        self._at_fact_by_rover = {r: [] for r in self._rovers}
        for f in self._facts_by_pred.get("at", []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                rov, wp = args[0], args[1]
                if rov in self._rovers:
                    self._at_fact_by_rover.setdefault(rov, []).append(f.global_id)
                    self._at_facts_detail[f.global_id] = (rov, wp)

    def _index_goals(self):
        # Parse goal bitset into structured goal sets for this domain
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goals_comm_soil = set()
        self._goals_comm_rock = set()
        self._goals_comm_image = set()
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "communicated_soil_data" and len(args) == 1:
                self._goals_comm_soil.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self._goals_comm_rock.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self._goals_comm_image.add((args[0], args[1]))

    # -------------------------------
    # Parser for grounded fact names with +/- prefix
    # -------------------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
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
