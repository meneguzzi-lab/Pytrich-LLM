from typing import Dict, List, Set, Tuple, Optional

# NOTE: The framework should already provide the Heuristic base class.
# We inherit from it and implement evaluate(state_bitwise, task_network).

class RoverGPT5Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Rover HDDL domain.

    Design rationale (admissible lower bound):
    - Each goal of the form communicated_* requires at least one communication action.
      -> We count 1 for each unachieved communication goal (soil/rock/image).
    - If the required data has not yet been acquired (no have_* fact), at least one
      acquisition action is needed:
        * Soil: sample_soil (1)
        * Rock: sample_rock (1)
        * Image: take_image (1)
    - For samples (soil/rock), if all eligible rovers' stores are full, at least one
      drop action is required before acquiring any new sample. Lower bound across all
      pending samples: max(0, needed_samples - number_of_empty_stores).
    - For images, if some cameras are already calibrated and support any of the
      required modes, these may avoid one calibration per camera. Minimal calibrations
      needed across all pending images is bounded by:
        max(0, needed_images - number_of_currently_calibrated_cameras_supporting_needed_modes)

    We intentionally do NOT add navigation costs in this sum, because they can be
    shared across multiple goals and are difficult to aggregate safely without
    overestimation. The counts above are action landmarks that cannot be avoided
    (each goal needs its own communicate; each data item needs its own acquisition).

    This yields an admissible (but informative) heuristic that scales to large
    instances and leverages domain structure.
    """

    def __init__(self, model, rover_gpt5_heuristic_5: Optional[dict] = None):
        self.model = model
        # Internal parsed structures
        self._pred_to_facts: Dict[str, List] = {}

        # Goal indexing
        self.soil_goals: List[str] = []                 # waypoint names for communicated_soil_data
        self.rock_goals: List[str] = []                 # waypoint names for communicated_rock_data
        self.image_goals: List[Tuple[str, str]] = []    # (objective, mode)

        self.comm_soil_fact_by_p: Dict[str, object] = {}
        self.comm_rock_fact_by_p: Dict[str, object] = {}
        self.comm_image_fact_by_om: Dict[Tuple[str, str], object] = {}

        # Acquisition facts
        self.have_soil_by_p: Dict[str, List[object]] = {}      # p -> [Fact have_soil_analysis[x,p]]
        self.have_rock_by_p: Dict[str, List[object]] = {}      # p -> [Fact have_rock_analysis[x,p]]
        self.have_image_by_om: Dict[Tuple[str, str], List[object]] = {}  # (o,m) -> [Fact have_image[x,o,m]]

        # Rover equipment and stores
        self.equip_soil: Set[str] = set()   # rovers equipped for soil
        self.equip_rock: Set[str] = set()   # rovers equipped for rock
        self.equip_img: Set[str] = set()    # rovers equipped for imaging

        self.store_by_rover: Dict[str, str] = {}        # rover -> store
        self.empty_fact_by_store: Dict[str, object] = {}  # store -> Fact empty[store]
        self.full_fact_by_store: Dict[str, object] = {}   # store -> Fact full[store]

        # Camera info
        self.on_board: Dict[str, str] = {}             # camera -> rover
        self.supports: Dict[str, Set[str]] = {}        # camera -> set(modes)
        self.calib_target: Dict[str, str] = {}         # camera -> objective
        self.calibrated_fact: Dict[Tuple[str, str], object] = {}  # (camera, rover) -> Fact calibrated[c,x]

        # Visibility and lander info (may be handy for potential extensions)
        self.visible: Dict[str, Set[str]] = {}         # p1 -> set(p2) if visible[p1,p2]
        self.visible_from: Dict[str, Set[str]] = {}    # objective -> set(waypoints) if visible_from[o,p]
        self.lander_positions: Set[str] = set()        # { waypoint } for at_lander[l, waypoint]
        self.wp_visible_to_any_lander: Set[str] = set()

        # AT facts for quick rover location lookup if needed later (not essential for the heuristic bound)
        self.at_by_rover: Dict[str, List[Tuple[str, object]]] = {}  # rover -> list of (waypoint, Fact at[rover,wp])

        # Parse all facts once
        self._index_facts()
        # Parse goals once
        self._index_goals()
        # Prepare lander visibility cache
        self._compute_lander_visibility_cache()

    # ------------------------
    # Parsing and preprocessing
    # ------------------------

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _add_visible(self, p1: str, p2: str):
        if p1 not in self.visible:
            self.visible[p1] = set()
        self.visible[p1].add(p2)

    def _index_facts(self):
        # Iterate over all grounded facts in the model
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'communicated_soil_data' and len(args) == 1:
                p = args[0]
                self.comm_soil_fact_by_p[p] = fact
            elif pred == 'communicated_rock_data' and len(args) == 1:
                p = args[0]
                self.comm_rock_fact_by_p[p] = fact
            elif pred == 'communicated_image_data' and len(args) == 2:
                o, m = args[0], args[1]
                self.comm_image_fact_by_om[(o, m)] = fact

            elif pred == 'have_soil_analysis' and len(args) == 2:
                x, p = args[0], args[1]
                self.have_soil_by_p.setdefault(p, []).append(fact)
            elif pred == 'have_rock_analysis' and len(args) == 2:
                x, p = args[0], args[1]
                self.have_rock_by_p.setdefault(p, []).append(fact)
            elif pred == 'have_image' and len(args) == 3:
                x, o, m = args[0], args[1], args[2]
                self.have_image_by_om.setdefault((o, m), []).append(fact)

            elif pred == 'equipped_for_soil_analysis' and len(args) == 1:
                self.equip_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis' and len(args) == 1:
                self.equip_rock.add(args[0])
            elif pred == 'equipped_for_imaging' and len(args) == 1:
                self.equip_img.add(args[0])

            elif pred == 'store_of' and len(args) == 2:
                s, x = args[0], args[1]
                self.store_by_rover[x] = s
            elif pred == 'empty' and len(args) == 1:
                s = args[0]
                self.empty_fact_by_store[s] = fact
            elif pred == 'full' and len(args) == 1:
                s = args[0]
                self.full_fact_by_store[s] = fact

            elif pred == 'on_board' and len(args) == 2:
                c, x = args[0], args[1]
                self.on_board[c] = x
            elif pred == 'supports' and len(args) == 2:
                c, m = args[0], args[1]
                self.supports.setdefault(c, set()).add(m)
            elif pred == 'calibration_target' and len(args) == 2:
                c, o = args[0], args[1]
                self.calib_target[c] = o
            elif pred == 'calibrated' and len(args) == 2:
                c, x = args[0], args[1]
                self.calibrated_fact[(c, x)] = fact

            elif pred == 'visible' and len(args) == 2:
                p1, p2 = args[0], args[1]
                self._add_visible(p1, p2)
            elif pred == 'visible_from' and len(args) == 2:
                o, p = args[0], args[1]
                if o not in self.visible_from:
                    self.visible_from[o] = set()
                self.visible_from[o].add(p)
            elif pred == 'at_lander' and len(args) == 2:
                l, p = args[0], args[1]
                self.lander_positions.add(p)

            elif pred == 'at' and len(args) == 2:
                x, p = args[0], args[1]
                self.at_by_rover.setdefault(x, []).append((p, fact))

    def _index_goals(self):
        # Extract goal facts explicitly and parse them into categories
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if not pred:
                continue
            if pred == 'communicated_soil_data' and len(args) == 1:
                self.soil_goals.append(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.rock_goals.append(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                self.image_goals.append((args[0], args[1]))

    def _compute_lander_visibility_cache(self):
        # Build the set of waypoints from which at least one lander's location is visible
        wp_set = set()
        if self.lander_positions:
            for wp1, vis_to in self.visible.items():
                for lp in self.lander_positions:
                    if lp in vis_to:
                        wp_set.add(wp1)
                        break
        self.wp_visible_to_any_lander = wp_set

    # ------------------------
    # Utility: fast truth check
    # ------------------------

    @staticmethod
    def _is_true(state_bitwise: int, fact_obj) -> bool:
        if fact_obj is None:
            return False
        gid = fact_obj.global_id
        return (state_bitwise >> gid) & 1 == 1

    # ------------------------
    # Heuristic computation
    # ------------------------

    def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
        # Count remaining (unachieved) goals per category and add minimal required actions
        h = 0

        # 1) SOIL goals: communication + (if no have_soil) sample + store-empty lower bound
        needed_soil_comm = 0
        needed_soil_samples = 0
        for p in self.soil_goals:
            comm_fact = self.comm_soil_fact_by_p.get(p)
            if not self._is_true(state_bitwise, comm_fact):
                needed_soil_comm += 1  # one communication per soil goal

                # Check if any rover already has soil analysis for this waypoint p
                have_list = self.have_soil_by_p.get(p, [])
                already_have = any(self._is_true(state_bitwise, f) for f in have_list)
                if not already_have:
                    needed_soil_samples += 1  # one sample action needed for this soil goal

        # Lower bound on required drops for soil (to empty a store before sampling)
        if needed_soil_samples > 0:
            empty_stores_soil = 0
            for x in self.equip_soil:
                s = self.store_by_rover.get(x)
                if s is not None:
                    fempty = self.empty_fact_by_store.get(s)
                    if self._is_true(state_bitwise, fempty):
                        empty_stores_soil += 1
            # Each empty store can immediately support one new sample before it becomes full.
            needed_soil_drops = max(0, needed_soil_samples - empty_stores_soil)
        else:
            needed_soil_drops = 0

        h += needed_soil_comm + needed_soil_samples + needed_soil_drops

        # 2) ROCK goals: communication + (if no have_rock) sample + store-empty lower bound
        needed_rock_comm = 0
        needed_rock_samples = 0
        for p in self.rock_goals:
            comm_fact = self.comm_rock_fact_by_p.get(p)
            if not self._is_true(state_bitwise, comm_fact):
                needed_rock_comm += 1

                have_list = self.have_rock_by_p.get(p, [])
                already_have = any(self._is_true(state_bitwise, f) for f in have_list)
                if not already_have:
                    needed_rock_samples += 1

        if needed_rock_samples > 0:
            empty_stores_rock = 0
            for x in self.equip_rock:
                s = self.store_by_rover.get(x)
                if s is not None:
                    fempty = self.empty_fact_by_store.get(s)
                    if self._is_true(state_bitwise, fempty):
                        empty_stores_rock += 1
            needed_rock_drops = max(0, needed_rock_samples - empty_stores_rock)
        else:
            needed_rock_drops = 0

        h += needed_rock_comm + needed_rock_samples + needed_rock_drops

        # 3) IMAGE goals: communication + (if no have_image) take_image + calibration lower bound
        needed_img_comm = 0
        needed_images = 0
        needed_modes: Set[str] = set()
        for (o, m) in self.image_goals:
            comm_fact = self.comm_image_fact_by_om.get((o, m))
            if not self._is_true(state_bitwise, comm_fact):
                needed_img_comm += 1
                have_list = self.have_image_by_om.get((o, m), [])
                already_have = any(self._is_true(state_bitwise, f) for f in have_list)
                if not already_have:
                    needed_images += 1
                    needed_modes.add(m)

        # Calibrated cameras that can support at least one of the needed modes
        calibrated_supply = 0
        if needed_images > 0 and needed_modes:
            for (c, x), fcal in self.calibrated_fact.items():
                if self._is_true(state_bitwise, fcal):
                    modes_supported = self.supports.get(c, set())
                    if modes_supported & needed_modes:
                        calibrated_supply += 1
        needed_calibrations = max(0, needed_images - calibrated_supply)

        h += needed_img_comm + needed_images + needed_calibrations

        return int(h)
