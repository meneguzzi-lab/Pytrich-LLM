from typing import Dict, List, Tuple, Set, Optional

try:
    # Expected base class in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub to allow standalone parsing/tests
    class Heuristic:
        def __init__(self, model, config=None):
            self.model = model
        def estimate(self, state_bitwise: int, task_network: List):
            return 0


class RoverGPT5Heuristic9(Heuristic):
    """
    Rover domain-dependent heuristic (admissible lower bound) for Pytrich Planner.

    Design rationale and domain insights:
    - Goals in Rover are to communicate soil/rock/image data. Each goal requires at least one
      distinct communication action (cannot communicate multiple goals with one action), thus
      counting 1 per unsatisfied goal is a safe admissible base.
    - To communicate, the rover must first possess the corresponding data:
        * Soil: have_soil_analysis(x, wp) via sample_soil
        * Rock: have_rock_analysis(x, wp) via sample_rock
        * Image: have_image(x, o, m) via calibrate + take_image
      These actions are unique per goal instance (distinct waypoint or distinct objective/mode),
      so we can safely add their minimal lower bounds without risking overestimation from shared actions.
    - Navigation and store management (empty/full) can sometimes be shared across goals. To keep
      the heuristic admissible, we intentionally ignore navigation, empty_store, and visibility
      requirements in the cost (treat them as zero-cost in the lower bound). This preserves
      admissibility while still providing meaningful guidance.

    Heuristic formula per goal (lower bound):
    - Soil goal (communicated_soil_data[wp]):
        if already communicated: +0
        elif have_soil_analysis[*, wp] true in state: +1 (communicate)
        else: +2 (sample_soil + communicate)

    - Rock goal (communicated_rock_data[wp]):
        if already communicated: +0
        elif have_rock_analysis[*, wp] true in state: +1 (communicate)
        else: +2 (sample_rock + communicate)

    - Image goal (communicated_image_data[o, m]):
        if already communicated: +0
        elif have_image[*, o, m] true in state: +1 (communicate)
        elif exists a capable (camera c on rover x) with calibrated[c, x] true in current state:
             +2 (take_image + communicate)
        else: +3 (calibrate + take_image + communicate)

    Notes:
    - We do not add any navigation lower bound, nor empty_store/drop costs; this is necessary to
      avoid overcounting across multiple goals and to maintain admissibility.
    - The check for calibrated[c, x] to reduce the image cost to 2 is conservative: even if the
      rover is not currently at a viewpoint visible for the objective, ignoring navigation keeps it admissible.
    - All state membership tests use bitwise checks for efficiency on large instances.
    """

    def __init__(self, model, rover_gpt5_heuristic_9: Optional[dict] = None):
        super().__init__(model, rover_gpt5_heuristic_9)
        self.model = model

        # Goal partitions
        self.soil_goals: List[Tuple[str, int]] = []        # (wp, communicated_fact_id)
        self.rock_goals: List[Tuple[str, int]] = []        # (wp, communicated_fact_id)
        self.image_goals: List[Tuple[Tuple[str, str], int]] = []  # ((o, m), communicated_fact_id)

        # Supporting indices (static wrt model)
        self.have_soil_by_wp: Dict[str, List[int]] = {}     # wp -> [fact_id have_soil_analysis[x,wp]]
        self.have_rock_by_wp: Dict[str, List[int]] = {}     # wp -> [fact_id have_rock_analysis[x,wp]]
        self.have_image_by_om: Dict[Tuple[str, str], List[int]] = {}  # (o,m) -> [fact_id have_image[x,o,m]]

        # For calibrated-based image LB optimization
        self.supports_by_camera: Dict[str, Set[str]] = {}   # c -> {m}
        self.calibration_targets: Set[Tuple[str, str]] = set()  # (c,o)
        self.on_board_cam_rover: Set[Tuple[str, str]] = set()   # (c,x)
        self.calibrated_id_by_cx: Dict[Tuple[str, str], int] = {}  # (c,x) -> fact_id calibrated[c,x]
        self.calibrated_ids_by_om: Dict[Tuple[str, str], List[int]] = {}  # (o,m) -> [fact_id calibrated[c,x] for capable pairs]

        # Precompute indices once
        self._precompute_indices()

    # -------------------- Heuristic API --------------------
    def estimate(self, state_bitwise: int, task_network: List) -> int:
        h = 0
        is_true = self._is_true

        # Soil goals
        for (wp, comm_fid) in self.soil_goals:
            if is_true(state_bitwise, comm_fid):
                continue  # already satisfied
            # If any have_soil_analysis[* , wp] is true -> need 1 (communicate)
            found_have = False
            for fid in self.have_soil_by_wp.get(wp, []):
                if is_true(state_bitwise, fid):
                    found_have = True
                    break
            if found_have:
                h += 1
            else:
                h += 2  # sample_soil + communicate

        # Rock goals
        for (wp, comm_fid) in self.rock_goals:
            if is_true(state_bitwise, comm_fid):
                continue
            found_have = False
            for fid in self.have_rock_by_wp.get(wp, []):
                if is_true(state_bitwise, fid):
                    found_have = True
                    break
            if found_have:
                h += 1
            else:
                h += 2  # sample_rock + communicate

        # Image goals
        for ((o, m), comm_fid) in self.image_goals:
            if is_true(state_bitwise, comm_fid):
                continue
            # If have_image[* , o, m] present -> +1
            have_list = self.have_image_by_om.get((o, m), [])
            have_found = False
            for fid in have_list:
                if is_true(state_bitwise, fid):
                    have_found = True
                    break
            if have_found:
                h += 1
                continue

            # Else, if any capable (c,x) currently calibrated -> +2 (take_image + communicate)
            cal_list = self.calibrated_ids_by_om.get((o, m), [])
            cal_found = False
            for fid in cal_list:
                if is_true(state_bitwise, fid):
                    cal_found = True
                    break
            if cal_found:
                h += 2
            else:
                h += 3  # calibrate + take_image + communicate

        return h

    # -------------------- Precomputation --------------------
    def _precompute_indices(self):
        # 1) Extract goals (communicated_*) by iterating bits in model.goals
        for bit in self._iterate_bits(self.model.goals):
            fact = self.model.get_component(bit)
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'communicated_soil_data' and len(args) == 1:
                self.soil_goals.append((args[0], fact.global_id))
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.rock_goals.append((args[0], fact.global_id))
            elif pred == 'communicated_image_data' and len(args) == 2:
                o, m = args[0], args[1]
                self.image_goals.append(((o, m), fact.global_id))
            # Ignore any other goal predicates (if present)

        # 2) Single pass over all facts to build main indices
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            gid = fact.global_id

            if pred == 'have_soil_analysis' and len(args) == 2:
                r, wp = args[0], args[1]
                self.have_soil_by_wp.setdefault(wp, []).append(gid)

            elif pred == 'have_rock_analysis' and len(args) == 2:
                r, wp = args[0], args[1]
                self.have_rock_by_wp.setdefault(wp, []).append(gid)

            elif pred == 'have_image' and len(args) == 3:
                r, o, m = args[0], args[1], args[2]
                self.have_image_by_om.setdefault((o, m), []).append(gid)

            elif pred == 'supports' and len(args) == 2:
                c, m = args[0], args[1]
                self.supports_by_camera.setdefault(c, set()).add(m)

            elif pred == 'calibration_target' and len(args) == 2:
                c, o = args[0], args[1]
                self.calibration_targets.add((c, o))

            elif pred == 'on_board' and len(args) == 2:
                c, x = args[0], args[1]
                self.on_board_cam_rover.add((c, x))

            elif pred == 'calibrated' and len(args) == 2:
                c, x = args[0], args[1]
                self.calibrated_id_by_cx[(c, x)] = gid

        # 3) Precompute calibrated fact ids per (objective, mode) capability
        # For each image goal key (o,m), compute which calibrated[c,x] facts would be relevant
        # (i.e., camera c supports m, calibration_target(c,o), on_board(c,x))
        relevant_oms: Set[Tuple[str, str]] = set(k for (k, _) in self.image_goals)
        # Also include any other possible (o,m) pairs that might appear, to be robust
        relevant_oms.update(self.have_image_by_om.keys())

        for (o, m) in relevant_oms:
            cal_ids: List[int] = []
            for (c, o2) in self.calibration_targets:
                if o2 != o:
                    continue
                if m not in self.supports_by_camera.get(c, set()):
                    continue
                # For every rover x with this camera onboard
                for (c2, x) in self.on_board_cam_rover:
                    if c2 != c:
                        continue
                    fid = self.calibrated_id_by_cx.get((c, x))
                    if fid is not None:
                        cal_ids.append(fid)
            if cal_ids:
                self.calibrated_ids_by_om[(o, m)] = cal_ids

    # -------------------- Utilities --------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name in format '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Returns (predicate, [args...]) or (None, []).
        """
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        lb = s.find('[')
        if lb > 0 and s.endswith(']'):
            pred = s[:lb]
            args_str = s[lb + 1 : -1]
            args = args_str.split(',') if args_str else []
            return pred, args
        return None, []

    @staticmethod
    def _iterate_bits(bitset: int):
        """Yield indexes of set bits in bitset."""
        idx = 0
        b = bitset
        while b:
            if b & 1:
                yield idx
            b >>= 1
            idx += 1

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return (state_bitwise >> fact_id) & 1 == 1


# Alias for configuration systems that expect a callable/class reference by name
def get_heuristic_class():
    return RoverGPT5Heuristic9
