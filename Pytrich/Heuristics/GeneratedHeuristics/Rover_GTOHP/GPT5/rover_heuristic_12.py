from typing import Dict, List, Tuple, Set
from collections import defaultdict

try:
    from Heuristics.Heuristic import Heuristic
except Exception:
    # Fallback import path if framework layout differs
    from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT5Heuristic12(Heuristic):
    """
    RoverGPT5Heuristic12: Domain-dependent heuristic for the Rover HDDL domain.

    Design rationale (domain insights):
    - Goals are of three kinds: communicated_soil_data(p), communicated_rock_data(p), communicated_image_data(o,m).
    - Achieving these requires, in the most optimistic (lower-bound) sense and ignoring navigation:
        Soil:
            - If already communicated -> 0 actions
            - Else if have_soil_analysis(x,p) for some rover x -> at least 1 (communicate)
            - Else -> at least 2 (sample_soil + communicate)
        Rock:
            - If already communicated -> 0 actions
            - Else if have_rock_analysis(x,p) -> at least 1 (communicate)
            - Else -> at least 2 (sample_rock + communicate)
        Image (o,m):
            - If already communicated -> 0 actions
            - Else if have_image(x,o,m) for some rover x -> at least 1 (communicate)
            - Else if some relevant (c,x) pair is currently calibrated (calibrated(c,x)) where
                  on_board(c,x), supports(c,m), calibration_target(c,o), equipped_for_imaging(x)
                -> at least 2 (take_image + communicate)
            - Else -> at least 3 (calibrate + take_image + communicate)
    - Store capacity lower bound:
        Sampling (soil/rock) requires an empty store, and each sample turns a store full. Without counting navigation,
        a lower bound on the number of drop actions required is:
            max(0, outstanding_samples - number_of_empty_stores_currently_true)
        This is admissible because each empty store can enable at most one immediate sample without a drop.

    Implementation notes:
    - Uses bitwise membership tests over fact global_ids for speed.
    - Precomputes indices for quick lookup of communicated/have/calibrated/image facts and relevant camera-rover pairs.
    - Parses goals once from model.goals (bitwise) using model.state_explicit_repr.

    This heuristic respects admissibility by:
      - Ignoring navigation and positional/visibility constraints (which only add cost).
      - Counting only minimal action patterns per remaining goal.
      - Adding a safe lower-bound for necessary drop actions due to store capacity.
    """

    def __init__(self, model, rover_gpt5_heuristic_12: bool = True):
        super().__init__(model)
        self.model = model

        # Indices for fast lookup
        self.comm_soil_id_by_wp: Dict[str, int] = {}
        self.have_soil_ids_by_wp: Dict[str, List[int]] = defaultdict(list)

        self.comm_rock_id_by_wp: Dict[str, int] = {}
        self.have_rock_ids_by_wp: Dict[str, List[int]] = defaultdict(list)

        self.comm_image_id_by_om: Dict[Tuple[str, str], int] = {}
        self.have_image_ids_by_om: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        # Static facts to derive relevant calibration pairs
        self.on_board: Set[Tuple[str, str]] = set()              # (camera, rover)
        self.supports: Set[Tuple[str, str]] = set()               # (camera, mode)
        self.calibration_target: Set[Tuple[str, str]] = set()     # (camera, objective)
        self.equipped_imaging: Set[str] = set()                   # {rover}

        self.calibrated_id_by_cx: Dict[Tuple[str, str], int] = {}  # (camera, rover) -> fact id
        self.cal_ids_by_om: Dict[Tuple[str, str], List[int]] = defaultdict(list)  # (objective, mode) -> list of calibrated ids

        self.empty_store_ids: List[int] = []
        self.store_of: Dict[str, str] = {}  # store -> rover (not strictly needed, but may be useful)

        # Parse all facts once
        self._index_all_facts()

        # After static indices exist, precompute relevant calibrated fact ids per (objective, mode)
        self._precompute_calibration_links()

        # Parse goal facts once (bitwise goal set -> explicit names -> parsed)
        self.goal_items = self._parse_goal_items()

        # Pretty name
        self.name = "RoverGPT5Heuristic12"

    # -----------------------------
    # Indexing and parsing helpers
    # -----------------------------

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_all_facts(self):
        # Iterate over all grounded facts to build id indices
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            # Communicated soil/rock and have_* mappings
            if pred == "communicated_soil_data" and len(args) == 1:
                wp = args[0]
                self.comm_soil_id_by_wp[wp] = fact.global_id
            elif pred == "have_soil_analysis" and len(args) == 2:
                rover, wp = args
                self.have_soil_ids_by_wp[wp].append(fact.global_id)

            elif pred == "communicated_rock_data" and len(args) == 1:
                wp = args[0]
                self.comm_rock_id_by_wp[wp] = fact.global_id
            elif pred == "have_rock_analysis" and len(args) == 2:
                rover, wp = args
                self.have_rock_ids_by_wp[wp].append(fact.global_id)

            # Image related facts
            elif pred == "communicated_image_data" and len(args) == 2:
                o, m = args
                self.comm_image_id_by_om[(o, m)] = fact.global_id
            elif pred == "have_image" and len(args) == 3:
                rover, o, m = args
                self.have_image_ids_by_om[(o, m)].append(fact.global_id)

            # Static facts (on_board, supports, calibration_target, equipped_for_imaging)
            elif pred == "on_board" and len(args) == 2:
                c, x = args
                self.on_board.add((c, x))
            elif pred == "supports" and len(args) == 2:
                c, m = args
                self.supports.add((c, m))
            elif pred == "calibration_target" and len(args) == 2:
                c, o = args
                self.calibration_target.add((c, o))
            elif pred == "equipped_for_imaging" and len(args) == 1:
                x = args[0]
                self.equipped_imaging.add(x)

            # Calibrated mapping
            elif pred == "calibrated" and len(args) == 2:
                c, x = args
                self.calibrated_id_by_cx[(c, x)] = fact.global_id

            # Store-related for drop lower bound
            elif pred == "empty" and len(args) == 1:
                s = args[0]
                self.empty_store_ids.append(fact.global_id)
            elif pred == "store_of" and len(args) == 2:
                s, x = args
                self.store_of[s] = x

    def _precompute_calibration_links(self):
        """For each (objective, mode), find all calibrated(c,x) fact IDs that are relevant pairs:
            on_board(c,x) and supports(c,m) and calibration_target(c,o) and equipped_for_imaging(x).
        """
        # Build all relevant (o, m) combos that appear in goals or facts
        # We will fill self.cal_ids_by_om lazily when goals parsed; however we can prebuild over all objective-mode pairs
        # by scanning supports and calibration_target
        # Combine cameras and rovers into candidate pairs
        for (c, x) in self.on_board:
            if x not in self.equipped_imaging:
                continue
            for (c2, m) in list(self.supports):
                if c2 != c:
                    continue
                # For each objective this camera can calibrate to
                for (c3, o) in list(self.calibration_target):
                    if c3 != c:
                        continue
                    # Valid (c,x,o,m)
                    cal_id = self.calibrated_id_by_cx.get((c, x), None)
                    if cal_id is not None:
                        self.cal_ids_by_om[(o, m)].append(cal_id)

    def _parse_goal_items(self):
        """Parse the model's goal facts into a structured list for quick evaluation.
           Each item is a dict containing necessary ids/lists for fast lower-bound computation.
        """
        goal_names = self.model.state_explicit_repr(self.model.goals)
        items = []
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if not pred:
                continue
            if pred == "communicated_soil_data" and len(args) == 1:
                wp = args[0]
                comm_id = self.comm_soil_id_by_wp.get(wp, None)
                have_ids = self.have_soil_ids_by_wp.get(wp, [])
                if comm_id is not None:
                    items.append({
                        "kind": "soil",
                        "wp": wp,
                        "comm_id": comm_id,
                        "have_ids": have_ids,
                    })
            elif pred == "communicated_rock_data" and len(args) == 1:
                wp = args[0]
                comm_id = self.comm_rock_id_by_wp.get(wp, None)
                have_ids = self.have_rock_ids_by_wp.get(wp, [])
                if comm_id is not None:
                    items.append({
                        "kind": "rock",
                        "wp": wp,
                        "comm_id": comm_id,
                        "have_ids": have_ids,
                    })
            elif pred == "communicated_image_data" and len(args) == 2:
                o, m = args
                comm_id = self.comm_image_id_by_om.get((o, m), None)
                have_img_ids = self.have_image_ids_by_om.get((o, m), [])
                cal_ids = self.cal_ids_by_om.get((o, m), [])
                if comm_id is not None:
                    items.append({
                        "kind": "img",
                        "o": o,
                        "m": m,
                        "comm_id": comm_id,
                        "have_img_ids": have_img_ids,
                        "cal_ids": cal_ids,
                    })
        return items

    # -----------------------------
    # Utility bitwise checks
    # -----------------------------

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    @staticmethod
    def _any_true(state_bitwise: int, ids: List[int]) -> bool:
        for fid in ids:
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    # -----------------------------
    # Heuristic evaluation
    # -----------------------------

    def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
        """Compute admissible lower bound on remaining effort.

        Parameters:
        - state_bitwise: current state as bitwise integer
        - task_network: current (possibly partial) HTN task network; unused here to keep admissibility simple and fast

        Returns: non-negative integer lower bound.
        """
        if task_network is None:
            task_network = []

        total_lb = 0
        outstanding_samples = 0  # for drop() lower-bound across soil+rock goals

        for item in self.goal_items:
            kind = item["kind"]
            comm_id = item["comm_id"]
            if self._is_true(state_bitwise, comm_id):
                continue  # already achieved

            if kind == "soil":
                # Need communicated_soil_data[wp]
                if self._any_true(state_bitwise, item.get("have_ids", [])):
                    total_lb += 1  # communicate only
                else:
                    total_lb += 2  # sample + communicate
                    outstanding_samples += 1

            elif kind == "rock":
                if self._any_true(state_bitwise, item.get("have_ids", [])):
                    total_lb += 1
                else:
                    total_lb += 2
                    outstanding_samples += 1

            elif kind == "img":
                # communicated_image_data[o,m]
                if self._any_true(state_bitwise, item.get("have_img_ids", [])):
                    total_lb += 1  # just communicate
                else:
                    # if already calibrated for some (c,x) pair suitable for (o,m): need take_image + communicate
                    if self._any_true(state_bitwise, item.get("cal_ids", [])):
                        total_lb += 2
                    else:
                        total_lb += 3  # calibrate + take_image + communicate

        # Lower bound for drop actions due to store capacity (shared by soil/rock sampling)
        if outstanding_samples > 0 and self.empty_store_ids:
            empty_now = 0
            # Count how many empty stores are currently true
            for fid in self.empty_store_ids:
                if ((state_bitwise >> fid) & 1) == 1:
                    empty_now += 1
            drops_needed = max(0, outstanding_samples - empty_now)
            total_lb += drops_needed

        return total_lb

    # Optional: expose a name / string representation
    def __str__(self):
        return f"{self.name}"
