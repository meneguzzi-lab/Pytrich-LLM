from typing import Dict, List, Tuple, Set, Optional

# Pytrich provides this base class
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT5Heuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Rover HDDL domain.

    Design rationale (admissible, fast, domain-specific):
    - In Rover, each goal of the form communicated_* must ultimately execute a unique
      communication action. Therefore, for every unachieved communication goal we add +1.
    - To be able to communicate, data must exist: soil/rock samples and images.
      For soil/rock goals, if no rover currently has have_*_analysis for the target waypoint,
      at least one sample_* action is required (add +1). For image goals, if no rover has the
      have_image for (objective,mode), at least one take_image action is required (add +1).
    - We deliberately do NOT count navigation or calibration in the sum to preserve admissibility,
      since those actions can often be shared across multiple goals; summing them could overestimate.
      Counting only the per-goal, non-shareable actions (sample/take and communicate) yields a safe
      lower bound that still guides search effectively across scales.

    Notes:
    - This heuristic uses bitwise checks for state membership (fast) and precomputes goal/fact
      indices once (at init). It avoids expensive conversion of the entire state each call.
    - It also parses the hierarchical goals from model.goals, but the estimate is purely state- and
      goal-based (task-network independent) to remain robust and efficient.
    - If goals are already achieved, returns 0 even if task network is not empty. This keeps the
      heuristic admissible; the planner will still refine the remaining tasks with g-cost guiding.
    """

    def __init__(self, model, rover_gpt5_heuristic_19: bool = True):
        super().__init__(model)
        self.model = model

        # Fact name -> Fact (for quick id lookup)
        self._fact_by_name: Dict[str, object] = {}

        # Parsed indices for quick access
        self.rovers: Set[str] = set()
        self.stores_by_rover: Dict[str, List[str]] = {}

        # Soil/Rock/Image possession facts indexed by goal arguments
        self.have_soil_by_wp: Dict[str, List[Tuple[int, str]]] = {}      # wp -> [(fact_id, rover)]
        self.have_rock_by_wp: Dict[str, List[Tuple[int, str]]] = {}      # wp -> [(fact_id, rover)]
        self.have_image_by_om: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}  # (obj,mode) -> [(fact_id, rover)]

        # Communication fact ids for quick bitwise checking
        self.comm_soil_id_by_wp: Dict[str, int] = {}
        self.comm_rock_id_by_wp: Dict[str, int] = {}
        self.comm_image_id_by_om: Dict[Tuple[str, str], int] = {}

        # Empty store ids (store -> fact_id)
        self.empty_store_id_by_store: Dict[str, int] = {}

        # Equipped rovers (static capabilities)
        self.equipped_soil: Set[str] = set()
        self.equipped_rock: Set[str] = set()
        self.equipped_imaging: Set[str] = set()

        # AT (position) indices to speed-up optional checks (kept simple here; not used in sum)
        self.at_id_by_r_w: Dict[Tuple[str, str], int] = {}
        self.at_ids_by_rover: Dict[str, List[Tuple[str, int]]] = {}  # rover -> list of (w, fact_id)

        # Goals parsed from model.goals
        self.goal_soil_wps: List[str] = []
        self.goal_rock_wps: List[str] = []
        self.goal_image_oms: List[Tuple[str, str]] = []

        self._preprocess_model()

    # Heuristic interface ----------------------------------------------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        return self.evaluate(state_bitwise, task_network)

    def value(self, state_bitwise: int, task_network=None) -> int:
        return self.evaluate(state_bitwise, task_network)

    # Core heuristic ---------------------------------------------------------
    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        h = 0

        # SOIL goals: for each unachieved communication, add 1 for communication
        # and +1 for sampling if data not yet available.
        for wp in self.goal_soil_wps:
            cs_id = self.comm_soil_id_by_wp.get(wp)
            if cs_id is None:
                continue  # safety
            if self._is_true(state_bitwise, cs_id):
                continue  # already communicated

            # communication (unique action per goal)
            h += 1

            # if no rover currently has the sample for this waypoint, must sample at least once
            if not self._any_have_soil(state_bitwise, wp):
                h += 1

        # ROCK goals: same structure
        for wp in self.goal_rock_wps:
            cr_id = self.comm_rock_id_by_wp.get(wp)
            if cr_id is None:
                continue
            if self._is_true(state_bitwise, cr_id):
                continue

            h += 1  # unique communication action

            if not self._any_have_rock(state_bitwise, wp):
                h += 1  # at least one sample_rock required

        # IMAGE goals: for each unachieved communication, add 1 for communication
        # and +1 for taking the image if have_image is not yet present (calibration/navigation ignored for admissibility).
        for (obj, mode) in self.goal_image_oms:
            ci_id = self.comm_image_id_by_om.get((obj, mode))
            if ci_id is None:
                continue
            if self._is_true(state_bitwise, ci_id):
                continue

            h += 1  # communication action per (objective, mode)

            if not self._any_have_image(state_bitwise, obj, mode):
                h += 1  # take_image action at least once

        return h

    # Helpers ----------------------------------------------------------------
    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return (state_bitwise >> fact_id) & 1 == 1

    def _any_have_soil(self, state_bitwise: int, wp: str) -> bool:
        lst = self.have_soil_by_wp.get(wp)
        if not lst:
            return False
        for fid, _ in lst:
            if self._is_true(state_bitwise, fid):
                return True
        return False

    def _any_have_rock(self, state_bitwise: int, wp: str) -> bool:
        lst = self.have_rock_by_wp.get(wp)
        if not lst:
            return False
        for fid, _ in lst:
            if self._is_true(state_bitwise, fid):
                return True
        return False

    def _any_have_image(self, state_bitwise: int, obj: str, mode: str) -> bool:
        lst = self.have_image_by_om.get((obj, mode))
        if not lst:
            return False
        for fid, _ in lst:
            if self._is_true(state_bitwise, fid):
                return True
        return False

    # Preprocessing ----------------------------------------------------------
    def _preprocess_model(self):
        # 1) Index all facts by name and parse structures we need
        for f in self.model.facts:
            self._fact_by_name[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            # Track rovers and waypoint positions
            if pred == 'at' and len(args) == 2:
                r, w = args
                self.rovers.add(r)
                self.at_id_by_r_w[(r, w)] = f.global_id
                self.at_ids_by_rover.setdefault(r, []).append((w, f.global_id))

            # store_of
            elif pred == 'store_of' and len(args) == 2:
                s, r = args
                self.rovers.add(r)
                self.stores_by_rover.setdefault(r, []).append(s)

            # empty store facts
            elif pred == 'empty' and len(args) == 1:
                s = args[0]
                self.empty_store_id_by_store[s] = f.global_id

            # capabilities
            elif pred == 'equipped_for_soil_analysis' and len(args) == 1:
                self.equipped_soil.add(args[0])
            elif pred == 'equipped_for_rock_analysis' and len(args) == 1:
                self.equipped_rock.add(args[0])
            elif pred == 'equipped_for_imaging' and len(args) == 1:
                self.equipped_imaging.add(args[0])

            # possession of data
            elif pred == 'have_soil_analysis' and len(args) == 2:
                r, wp = args
                self.have_soil_by_wp.setdefault(wp, []).append((f.global_id, r))
            elif pred == 'have_rock_analysis' and len(args) == 2:
                r, wp = args
                self.have_rock_by_wp.setdefault(wp, []).append((f.global_id, r))
            elif pred == 'have_image' and len(args) == 3:
                r, obj, mode = args
                self.have_image_by_om.setdefault((obj, mode), []).append((f.global_id, r))

            # communications
            elif pred == 'communicated_soil_data' and len(args) == 1:
                wp = args[0]
                self.comm_soil_id_by_wp[wp] = f.global_id
            elif pred == 'communicated_rock_data' and len(args) == 1:
                wp = args[0]
                self.comm_rock_id_by_wp[wp] = f.global_id
            elif pred == 'communicated_image_data' and len(args) == 2:
                obj, mode = args
                self.comm_image_id_by_om[(obj, mode)] = f.global_id

        # 2) Parse the goal facts from model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            if pred == 'communicated_soil_data' and len(args) == 1:
                self.goal_soil_wps.append(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.goal_rock_wps.append(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                self.goal_image_oms.append((args[0], args[1]))

    @staticmethod
    def _parse_fact_name(name: str):
        """
        Parse grounded fact name of the form +predicate[arg1,arg2] or -predicate[arg1,arg2].
        Returns (predicate, [args]) or (None, []).
        """
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name = name[1:]
            bracket_idx = name.find('[')
            if bracket_idx > 0 and name.endswith(']'):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []
