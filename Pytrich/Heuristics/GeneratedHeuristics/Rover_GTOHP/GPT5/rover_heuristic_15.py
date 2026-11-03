from typing import Dict, List, Tuple, Set, Any

try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub if the environment provides Heuristic elsewhere.
    class Heuristic:
        def __init__(self, model):
            self.model = model
        def __call__(self, state, task_network):
            return 0


class RoverGPT5Heuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Rover HDDL domain.

    Design rationale and domain insights:
    - Rover pipelines split into three independent data products:
      1) Soil data at a waypoint p -> communicated_soil_data[p]
         Minimal primitive actions (ignoring navigation/visibility, which only reduces the lower bound):
           - If already communicated: 0
           - If have_soil_analysis by any rover: 1 (communicate)
           - Else if any equipped rover has an empty store: 2 (sample + communicate)
           - Else: 3 (drop + sample + communicate)

      2) Rock data at a waypoint p -> communicated_rock_data[p]
         Minimal primitive actions analogous to soil.

      3) Image data for objective o in mode m -> communicated_image_data[o,m]
         Minimal primitive actions (ignoring navigation/visibility):
           - If already communicated: 0
           - If have_image by any rover: 1 (communicate)
           - Else if any camera-rover pair (c,x) for target (o,m) is currently calibrated: 2 (take_image + communicate)
           - Else: 3 (calibrate + take_image + communicate)

    - We use only facts and bitwise checks for speed. We do not perform expensive state expansions.
    - This heuristic is admissible and consistent under unit action costs:
        * It ignores navigation/visibility constraints and other resource couplings, hence it provides a lower bound.
        * It never counts fewer steps than strictly needed in the best case under the domain rules above.

    Implementation details:
    - We pre-index all relevant facts (by predicate and arguments) once in __init__.
    - At evaluation time, we only use fast bitwise tests for goal satisfaction and auxiliary conditions.
    - We derive goal instances from model.goals via model.state_explicit_repr(model.goals) only once.

    Notes:
    - This heuristic focuses on the primary communicated_* goals observed in Rover benchmarks and scales to large instances.
    - It is robust to intermediate states and partial progress (have_* or calibrated present).
    """

    def __init__(self, model, rover_gpt5_heuristic_15: bool = True):
        super().__init__(model)

        # Internal indices
        self.fid_communicated_soil_by_p: Dict[str, int] = {}
        self.fid_communicated_rock_by_p: Dict[str, int] = {}
        self.fid_communicated_image_by_om: Dict[Tuple[str, str], int] = {}

        self.fids_have_soil_by_p: Dict[str, List[int]] = {}
        self.fids_have_rock_by_p: Dict[str, List[int]] = {}
        self.fids_have_image_by_om: Dict[Tuple[str, str], List[int]] = {}

        self.rovers_soil: Set[str] = set()
        self.rovers_rock: Set[str] = set()
        self.rovers_imaging: Set[str] = set()

        self.stores_by_rover: Dict[str, List[str]] = {}
        self.fid_empty_by_store: Dict[str, int] = {}

        # Imaging support/calibration structures
        self.on_board: Set[Tuple[str, str]] = set()             # (camera, rover)
        self.supports: Set[Tuple[str, str]] = set()              # (camera, mode)
        self.calibration_target: Set[Tuple[str, str]] = set()    # (camera, objective)
        self.fid_calibrated_by_cx: Dict[Tuple[str, str], int] = {}
        self.fids_calibrated_candidates_by_om: Dict[Tuple[str, str], List[int]] = {}
        self.candidate_exists_by_om: Dict[Tuple[str, str], bool] = {}

        # Goals unpacked
        self._soil_goals: List[str] = []            # list of p
        self._rock_goals: List[str] = []            # list of p
        self._image_goals: List[Tuple[str, str]] = []  # list of (o, m)

        # Build indices over facts once
        self._index_facts()

        # Extract communicated_* goals from the model goals bitset
        self._extract_goals()

        # Precompute imaging calibration candidates by (o,m)
        self._prepare_imaging_candidates()

    # ------------------------ Parsing and Indexing ------------------------
    def _parse_fact_name(self, name: str):
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

    def _index_facts(self):
        # Iterate all facts in the model once and index useful ones
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            # Communicated goals
            if pred == "communicated_soil_data" and len(args) == 1:
                self.fid_communicated_soil_by_p[args[0]] = fact.global_id
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.fid_communicated_rock_by_p[args[0]] = fact.global_id
            elif pred == "communicated_image_data" and len(args) == 2:
                self.fid_communicated_image_by_om[(args[0], args[1])] = fact.global_id

            # Intermediate achievements
            elif pred == "have_soil_analysis" and len(args) == 2:
                p = args[1]
                self.fids_have_soil_by_p.setdefault(p, []).append(fact.global_id)
            elif pred == "have_rock_analysis" and len(args) == 2:
                p = args[1]
                self.fids_have_rock_by_p.setdefault(p, []).append(fact.global_id)
            elif pred == "have_image" and len(args) == 3:
                o, m = args[1], args[2]
                self.fids_have_image_by_om.setdefault((o, m), []).append(fact.global_id)

            # Equipment
            elif pred == "equipped_for_soil_analysis" and len(args) == 1:
                self.rovers_soil.add(args[0])
            elif pred == "equipped_for_rock_analysis" and len(args) == 1:
                self.rovers_rock.add(args[0])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                self.rovers_imaging.add(args[0])

            # Store relations
            elif pred == "store_of" and len(args) == 2:
                s, x = args[0], args[1]
                self.stores_by_rover.setdefault(x, []).append(s)
            elif pred == "empty" and len(args) == 1:
                s = args[0]
                self.fid_empty_by_store[s] = fact.global_id

            # Imaging support
            elif pred == "on_board" and len(args) == 2:
                c, x = args[0], args[1]
                self.on_board.add((c, x))
            elif pred == "supports" and len(args) == 2:
                c, m = args[0], args[1]
                self.supports.add((c, m))
            elif pred == "calibration_target" and len(args) == 2:
                c, o = args[0], args[1]
                self.calibration_target.add((c, o))
            elif pred == "calibrated" and len(args) == 2:
                c, x = args[0], args[1]
                self.fid_calibrated_by_cx[(c, x)] = fact.global_id

    def _extract_goals(self):
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if not pred:
                continue
            if pred == "communicated_soil_data" and len(args) == 1:
                p = args[0]
                if p in self.fid_communicated_soil_by_p:
                    self._soil_goals.append(p)
            elif pred == "communicated_rock_data" and len(args) == 1:
                p = args[0]
                if p in self.fid_communicated_rock_by_p:
                    self._rock_goals.append(p)
            elif pred == "communicated_image_data" and len(args) == 2:
                o, m = args[0], args[1]
                if (o, m) in self.fid_communicated_image_by_om:
                    self._image_goals.append((o, m))
        # De-duplicate if needed
        self._soil_goals = list(dict.fromkeys(self._soil_goals))
        self._rock_goals = list(dict.fromkeys(self._rock_goals))
        self._image_goals = list(dict.fromkeys(self._image_goals))

    def _prepare_imaging_candidates(self):
        # For each (o,m), compute candidate calibrated fact IDs that would allow immediate take_image
        # Candidate if: on_board(c,x) and supports(c,m) and calibration_target(c,o)
        # Also record whether any candidate exists at all.
        for (c, x) in self.on_board:
            for (c2, m) in self.supports:
                if c2 != c:
                    continue
                for (c3, o) in self.calibration_target:
                    if c3 != c:
                        continue
                    key = (o, m)
                    fid_cal = self.fid_calibrated_by_cx.get((c, x), None)
                    if fid_cal is not None:
                        self.fids_calibrated_candidates_by_om.setdefault(key, []).append(fid_cal)
                    # Mark that such a (c,x) exists for (o,m)
                    self.candidate_exists_by_om[key] = True
        # For pairs without any candidate seen, mark false explicitly
        for (o, m) in self.fid_communicated_image_by_om.keys():
            if (o, m) not in self.candidate_exists_by_om:
                self.candidate_exists_by_om[(o, m)] = False

    # ----------------------------- Heuristic ------------------------------
    def __call__(self, state_bitwise: int, task_network: List[Any]) -> int:
        cost = 0

        # Soil goals
        for p in self._soil_goals:
            fid_comm = self.fid_communicated_soil_by_p.get(p, None)
            if fid_comm is None:
                continue
            if self._is_true(state_bitwise, fid_comm):
                continue  # already achieved
            # Not communicated yet
            # If any have_soil_analysis[x,p] present -> need 1 (communicate)
            have_list = self.fids_have_soil_by_p.get(p, [])
            if self._any_true(state_bitwise, have_list):
                cost += 1
            else:
                # Else: need sampling first. If any equipped rover has an empty store => 2 (sample+communicate); else 3 (drop+sample+communicate)
                if self._any_equipped_rover_has_empty_store(state_bitwise, self.rovers_soil):
                    cost += 2
                else:
                    cost += 3

        # Rock goals
        for p in self._rock_goals:
            fid_comm = self.fid_communicated_rock_by_p.get(p, None)
            if fid_comm is None:
                continue
            if self._is_true(state_bitwise, fid_comm):
                continue
            have_list = self.fids_have_rock_by_p.get(p, [])
            if self._any_true(state_bitwise, have_list):
                cost += 1
            else:
                if self._any_equipped_rover_has_empty_store(state_bitwise, self.rovers_rock):
                    cost += 2
                else:
                    cost += 3

        # Image goals
        for (o, m) in self._image_goals:
            fid_comm = self.fid_communicated_image_by_om.get((o, m), None)
            if fid_comm is None:
                continue
            if self._is_true(state_bitwise, fid_comm):
                continue
            # If have_image for (o,m) already -> 1 (communicate)
            have_list = self.fids_have_image_by_om.get((o, m), [])
            if self._any_true(state_bitwise, have_list):
                cost += 1
            else:
                # If any calibrated candidate exists and is currently calibrated -> 2 (take_image+communicate)
                cal_list = self.fids_calibrated_candidates_by_om.get((o, m), [])
                if self._any_true(state_bitwise, cal_list):
                    cost += 2
                else:
                    # Else: 3 (calibrate + take_image + communicate) if at least one feasible (c,x) exists for (o,m)
                    # If none exists (pathological input), still return 3 (a safe lower bound that won't misguide completeness)
                    cost += 3

        return cost

    # --------------------------- Helper methods ---------------------------
    def _is_true(self, state_bitwise: int, fid: int) -> bool:
        return bool((state_bitwise >> fid) & 1)

    def _any_true(self, state_bitwise: int, fids: List[int]) -> bool:
        for fid in fids:
            if ((state_bitwise >> fid) & 1):
                return True
        return False

    def _any_equipped_rover_has_empty_store(self, state_bitwise: int, equipped_rovers: Set[str]) -> bool:
        # Check if there exists any rover in equipped_rovers that has at least one store s with empty[s] true
        for x in equipped_rovers:
            stores = self.stores_by_rover.get(x, [])
            if not stores:
                # If store is unknown for a rover (unlikely), assume it can be empty to keep the estimate a lower bound
                return True
            for s in stores:
                fid_empty = self.fid_empty_by_store.get(s, None)
                if fid_empty is not None and self._is_true(state_bitwise, fid_empty):
                    return True
        return False


# Alias for external configuration if needed
rover_gpt5_heuristic_15 = RoverGPT5Heuristic15
