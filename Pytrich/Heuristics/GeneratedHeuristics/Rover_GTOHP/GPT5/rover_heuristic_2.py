from typing import Dict, List, Tuple, Optional, Iterable

try:
    # Pytrich exports Heuristic in its heuristics package
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback class if running standalone (for type hints/tests); will be overridden in planner
    class Heuristic:
        def __init__(self, model):
            self.model = model
        def h(self, state_bitwise: int, task_network: List) -> int:
            return 0


class RoverGPT5Heuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Rover HDDL domain.

    Rationale (lower-bound, efficient, domain-aware):
    - Goals are about communicated_* data. To communicate any piece of data, at least one primitive
      communicate_* action is needed (lower bound = 1 op per unsatisfied goal).
    - For soil/rock data: if it is not yet communicated, but already sampled (have_*_analysis), the
      remaining lower bound is 1 (the communicate action). If not yet sampled, a sample action is
      needed (+1), plus a communicate (+1) => lower bound 2.
      (We ignore navigation and potential store drops for admissibility; this keeps the estimate a
      true lower bound and very fast.)
    - For image data: if it is not communicated but have_image exists, then lower bound is 1 (send).
      If an appropriate rover+camera is already calibrated for the objective/mode, then LB is 2
      (take_image + communicate). Otherwise, if a capable pair exists (on_board/supports/target/
      equipped), then LB is 3 (calibrate + take_image + communicate). If no capable pair exists we
      add a small penalty (heuristic remains a lower bound under solvable instances).

    This heuristic uses only bitwise checks over pre-indexed facts and avoids any explicit state
    reconstruction, which makes it scale well to the largest instances.
    """

    def __init__(self, model, rover_gpt5_heuristic_2: bool = True):
        super().__init__(model)
        self.model = model

        # Fast name->Fact mapping (only positive facts are relevant for membership checks)
        self.fact_by_name: Dict[str, object] = {}
        # (predicate, args_tuple) -> Fact (positive facts only)
        self.pos_fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}

        for fact in self.model.facts:
            name = fact.name
            if not name:
                continue
            self.fact_by_name[name] = fact
            pred, args = self._parse_fact_name(name)
            if pred is not None and name.startswith("+"):
                key = (pred, tuple(args))
                self.pos_fact_by_key[key] = fact

        # Pre-extract goal requirements (soil, rock, image)
        self.goal_soil: List[Tuple[str, object]] = []      # list of (wp, goal_fact)
        self.goal_rock: List[Tuple[str, object]] = []      # list of (wp, goal_fact)
        self.goal_image: List[Tuple[Tuple[str, str], object]] = []  # list of ((obj, mode), goal_fact)

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "communicated_soil_data" and len(args) == 1:
                f = self.fact_by_name.get(gname)
                if f is not None:
                    self.goal_soil.append((args[0], f))
            elif pred == "communicated_rock_data" and len(args) == 1:
                f = self.fact_by_name.get(gname)
                if f is not None:
                    self.goal_rock.append((args[0], f))
            elif pred == "communicated_image_data" and len(args) == 2:
                f = self.fact_by_name.get(gname)
                if f is not None:
                    self.goal_image.append(((args[0], args[1]), f))

        # Preindex helper maps for quick state checks
        # have_soil_analysis[r, wp], have_rock_analysis[r, wp]
        self.have_soil_by_wp: Dict[str, List[object]] = {}
        self.have_rock_by_wp: Dict[str, List[object]] = {}

        # have_image[r, obj, mode]
        self.have_image_by_om: Dict[Tuple[str, str], List[object]] = {}

        # Capabilities for imaging: for each (obj, mode) -> list of calibrated[c, r] facts
        # where there exists on_board[c, r], supports[c, mode], calibration_target[c, obj],
        # and equipped_for_imaging[r]. If any of these calibrated facts is true, we need only
        # take_image + communicate (LB=2). Otherwise if at least a capable pair exists, LB=3.
        self.calibrated_for_om: Dict[Tuple[str, str], List[object]] = {}

        # Auxiliary indices to compute calibrated_for_om
        on_board: List[Tuple[str, str]] = []  # (camera, rover)
        supports: Dict[str, List[str]] = {}   # camera -> [modes]
        cal_target: Dict[str, List[str]] = {} # camera -> [objectives]
        equipped_imaging: set = set()         # rovers equipped for imaging

        for fact in self.model.facts:
            name = fact.name
            if not name or not name.startswith("+"):
                continue
            pred, args = self._parse_fact_name(name)
            if pred == "have_soil_analysis" and len(args) == 2:
                # args: rover, wp
                self.have_soil_by_wp.setdefault(args[1], []).append(fact)
            elif pred == "have_rock_analysis" and len(args) == 2:
                self.have_rock_by_wp.setdefault(args[1], []).append(fact)
            elif pred == "have_image" and len(args) == 3:
                # args: rover, obj, mode
                key = (args[1], args[2])
                self.have_image_by_om.setdefault(key, []).append(fact)
            elif pred == "on_board" and len(args) == 2:
                on_board.append((args[0], args[1]))  # (camera, rover)
            elif pred == "supports" and len(args) == 2:
                supports.setdefault(args[0], []).append(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                cal_target.setdefault(args[0], []).append(args[1])
            elif pred == "equipped_for_imaging" and len(args) == 1:
                equipped_imaging.add(args[0])

        # Build calibrated mappings for each (obj, mode)
        # calibrated[camera, rover] fact must exist to check current calibration
        for (cam, rov) in on_board:
            if rov not in equipped_imaging:
                continue
            cam_modes = supports.get(cam, [])
            cam_objs = cal_target.get(cam, [])
            for m in cam_modes:
                for o in cam_objs:
                    key = (o, m)
                    calib_fact = self.pos_fact_by_key.get(("calibrated", (cam, rov)))
                    if calib_fact is not None:
                        self.calibrated_for_om.setdefault(key, []).append(calib_fact)

    # ------------------------------------------------------------------
    # Core heuristic: lower bound remaining steps to reach goal
    # ------------------------------------------------------------------
    def h(self, state_bitwise: int, task_network: List) -> int:
        cost = 0

        # Soil data goals
        for wp, goal_fact in self.goal_soil:
            if self._is_true(state_bitwise, goal_fact):
                continue
            # If data already sampled somewhere, only communication remains (LB=1)
            if self._any_true(state_bitwise, self.have_soil_by_wp.get(wp, [])):
                cost += 1
            else:
                # Need at least a sample + a communicate (LB=2)
                cost += 2

        # Rock data goals
        for wp, goal_fact in self.goal_rock:
            if self._is_true(state_bitwise, goal_fact):
                continue
            if self._any_true(state_bitwise, self.have_rock_by_wp.get(wp, [])):
                cost += 1
            else:
                cost += 2

        # Image goals
        for (obj, mode), goal_fact in self.goal_image:
            if self._is_true(state_bitwise, goal_fact):
                continue

            # If image already taken by any rover, only communication remains
            if self._any_true(state_bitwise, self.have_image_by_om.get((obj, mode), [])):
                cost += 1
                continue

            # Else, check if any capable pair is currently calibrated
            calib_list = self.calibrated_for_om.get((obj, mode), [])
            if self._any_true(state_bitwise, calib_list):
                # Need take_image + communicate (LB=2)
                cost += 2
            else:
                # If any capable pair exists (even if not calibrated), calibrate + take + communicate (LB=3)
                if len(calib_list) > 0:
                    cost += 3
                else:
                    # No capable pair found in the model for this (obj, mode). This likely won't happen
                    # in solvable instances; keep a tiny fallback lower bound of 1 to stay safe.
                    cost += 1

        return cost

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            core = name[1:]
            bracket_idx = core.find("[")
            if bracket_idx > 0 and core.endswith("]"):
                predicate = core[:bracket_idx]
                args_str = core[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact) -> bool:
        if fact is None:
            return False
        return ((state_bitwise >> fact.global_id) & 1) == 1

    @staticmethod
    def _any_true(state_bitwise: int, facts: Optional[Iterable[object]]) -> bool:
        if not facts:
            return False
        for f in facts:
            if ((state_bitwise >> f.global_id) & 1) == 1:
                return True
        return False


# Alias to keep Pytrich discovery simple if it inspects module attributes
HeuristicClass = RoverGPT5Heuristic2
