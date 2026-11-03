from typing import Dict, List, Set, Tuple, Any

try:
    # Pytrich base heuristic
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback shim for static checkers/tests (not used by Pytrich runtime)
    class Heuristic:
        def __init__(self, model: Any):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network: List[Any]) -> int:
            return 0

# NOTE ON FACTS FORMAT
# Facts are stored with names like '+predicate[arg1,arg2]'. Use bitwise state to test truth efficiently.

class RoverGPT5Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for HDDL Rover domain.

    Design rationale (lower-bound, domain-informed):
    - The ultimate goals are communicated_* facts. Each unsatisfied communicated_* goal requires
      at least one communication action. Some require preceding data acquisition:
        * Soil/Rock: if have_*_analysis for the target waypoint is not yet present, at least one
          sampling action is needed. If no EQUIPPED rover is currently at that waypoint, at least
          one navigate is required before sampling (lower bound of 1 nav step).
        * Image: if have_image for (objective, mode) is not yet present, we must produce it.
          Lower-bound production cost is estimated as follows, for any rover x with onboard camera c
          such that calibration_target(c, objective) and supports(c, mode) and equipped_for_imaging(x):
            - If x is currently at any waypoint p with visible_from(objective, p):
                - if calibrated(c,x): need at least 1 action (take_image)
                - else: at least 2 actions (calibrate + take_image)
            - Else (x not at a visible waypoint):
                - if calibrated(c,x): at least 2 (navigate + take_image)
                - else: at least 3 (navigate + calibrate + take_image)
          We take the minimum over all suitable (x,c) pairs, then add 1 for the communication itself.
    - Additionally, as an HTN-aware tie-breaker we count the number of primitive operators already in the
      current task network; this is a lower bound on remaining cost because each primitive must be executed.
    - The final heuristic is the max between the goal-based lower bound and the operator-count lower bound
      (summing them could overcount shared work). This keeps the heuristic admissible and informative.

    This heuristic uses only bitwise state tests (fast) and pre-indexed facts; it scales to large Rover
    instances by avoiding per-state string scans.
    """

    def __init__(self, model, rover_gpt5_heuristic_11: bool = True):
        super().__init__(model)
        self._enabled = rover_gpt5_heuristic_11

        # --- Pre-index all facts by predicate/arguments for fast bitwise checks ---
        self.facts = model.facts

        # Maps
        self.fact_by_name: Dict[str, Any] = {}              # full "+pred[args]" -> Fact
        self.pred_index: Dict[str, List[Tuple[Any, Tuple[str, ...]]]] = {}  # pred -> list of (Fact, args)

        # Domain-specific indices
        self.visible_from: Dict[str, Set[str]] = {}         # objective -> set(waypoint)
        self.at_by_rover: Dict[str, List[Tuple[str, Any]]] = {}     # rover -> list of (waypoint, Fact(+at[r,p]))
        self.at_fact_by_r_p: Dict[Tuple[str, str], Any] = {}        # (rover, waypoint) -> Fact(+at)

        self.equipped_soil: Set[str] = set()                # rovers
        self.equipped_rock: Set[str] = set()                # rovers
        self.equipped_imaging: Set[str] = set()             # rovers

        self.onboard_cam_rover: Dict[str, str] = {}         # camera -> rover
        self.cameras_by_rover: Dict[str, Set[str]] = {}     # rover -> set(cameras)
        self.supports: Dict[str, Set[str]] = {}             # camera -> set(modes)
        self.calibration_target: Dict[str, str] = {}        # camera -> objective
        self.calibrated_fact: Dict[Tuple[str, str], Any] = {}  # (camera, rover) -> Fact(+calibrated)

        self.have_soil_by_p: Dict[str, List[Any]] = {}      # waypoint -> list of Fact(+have_soil_analysis[x,p])
        self.have_rock_by_p: Dict[str, List[Any]] = {}      # waypoint -> list of Fact(+have_rock_analysis[x,p])
        self.have_image_by_om: Dict[Tuple[str, str], List[Any]] = {}  # (objective, mode) -> list of Fact(+have_image[x,o,m])

        # Goal info pre-parsed
        self.goal_items: List[Tuple[Any, str, Tuple[str, ...]]] = []  # list of (goal_fact, goal_type, args)

        # Build indices from all ground facts in the model
        for fact in self.facts:
            self.fact_by_name[fact.name] = fact
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            self.pred_index.setdefault(pred, []).append((fact, tuple(args)))

            if pred == 'visible_from' and len(args) == 2:
                obj, wp = args
                self.visible_from.setdefault(obj, set()).add(wp)

            elif pred == 'at' and len(args) == 2:
                r, p = args
                self.at_by_rover.setdefault(r, []).append((p, fact))
                self.at_fact_by_r_p[(r, p)] = fact

            elif pred == 'equipped_for_soil_analysis' and len(args) == 1:
                self.equipped_soil.add(args[0])

            elif pred == 'equipped_for_rock_analysis' and len(args) == 1:
                self.equipped_rock.add(args[0])

            elif pred == 'equipped_for_imaging' and len(args) == 1:
                self.equipped_imaging.add(args[0])

            elif pred == 'on_board' and len(args) == 2:
                cam, rov = args
                self.onboard_cam_rover[cam] = rov
                self.cameras_by_rover.setdefault(rov, set()).add(cam)

            elif pred == 'supports' and len(args) == 2:
                cam, mode = args
                self.supports.setdefault(cam, set()).add(mode)

            elif pred == 'calibration_target' and len(args) == 2:
                cam, obj = args
                self.calibration_target[cam] = obj

            elif pred == 'calibrated' and len(args) == 2:
                cam, rov = args
                self.calibrated_fact[(cam, rov)] = fact

            elif pred == 'have_soil_analysis' and len(args) == 2:
                rov, wp = args
                self.have_soil_by_p.setdefault(wp, []).append(fact)

            elif pred == 'have_rock_analysis' and len(args) == 2:
                rov, wp = args
                self.have_rock_by_p.setdefault(wp, []).append(fact)

            elif pred == 'have_image' and len(args) == 3:
                rov, obj, mode = args
                self.have_image_by_om.setdefault((obj, mode), []).append(fact)

        # Precompute feasible (rover, camera) pairs for each (objective, mode)
        self.image_pairs_by_om: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
        for cam, obj in self.calibration_target.items():
            if cam not in self.onboard_cam_rover:
                continue
            rov = self.onboard_cam_rover[cam]
            if rov not in self.equipped_imaging:
                continue
            modes = self.supports.get(cam, set())
            for mode in modes:
                self.image_pairs_by_om.setdefault((obj, mode), []).append((rov, cam))

        # Parse goal facts into typed structures
        for gname in self.model.state_explicit_repr(self.model.goals):
            gf = self.fact_by_name.get(gname, None)
            if gf is None:
                # should not happen, but keep robust
                continue
            pred, args = self._parse_fact_name(gname)
            if pred == 'communicated_soil_data' and len(args) == 1:
                self.goal_items.append((gf, 'soil', (args[0],)))
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.goal_items.append((gf, 'rock', (args[0],)))
            elif pred == 'communicated_image_data' and len(args) == 2:
                self.goal_items.append((gf, 'image', (args[0], args[1])))
            else:
                # Unknown goal type: still include, with minimal 1-action LB
                self.goal_items.append((gf, 'other', tuple(args)))

    # ---------------- Heuristic API ----------------
    def evaluate(self, state_bitwise: int, task_network: List[Any]) -> int:
        if not self._enabled:
            return 0

        # Lower bound from remaining goals
        goal_lb = 0
        for goal_fact, gtype, gargs in self.goal_items:
            if self._is_true(state_bitwise, goal_fact):
                continue  # already satisfied

            if gtype == 'soil':
                p = gargs[0]
                goal_lb += self._soil_goal_lb(state_bitwise, p)
            elif gtype == 'rock':
                p = gargs[0]
                goal_lb += self._rock_goal_lb(state_bitwise, p)
            elif gtype == 'image':
                o, m = gargs
                goal_lb += self._image_goal_lb(state_bitwise, o, m)
            else:
                # Unknown goal fact: at least 1 action to achieve
                goal_lb += 1

        # Lower bound from already-instantiated primitive operators in the task network
        op_lb = 0
        if task_network:
            for t in task_network:
                # Operators in Pytrich have apply/relaxed_apply; abstract tasks/decomps do not.
                if hasattr(t, 'apply') and hasattr(t, 'applicable'):
                    op_lb += 1

        # Admissible combination: max of independent lower bounds
        return max(goal_lb, op_lb)

    # Allow heuristic(state, tn) call style
    __call__ = evaluate

    # ---------------- Internals: cost components ----------------
    def _soil_goal_lb(self, state: int, waypoint: str) -> int:
        # Base: must perform at least 1 communication action
        cost = 1
        # If already have soil analysis for this waypoint (any rover), we can communicate directly
        have_list = self.have_soil_by_p.get(waypoint, [])
        if self._any_true(state, have_list):
            return cost
        # Else: we must sample (at least 1 action)
        cost += 1
        # If no equipped rover is currently at that waypoint, at least one navigate is needed
        if not self._exists_equipped_at_waypoint(state, waypoint, self.equipped_soil):
            cost += 1
        return cost

    def _rock_goal_lb(self, state: int, waypoint: str) -> int:
        cost = 1
        have_list = self.have_rock_by_p.get(waypoint, [])
        if self._any_true(state, have_list):
            return cost
        cost += 1
        if not self._exists_equipped_at_waypoint(state, waypoint, self.equipped_rock):
            cost += 1
        return cost

    def _image_goal_lb(self, state: int, objective: str, mode: str) -> int:
        # Base: 1 communicate action
        base_comm = 1
        # If image already taken (any rover), we only need to communicate
        have_list = self.have_image_by_om.get((objective, mode), [])
        if self._any_true(state, have_list):
            return base_comm

        # Else, need to produce have_image first; compute minimal extra across feasible (rover,camera)
        min_extra = 3  # default lower bound if not at visible and not calibrated
        candidates = self.image_pairs_by_om.get((objective, mode), [])
        vis_set = self.visible_from.get(objective, set())

        for (rov, cam) in candidates:
            cal_fact = self.calibrated_fact.get((cam, rov), None)
            is_cal = self._is_true(state, cal_fact) if cal_fact is not None else False
            at_visible = self._rover_currently_at_any(state, rov, vis_set)

            if at_visible:
                extra = 1 if is_cal else 2  # (take) or (calibrate + take)
            else:
                extra = 2 if is_cal else 3  # (navigate + take) or (navigate + calibrate + take)

            if extra < min_extra:
                min_extra = extra

        return base_comm + min_extra

    # ---------------- Helper utilities ----------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[str, List[str]]:
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state: int, fact: Any) -> bool:
        if fact is None:
            return False
        return ((state >> fact.global_id) & 1) == 1

    @staticmethod
    def _any_true(state: int, facts: List[Any]) -> bool:
        for f in facts:
            if ((state >> f.global_id) & 1) == 1:
                return True
        return False

    def _exists_equipped_at_waypoint(self, state: int, waypoint: str, equipped_set: Set[str]) -> bool:
        # Returns True iff there exists rover r in equipped_set currently at 'waypoint'
        for r in equipped_set:
            f = self.at_fact_by_r_p.get((r, waypoint), None)
            if f is not None and self._is_true(state, f):
                return True
        return False

    def _rover_currently_at_any(self, state: int, rover: str, waypoints: Set[str]) -> bool:
        # Returns True iff rover is at any waypoint in the given set
        at_list = self.at_by_rover.get(rover, [])
        for (wp, f) in at_list:
            if wp in waypoints and self._is_true(state, f):
                return True
        return False
