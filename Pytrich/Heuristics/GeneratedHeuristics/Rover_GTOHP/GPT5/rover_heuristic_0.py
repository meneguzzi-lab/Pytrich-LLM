from typing import Dict, Tuple, List, Set, Optional

try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub if the framework import path differs during static checks.
    class Heuristic:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass


class RoverGPT5Heuristic0(Heuristic):
    """
    RoverGPT5Heuristic0: Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale and domain insights:
    - The Rover domain goals center around communicating soil, rock, and image data.
    - Each high-level task decomposes into a small set of primitives: navigation(s), sampling, calibration, imaging, and communication.
    - We exploit domain structure to derive admissible lower bounds on the remaining number of primitive actions by:
        * Estimating the minimal number of navigation steps as 0 or 1 (a safe lower bound), depending on whether a rover is already at a suitable waypoint.
        * Counting mandatory primitives (sample/rock/image/communication/calibration) when needed.
        * Considering current state facts such as whether stores are full/empty, whether data has already been acquired, and visibility to the lander.
    - We compute two admissible estimates and take their maximum (for robustness and to avoid double counting):
        1) Task-network-based estimate: sums a lower bound of the remaining primitives implied by each (abstract or primitive) task currently in the task network.
        2) Goal-based estimate: sums a lower bound of primitives required to achieve each outstanding communicated_* fact.

    Notes on admissibility and efficiency:
    - Navigation lengths are lower-bounded by 0 if already at the required waypoint; otherwise by 1. This is safe (true path length may be >1).
    - Communication always costs at least 1 primitive action; we add a 0-or-1 extra navigation bound to reach a comm-visible waypoint if necessary.
    - Sampling and imaging add fixed mandatory actions (1 for sample_*, 1 for calibrate, 1 for take_image).
    - We pre-index relevant facts and arguments for O(1) bitwise checks during heuristic evaluation; we never expand full explicit state representations at runtime.

    Implementation details:
    - Uses model.state_explicit_repr(model.goals) once at init to parse goal facts (bitwise int in Pytrich).
    - Maintains fast lookups for 'at[rover,wp]', 'visible[wp1,wp2]', 'store_of', 'empty/full', 'equipped_*', 'on_board', 'supports', 'calibration_target', 'visible_from'.
    - For each heuristic call:
        * Reads current rover positions using bitwise checks.
        * Estimates per-task lower bounds and per-unsatisfied-goal lower bounds.
        * Returns max(task_lb, goal_lb).

    Heuristic function value: non-negative integer.
    """

    def __init__(self, model, rover_gpt5_heuristic_0: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.cfg = rover_gpt5_heuristic_0 or {}

        # --- General-purpose maps over facts ---
        # Map of full fact name string -> Fact (for quick id lookup)
        self.fact_name_to_fact: Dict[str, object] = {}

        # Parse-and-store helpers
        self.rovers: Set[str] = set()
        self.stores: Set[str] = set()
        self.waypoints: Set[str] = set()
        self.cameras: Set[str] = set()
        self.objectives: Set[str] = set()
        self.modes: Set[str] = set()
        self.lander_wp: Optional[str] = None

        # Visibility relations: visible[wp1,wp2] and set of wps visible to lander
        self.visible_pairs: Set[Tuple[str, str]] = set()
        self.visible_to_lander: Set[str] = set()

        # Indexed 'at' facts: at[rover,wp] -> fact_id and per-rover reverse map
        self.at_fact_ids: Dict[Tuple[str, str], int] = {}
        self.at_by_rover: Dict[str, Dict[str, int]] = {}

        # Store facts
        self.empty_fact_ids: Dict[str, int] = {}
        self.full_fact_ids: Dict[str, int] = {}
        self.store_of: Dict[str, str] = {}  # store -> rover
        self.rover_store: Dict[str, str] = {}  # rover -> store

        # Equipment
        self.equipped_soil: Set[str] = set()
        self.equipped_rock: Set[str] = set()
        self.equipped_imaging: Set[str] = set()

        # Imaging-related indices
        self.on_board: Dict[str, str] = {}              # camera -> rover
        self.supports: Dict[str, Set[str]] = {}         # camera -> set(modes)
        self.calibration_target: Dict[str, str] = {}    # camera -> objective
        self.visible_from: Dict[str, Set[str]] = {}     # objective -> set(waypoints)

        # Precompute set of objectives whose vantage points include any lander-visible waypoint
        self.obj_has_lander_visible_vantage: Dict[str, bool] = {}

        # Goal facts parsed once
        self.goal_communicated_soil: List[str] = []                 # waypoints
        self.goal_communicated_rock: List[str] = []                 # waypoints
        self.goal_communicated_image: List[Tuple[str, str]] = []    # (objective, mode)
        self.goal_fact_ids: Set[int] = set()

        self._index_model_facts()
        self._index_goals()
        self._precompute_objective_visibility_intersections()

    # ------------------------------ Index building ------------------------------
    def _parse_bracketed(self, name: str) -> Tuple[str, List[str]]:
        name = name.strip()
        if not name:
            return name, []
        # Fact names may come with "+"/"-" in explicit repr; operators/tasks should not.
        if name[0] in "+-":
            name = name[1:]
        lb = name.find("[")
        if lb == -1 or not name.endswith("]"):
            return name, []
        head = name[:lb]
        args_str = name[lb + 1:-1]
        args = args_str.split(",") if args_str else []
        return head, args

    def _index_model_facts(self):
        # Build dictionaries for fast bitwise membership checks by global_id
        for f in self.model.facts:
            self.fact_name_to_fact[f.name] = f
            pred, args = self._parse_bracketed(f.name)
            if not pred:
                continue

            # Collect object names opportunistically
            try:
                if pred == 'available' and len(args) == 1:
                    self.rovers.add(args[0])
                elif pred == 'at' and len(args) == 2:
                    r, w = args
                    self.rovers.add(r)
                    self.waypoints.add(w)
                    self.at_fact_ids[(r, w)] = f.global_id
                    self.at_by_rover.setdefault(r, {})[w] = f.global_id
                elif pred == 'visible' and len(args) == 2:
                    w1, w2 = args
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
                    self.visible_pairs.add((w1, w2))
                elif pred == 'can_traverse' and len(args) == 3:
                    r, w1, w2 = args
                    self.rovers.add(r)
                    self.waypoints.add(w1)
                    self.waypoints.add(w2)
                elif pred == 'store_of' and len(args) == 2:
                    s, r = args
                    self.stores.add(s)
                    self.rovers.add(r)
                    self.store_of[s] = r
                    self.rover_store[r] = s
                elif pred == 'empty' and len(args) == 1:
                    s = args[0]
                    self.stores.add(s)
                    self.empty_fact_ids[s] = f.global_id
                elif pred == 'full' and len(args) == 1:
                    s = args[0]
                    self.stores.add(s)
                    self.full_fact_ids[s] = f.global_id
                elif pred == 'equipped_for_soil_analysis' and len(args) == 1:
                    self.rovers.add(args[0])
                    self.equipped_soil.add(args[0])
                elif pred == 'equipped_for_rock_analysis' and len(args) == 1:
                    self.rovers.add(args[0])
                    self.equipped_rock.add(args[0])
                elif pred == 'equipped_for_imaging' and len(args) == 1:
                    self.rovers.add(args[0])
                    self.equipped_imaging.add(args[0])
                elif pred == 'calibration_target' and len(args) == 2:
                    c, o = args
                    self.cameras.add(c)
                    self.objectives.add(o)
                    self.calibration_target[c] = o
                elif pred == 'visible_from' and len(args) == 2:
                    o, w = args
                    self.objectives.add(o)
                    self.waypoints.add(w)
                    self.visible_from.setdefault(o, set()).add(w)
                elif pred == 'on_board' and len(args) == 2:
                    c, r = args
                    self.cameras.add(c)
                    self.rovers.add(r)
                    self.on_board[c] = r
                elif pred == 'supports' and len(args) == 2:
                    c, m = args
                    self.cameras.add(c)
                    self.modes.add(m)
                    self.supports.setdefault(c, set()).add(m)
                elif pred == 'at_lander' and len(args) == 2:
                    # Exactly one lander assumed in problems; take its waypoint
                    _l, w = args
                    self.waypoints.add(w)
                    self.lander_wp = w
                elif pred == 'have_soil_analysis' and len(args) == 2:
                    r, w = args
                    self.rovers.add(r)
                    self.waypoints.add(w)
                elif pred == 'have_rock_analysis' and len(args) == 2:
                    r, w = args
                    self.rovers.add(r)
                    self.waypoints.add(w)
                elif pred == 'have_image' and len(args) == 3:
                    r, o, m = args
                    self.rovers.add(r)
                    self.objectives.add(o)
                    self.modes.add(m)
                elif pred == 'communicated_soil_data' and len(args) == 1:
                    self.waypoints.add(args[0])
                elif pred == 'communicated_rock_data' and len(args) == 1:
                    self.waypoints.add(args[0])
                elif pred == 'communicated_image_data' and len(args) == 2:
                    o, m = args
                    self.objectives.add(o)
                    self.modes.add(m)
            except Exception:
                # Be liberal in parsing; ignore any anomalies
                pass

        # Compute the set of waypoints from which the lander is visible
        if self.lander_wp is not None:
            for (w1, w2) in self.visible_pairs:
                if w2 == self.lander_wp:
                    self.visible_to_lander.add(w1)

    def _index_goals(self):
        # Parse bitwise goals once
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            fact = self.fact_name_to_fact.get(gname)
            if fact is None:
                continue
            self.goal_fact_ids.add(fact.global_id)
            pred, args = self._parse_bracketed(gname)
            if pred == 'communicated_soil_data' and len(args) == 1:
                self.goal_communicated_soil.append(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.goal_communicated_rock.append(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                self.goal_communicated_image.append((args[0], args[1]))

    def _precompute_objective_visibility_intersections(self):
        # Whether an objective has any vantage point that is also visible to the lander
        for o, vset in self.visible_from.items():
            self.obj_has_lander_visible_vantage[o] = any((w in self.visible_to_lander) for w in vset)

    # ------------------------------ Utility bit checks ------------------------------
    @staticmethod
    def _is_true_bit(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _fact_id(self, pred: str, args: List[str]) -> Optional[int]:
        name = f"+{pred}[{','.join(args)}]"
        f = self.fact_name_to_fact.get(name)
        return f.global_id if f is not None else None

    # ------------------------------ Current state decoders ------------------------------
    def _current_rover_positions(self, state_bitwise: int) -> Dict[str, Optional[str]]:
        pos: Dict[str, Optional[str]] = {r: None for r in self.rovers}
        for r in self.rovers:
            rmap = self.at_by_rover.get(r, {})
            for w, fact_id in rmap.items():
                if self._is_true_bit(state_bitwise, fact_id):
                    pos[r] = w
                    break
        return pos

    def _store_fullness(self, state_bitwise: int) -> Dict[str, Tuple[bool, bool]]:
        # Returns for each rover: (is_empty, is_full) based on its associated store (if any)
        res: Dict[str, Tuple[bool, bool]] = {}
        for r in self.rovers:
            s = self.rover_store.get(r)
            if s is None:
                res[r] = (True, False)  # treat as empty/no cost by default (admissible)
                continue
            e_id = self.empty_fact_ids.get(s)
            f_id = self.full_fact_ids.get(s)
            is_empty = self._is_true_bit(state_bitwise, e_id) if e_id is not None else False
            is_full = self._is_true_bit(state_bitwise, f_id) if f_id is not None else False
            res[r] = (is_empty, is_full)
        return res

    # ------------------------------ Lower-bound building blocks ------------------------------
    def _lb_nav_to_waypoint(self, rover_pos: Optional[str], target_w: str) -> int:
        # Lower bound on number of navigate actions to reach 'target_w'
        # 0 if already there, otherwise at least 1.
        if rover_pos == target_w:
            return 0
        return 1

    def _lb_nav_to_any(self, rover_pos: Optional[str], candidate_wps: Set[str]) -> int:
        # 0 if already at any candidate waypoint; otherwise at least 1.
        if rover_pos is not None and rover_pos in candidate_wps:
            return 0
        return 1 if candidate_wps else 0

    def _lb_comm_extra_nav(self, from_wp: Optional[str], requires_visible_to_lander: bool) -> int:
        # If a rover must be at a waypoint visible to the lander, the LB is:
        #   0 if already at such a waypoint, otherwise 1 (provided some visible wp exists)
        if not requires_visible_to_lander:
            return 0
        if from_wp in self.visible_to_lander:
            return 0
        # If no waypoint is visible to the lander at all (degenerate), 0 (admissible)
        return 1 if len(self.visible_to_lander) > 0 else 0

    # ------------------------------ Abstract task cost estimators ------------------------------
    def _lb_empty_store(self, state_bitwise: int, s: str) -> int:
        # 0 if empty[s]; else if full[s] -> 1; else 0 (admissible)
        e_id = self.empty_fact_ids.get(s)
        if e_id is not None and self._is_true_bit(state_bitwise, e_id):
            return 0
        f_id = self.full_fact_ids.get(s)
        if f_id is not None and self._is_true_bit(state_bitwise, f_id):
            return 1
        return 0

    def _lb_get_soil_data(self, state_bitwise: int, from_wp: str, rover_pos: Dict[str, Optional[str]], store_status: Dict[str, Tuple[bool, bool]]) -> int:
        # For any rover with soil equipment, compute:
        #   LB = drop(0/1 if its store full) + nav_to_from(0/1) + 1(sample) + nav_to_comm(0/1) + 1(communicate)
        # If any rover already has have_soil_analysis[x,from], we can skip sample & store handling, and just comm: 1 + nav_to_comm(0/1) from current wp of that rover.
        best = None
        for r in self.equipped_soil:
            # Check if analysis already available
            have_id = self._fact_id('have_soil_analysis', [r, from_wp])
            if have_id is not None and self._is_true_bit(state_bitwise, have_id):
                nav_comm = self._lb_comm_extra_nav(rover_pos.get(r), True)
                cand = 1 + nav_comm  # communicate only
            else:
                # Need sample and communicate
                _is_empty, is_full = store_status.get(r, (True, False))
                drop = 1 if is_full else 0
                nav1 = self._lb_nav_to_waypoint(rover_pos.get(r), from_wp)
                nav_comm = 0 if (from_wp in self.visible_to_lander) else 1
                cand = drop + nav1 + 1 + nav_comm + 1
            best = cand if best is None else min(best, cand)
        return best if best is not None else 0

    def _lb_get_rock_data(self, state_bitwise: int, from_wp: str, rover_pos: Dict[str, Optional[str]], store_status: Dict[str, Tuple[bool, bool]]) -> int:
        best = None
        for r in self.equipped_rock:
            have_id = self._fact_id('have_rock_analysis', [r, from_wp])
            if have_id is not None and self._is_true_bit(state_bitwise, have_id):
                nav_comm = self._lb_comm_extra_nav(rover_pos.get(r), True)
                cand = 1 + nav_comm
            else:
                _is_empty, is_full = store_status.get(r, (True, False))
                drop = 1 if is_full else 0
                nav1 = self._lb_nav_to_waypoint(rover_pos.get(r), from_wp)
                nav_comm = 0 if (from_wp in self.visible_to_lander) else 1
                cand = drop + nav1 + 1 + nav_comm + 1
            best = cand if best is None else min(best, cand)
        return best if best is not None else 0

    def _lb_get_image_data(self, state_bitwise: int, objective: str, mode: str, rover_pos: Dict[str, Optional[str]]) -> int:
        # For any rover with imaging, and any on-board camera supporting mode & calibrated target objective:
        #   If have_image[r,o,m]: LB = 1 + nav_comm(0/1)
        #   Else: LB = nav_vantage(0/1) + 1(calibrate) + 1(take_image) + nav_comm(0/1 if any vantage visible to lander else 1) + 1(communicate)
        vset = self.visible_from.get(objective, set())
        has_intersection = self.obj_has_lander_visible_vantage.get(objective, False)
        best = None

        # Build candidate rover-camera pairs satisfying objective/mode
        candidates: List[Tuple[str, str]] = []  # (rover, camera)
        for c, r in self.on_board.items():
            if c not in self.calibration_target:
                continue
            if self.calibration_target[c] != objective:
                continue
            if mode not in self.supports.get(c, set()):
                continue
            if r not in self.equipped_imaging:
                continue
            candidates.append((r, c))

        for (r, _c) in candidates:
            have_id = self._fact_id('have_image', [r, objective, mode])
            if have_id is not None and self._is_true_bit(state_bitwise, have_id):
                nav_comm = self._lb_comm_extra_nav(rover_pos.get(r), True)
                cand = 1 + nav_comm
            else:
                nav_vantage = self._lb_nav_to_any(rover_pos.get(r), vset)
                nav_comm = 0 if has_intersection else 1
                cand = nav_vantage + 1 + 1 + nav_comm + 1
            best = cand if best is None else min(best, cand)
        return best if best is not None else 0

    def _lb_send_soil_data(self, rover: str, from_wp: str, rover_pos: Dict[str, Optional[str]]) -> int:
        # Sending alone costs: 1(communicate) + nav_comm(0/1)
        # Nav LB: 0 if currently at any wp visible to lander, else 1.
        nav_comm = self._lb_comm_extra_nav(rover_pos.get(rover), True)
        return 1 + nav_comm

    def _lb_send_rock_data(self, rover: str, from_wp: str, rover_pos: Dict[str, Optional[str]]) -> int:
        nav_comm = self._lb_comm_extra_nav(rover_pos.get(rover), True)
        return 1 + nav_comm

    def _lb_send_image_data(self, rover: str, objective: str, mode: str, rover_pos: Dict[str, Optional[str]]) -> int:
        nav_comm = self._lb_comm_extra_nav(rover_pos.get(rover), True)
        return 1 + nav_comm

    def _lb_do_calibrate(self, rover: str, camera: str, rover_pos: Dict[str, Optional[str]]) -> int:
        # Calibration: 1(calibrate) + nav_vantage(0/1) where vantage satisfies visible_from(objective)
        obj = self.calibration_target.get(camera)
        if obj is None:
            return 1  # safest lower bound
        vset = self.visible_from.get(obj, set())
        nav_vantage = self._lb_nav_to_any(rover_pos.get(rover), vset)
        return nav_vantage + 1

    def _lb_do_navigate1(self, rover: str, to_wp: str, rover_pos: Dict[str, Optional[str]]) -> int:
        # 0 if already at(to), else at least 1 navigate
        return self._lb_nav_to_waypoint(rover_pos.get(rover), to_wp)

    def _lb_do_navigate2(self, rover: str, _from_wp: str, to_wp: str, rover_pos: Dict[str, Optional[str]]) -> int:
        # 0 if already at(to), else at least 1 navigate
        return self._lb_nav_to_waypoint(rover_pos.get(rover), to_wp)

    # ------------------------------ Task-network based estimate ------------------------------
    def _task_network_lower_bound(self, state_bitwise: int, task_network: List[int]) -> int:
        # Sum lower bounds over tasks and operators in the current TN
        if not task_network:
            return 0
        rover_pos = self._current_rover_positions(state_bitwise)
        store_status = self._store_fullness(state_bitwise)

        total_lb = 0
        for comp_id in task_network:
            comp = self.model.get_component(comp_id)
            name = getattr(comp, 'name', '')
            head, args = self._parse_bracketed(name)

            if comp_id >= self.model.iop_init and comp_id <= self.model.iop_end:
                # Primitive operator: at least its own cost (assumed 1 in this domain)
                total_lb += 1
                continue

            # Abstract tasks (best-effort coverage of Rover domain tasks)
            if head == 'get_soil_data' and len(args) == 1:
                total_lb += self._lb_get_soil_data(state_bitwise, args[0], rover_pos, store_status)
            elif head == 'get_rock_data' and len(args) == 1:
                total_lb += self._lb_get_rock_data(state_bitwise, args[0], rover_pos, store_status)
            elif head == 'get_image_data' and len(args) == 2:
                o, m = args
                total_lb += self._lb_get_image_data(state_bitwise, o, m, rover_pos)
            elif head == 'send_soil_data' and len(args) == 2:
                r, wp = args
                total_lb += self._lb_send_soil_data(r, wp, rover_pos)
            elif head == 'send_rock_data' and len(args) == 2:
                r, wp = args
                total_lb += self._lb_send_rock_data(r, wp, rover_pos)
            elif head == 'send_image_data' and len(args) == 3:
                r, o, m = args
                total_lb += self._lb_send_image_data(r, o, m, rover_pos)
            elif head == 'empty_store' and len(args) == 2:
                s, _r = args
                total_lb += self._lb_empty_store(state_bitwise, s)
            elif head == 'do_calibrate' and len(args) == 2:
                r, c = args
                total_lb += self._lb_do_calibrate(r, c, rover_pos)
            elif head == 'do_navigate1' and len(args) == 2:
                r, to_wp = args
                total_lb += self._lb_do_navigate1(r, to_wp, rover_pos)
            elif head == 'do_navigate2' and len(args) == 3:
                r, from_wp, to_wp = args
                total_lb += self._lb_do_navigate2(r, from_wp, to_wp, rover_pos)
            else:
                # Unknown task: return a very mild lower bound (0 or 1). We choose 0 to be safe.
                # If desired, this can be set to 1 to bias towards progress, still admissible but might be too weak/aggressive.
                total_lb += 0

        return max(0, total_lb)

    # ------------------------------ Goal-based estimate ------------------------------
    def _goals_lower_bound(self, state_bitwise: int) -> int:
        # Sum lower bounds for each unsatisfied communicated_* goal at this state
        rover_pos = self._current_rover_positions(state_bitwise)
        store_status = self._store_fullness(state_bitwise)
        total_lb = 0

        # communicated_soil_data[wp]
        for wp in self.goal_communicated_soil:
            fid = self._fact_id('communicated_soil_data', [wp])
            if fid is not None and self._is_true_bit(state_bitwise, fid):
                continue
            total_lb += self._lb_get_soil_data(state_bitwise, wp, rover_pos, store_status)

        # communicated_rock_data[wp]
        for wp in self.goal_communicated_rock:
            fid = self._fact_id('communicated_rock_data', [wp])
            if fid is not None and self._is_true_bit(state_bitwise, fid):
                continue
            total_lb += self._lb_get_rock_data(state_bitwise, wp, rover_pos, store_status)

        # communicated_image_data[o,m]
        for (o, m) in self.goal_communicated_image:
            fid = self._fact_id('communicated_image_data', [o, m])
            if fid is not None and self._is_true_bit(state_bitwise, fid):
                continue
            total_lb += self._lb_get_image_data(state_bitwise, o, m, rover_pos)

        return max(0, total_lb)

    # ------------------------------ Public heuristic API ------------------------------
    def compute(self, state_bitwise: int, task_network: List[int]) -> int:
        # Robust heuristic combines both estimates to avoid double counting and remain admissible
        if state_bitwise is None:
            return 0
        tn_lb = self._task_network_lower_bound(state_bitwise, task_network)
        goals_lb = self._goals_lower_bound(state_bitwise)
        return max(tn_lb, goals_lb)

    # Optional: Provide a friendly name
    def __str__(self):
        return 'RoverGPT5Heuristic0'

    __repr__ = __str__
