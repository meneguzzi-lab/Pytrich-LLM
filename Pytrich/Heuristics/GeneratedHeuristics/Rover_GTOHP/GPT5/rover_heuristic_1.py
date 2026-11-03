from typing import Dict, List, Tuple, Optional, Set

# Try both typical import paths used by Pytrich
try:
    from Pytrich.Heuristics.Heuristic import Heuristic
except Exception:  # pragma: no cover - fallback if different package root
    try:
        from Pytrich.Heuristics.heuristic import Heuristic
    except Exception:
        # Minimal fallback base class if framework import is unavailable at codegen time
        class Heuristic:  # type: ignore
            def __init__(self, model=None, name: str = "Heuristic"):
                self.model = model
                self.name = name


class RoverGPT5Heuristic1(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain tailored to Pytrich.

    Design rationale (concise):
    - Goal-based lower bound: For each unachieved communicated_* goal, we count mandatory remaining steps:
        * Soil/Rock: communicate (1) if already have_*_analysis somewhere, else sample + communicate (2).
        * Image (o, m): communicate (1) if have_image already; else take+communicate (2) if some relevant camera is
          already calibrated on-board and supports m; else calibrate+take+communicate (3).
      This ignores navigation and store management effects, yielding an admissible (optimistic) lower bound.

    - Task-network lower bound: We also compute a per-task minimal expansion cost ignoring navigation (except a cheap
      0/1 bound for do_navigate):
        * get_soil_data: >= 2 (sample + communicate)
        * get_rock_data: >= 2 (sample + communicate)
        * get_image_data: >= 3 (calibrate + take + communicate)
        * send_*: >= 1
        * do_calibrate: >= 1
        * empty_store: >= 0 (optimistic to remain admissible)
        * do_navigate1/2: >= 0 if already at destination, else >= 1
        * primitive operators left in the task network contribute their actual cost (usually 1 each).

      We take the maximum of (goal-based LB, task-network-based LB) for admissibility (prevents double-counting while
      ensuring both goal satisfaction and full task refinement are considered, as HTN planning requires empty TN).

    Performance notes:
    - We use bitwise checks on the state for truth evaluation, never expanding explicit state sets on the critical path.
    - We pre-index facts by predicate/arguments to allow quick "exists-any" checks (e.g., any rover has soil analysis
      for waypoint p?).

    Usage:
    - Instantiate with the model and the required lower-case parameter name: rover_gpt5_heuristic_1 (optional config).
    - The heuristic exposes evaluate(state_bitwise, task_network), h(state_bitwise, task_network), and __call__ aliases.
    """

    def __init__(self, model, rover_gpt5_heuristic_1: Optional[dict] = None):
        super().__init__(model=model, name="RoverGPT5Heuristic1")
        self.cfg = rover_gpt5_heuristic_1 or {}

        # Internal indices
        self.fact_by_name: Dict[str, object] = {}          # name -> Fact
        self.by_sig: Dict[Tuple[str, Tuple[str, ...]], object] = {}  # (pred, args) -> Fact
        self.goal_facts: List[object] = []                 # List[Fact] for goal bits

        # Domain-specific indices (store global_ids only for fast bit checks)
        self.have_soil_ids_by_p: Dict[str, List[int]] = {}
        self.have_rock_ids_by_p: Dict[str, List[int]] = {}
        self.have_image_ids_by_om: Dict[Tuple[str, str], List[int]] = {}
        self.calibrated_fact_ids: Dict[Tuple[str, str], int] = {}  # (camera, rover) -> fact id
        self.on_board_cam_to_rover: Dict[str, str] = {}     # camera -> rover
        self.supports_modes: Dict[str, Set[str]] = {}       # camera -> set(modes)
        self.caltargets_o_to_cams: Dict[str, Set[str]] = {} # objective -> set(cameras)
        self.calibrated_ids_for_om: Dict[Tuple[str, str], List[int]] = {}  # (obj, mode) -> [calibrated[c,x] ids]
        self.at_fact_ids: Dict[Tuple[str, str], int] = {}   # (rover, waypoint) -> fact id

        self._index_model()

    # Public API (be generous with entry points for compatibility)
    def evaluate(self, state_bitwise: int, task_network: Optional[List[object]] = None) -> int:
        return self._heuristic(state_bitwise, task_network)

    def h(self, state_bitwise: int, task_network: Optional[List[object]] = None) -> int:
        return self._heuristic(state_bitwise, task_network)

    def __call__(self, state_bitwise: int, task_network: Optional[List[object]] = None) -> int:
        return self._heuristic(state_bitwise, task_network)

    # --------------------------- Internal helpers --------------------------- #

    def _heuristic(self, state_bitwise: int, task_network: Optional[List[object]]) -> int:
        # Lower bound from remaining goals
        lb_goals = self._lb_from_goals(state_bitwise)
        # Lower bound from remaining tasks to refine (includes primitive ops in TN)
        lb_tasks = self._lb_from_task_network(state_bitwise, task_network or [])
        # Take the maximum to avoid double-counting yet enforce both conditions (goals and empty TN)
        return max(lb_goals, lb_tasks)

    def _index_model(self):
        # 1) Index all facts by name and by (pred, args)
        for f in self.model.facts:
            self.fact_by_name[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self.by_sig[key] = f

            # Domain-specific collections
            if pred == 'have_soil_analysis' and len(args) == 2:
                rover, wp = args
                self.have_soil_ids_by_p.setdefault(wp, []).append(f.global_id)
            elif pred == 'have_rock_analysis' and len(args) == 2:
                rover, wp = args
                self.have_rock_ids_by_p.setdefault(wp, []).append(f.global_id)
            elif pred == 'have_image' and len(args) == 3:
                rover, obj, mode = args
                self.have_image_ids_by_om.setdefault((obj, mode), []).append(f.global_id)
            elif pred == 'calibrated' and len(args) == 2:
                cam, rover = args
                self.calibrated_fact_ids[(cam, rover)] = f.global_id
            elif pred == 'on_board' and len(args) == 2:
                cam, rover = args
                self.on_board_cam_to_rover[cam] = rover
            elif pred == 'supports' and len(args) == 2:
                cam, mode = args
                self.supports_modes.setdefault(cam, set()).add(mode)
            elif pred == 'calibration_target' and len(args) == 2:
                cam, obj = args
                self.caltargets_o_to_cams.setdefault(obj, set()).add(cam)
            elif pred == 'at' and len(args) == 2:
                rover, wp = args
                self.at_fact_ids[(rover, wp)] = f.global_id

        # 2) Build goal facts list from model.goals (bitset -> names)
        for gname in self.model.state_explicit_repr(self.model.goals):
            fact = self.fact_by_name.get(gname)
            if fact is not None:
                self.goal_facts.append(fact)

        # 3) Precompute calibrated candidates for images per (objective, mode)
        #    For each objective o and mode m, collect calibrated facts for all cameras c s.t.
        #    calibration_target(c,o) and supports(c,m) and on_board(c,x)
        for o, cams in self.caltargets_o_to_cams.items():
            for c in cams:
                if c not in self.supports_modes:
                    continue
                rover = self.on_board_cam_to_rover.get(c)
                if rover is None:
                    continue
                for m in self.supports_modes[c]:
                    cal_id = self.calibrated_fact_ids.get((c, rover))
                    if cal_id is not None:
                        self.calibrated_ids_for_om.setdefault((o, m), []).append(cal_id)

    # Goal-based lower bound
    def _lb_from_goals(self, state: int) -> int:
        total = 0
        for gf in self.goal_facts:
            if self._is_true(state, gf.global_id):
                continue
            pred, args = self._parse_fact_name(gf.name)
            if pred is None:
                continue

            if pred == 'communicated_soil_data' and len(args) == 1:
                wp = args[0]
                ids = self.have_soil_ids_by_p.get(wp, [])
                if self._any_true(state, ids):
                    total += 1  # just communicate
                else:
                    total += 2  # sample + communicate (ignore navigation)

            elif pred == 'communicated_rock_data' and len(args) == 1:
                wp = args[0]
                ids = self.have_rock_ids_by_p.get(wp, [])
                if self._any_true(state, ids):
                    total += 1
                else:
                    total += 2  # sample + communicate

            elif pred == 'communicated_image_data' and len(args) == 2:
                obj, mode = args
                # If we already have the image, only communication is required
                have_ids = self.have_image_ids_by_om.get((obj, mode), [])
                if self._any_true(state, have_ids):
                    total += 1
                else:
                    # If some relevant camera is already calibrated: at least take + communicate
                    cand_ids = self.calibrated_ids_for_om.get((obj, mode), [])
                    if self._any_true(state, cand_ids):
                        total += 2
                    else:
                        total += 3  # calibrate + take + communicate (ignore navigation)

            else:
                # Unknown goal type (not expected in Rover benchmarks). Be safely optimistic.
                total += 0
        return total

    # Task-network lower bound (includes primitive operators still in TN)
    def _lb_from_task_network(self, state: int, task_network: List[object]) -> int:
        lb = 0
        for comp in task_network:
            # Primitive operator: must be executed; count its cost
            if hasattr(comp, 'cost'):
                try:
                    lb += int(comp.cost)
                except Exception:
                    lb += 1
                continue

            # Abstract task: cheap pattern-based LB
            name = getattr(comp, 'name', '')
            tpred, targs = self._parse_task_name(name)
            if not tpred:
                continue
            tpred_l = tpred.lower()

            if tpred_l == 'get_soil_data':
                lb += 2  # sample + communicate
            elif tpred_l == 'get_rock_data':
                lb += 2  # sample + communicate
            elif tpred_l == 'get_image_data':
                lb += 3  # calibrate + take + communicate
            elif tpred_l in ('send_soil_data', 'send_rock_data', 'send_image_data'):
                lb += 1
            elif tpred_l == 'do_calibrate':
                lb += 1
            elif tpred_l in ('do_navigate1', 'do_navigate2'):
                # Attempt a 0/1 bound using current at[rover, to]
                # do_navigate1(?x, ?to); do_navigate2(?x, ?from, ?to)
                rover = None
                to_wp = None
                if len(targs) >= 2:
                    rover = targs[0]
                    to_wp = targs[-1]  # last arg is 'to' for both variants
                if rover is not None and to_wp is not None:
                    at_id = self.at_fact_ids.get((rover, to_wp))
                    if at_id is not None and self._is_true(state, at_id):
                        lb += 0
                    else:
                        lb += 1  # at least one navigate action needed
                else:
                    # If we cannot parse, remain optimistic
                    lb += 0
            elif tpred_l == 'empty_store':
                lb += 0  # optimistic (drop may or may not be required)
            else:
                # Unknown abstract task; stay optimistic
                lb += 0
        return lb

    @staticmethod
    def _is_true(state: int, fact_gid: int) -> bool:
        return (state >> fact_gid) & 1 == 1

    @staticmethod
    def _any_true(state: int, fact_ids: List[int]) -> bool:
        for gid in fact_ids:
            if (state >> gid) & 1 == 1:
                return True
        return False

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names in the form +predicate[arg1,arg2,...]
        Returns (predicate, [args]) or (None, []).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        b = s.find('[')
        if b > 0 and s.endswith(']'):
            pred = s[:b]
            args_str = s[b + 1:-1]
            if args_str:
                return pred, [a.strip() for a in args_str.split(',')]
            return pred, []
        return None, []

    @staticmethod
    def _parse_task_name(name: str) -> Tuple[str, List[str]]:
        """
        Parse task/operator name strings of form taskName(arg1,arg2,...) used for Model components.
        If parentheses are absent, returns (name, []).
        """
        if not isinstance(name, str):
            return "", []
        s = name.strip()
        l = s.find('(')
        r = s.rfind(')')
        if l != -1 and r != -1 and r > l:
            head = s[:l]
            args_str = s[l + 1:r].strip()
            if args_str == "":
                return head, []
            return head, [a.strip() for a in args_str.split(',')]
        return s, []
