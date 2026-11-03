from typing import Dict, List, Tuple, Set, Optional

# Try multiple import paths to match Pytrich layout
try:
    from Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic
    except Exception:
        # Fallback minimal base to avoid import errors in static checks
        class Heuristic:
            def __init__(self, model, *args, **kwargs):
                self.model = model
            def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
                return 0
            def __call__(self, state_bitwise: int, task_network: List = None) -> int:
                return self.evaluate(state_bitwise, task_network)


class RoverGPT5Heuristic7(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (admissible lower bound, efficient on large instances):
    - We compute an additive sum of unavoidable primitive actions per outstanding goal,
      intentionally ignoring navigation and other shareable actions. This keeps the
      heuristic admissible while remaining informative and fast to compute.

      For each outstanding goal type:
        Soil (communicated_soil_data(p)):
          - If already communicated: +0
          - Else if any rover already has have_soil_analysis(r, p): +1 (one communicate)
          - Else: +2 (one sample_soil + one communicate)

        Rock (communicated_rock_data(p)):
          - If already communicated: +0
          - Else if any rover already has have_rock_analysis(r, p): +1 (communicate)
          - Else: +2 (sample_rock + communicate)

        Image (communicated_image_data(o, m)):
          - If already communicated: +0
          - Else if any rover already has have_image(r, o, m): +1 (communicate)
          - Else:
              If any camera on-board some rover that supports mode m is currently calibrated:
                  +2 (take_image + communicate)
              Else:
                  +3 (calibrate + take_image + communicate)

      Notes:
      - Navigation actions (navigate) and store handling (drop) are ignored on purpose in the
        summed bound because they can be shared/overlapped across subgoals, and summing lower
        bounds on potentially shareable parts risks inadmissibility. Sampling/taking/comm are
        inherently per-goal and not shareable, so summation is safe and admissible.
      - This approach balances computation time (only bitwise checks) with accuracy (captures
        core per-goal action requirements), and scales to very large instances.

    Implementation details:
    - We precompute and cache bitmasks for:
        * For each waypoint p: any have_soil_analysis(r,p) facts (bitmask)
        * For each waypoint p: any have_rock_analysis(r,p) facts (bitmask)
        * For each (objective o, mode m): any have_image(r,o,m) facts (bitmask)
        * For each mode m: any calibrated(c, r) facts for cameras that support m and are on-board some rover (bitmask)
        * For each goal fact: the Fact object and its global_id for O(1) checks
    - At evaluation time we only perform fast bitwise checks.

    Admissibility:
    - We only sum unavoidable, per-goal primitive actions and never count navigation or other potentially shareable steps.
      Therefore, h(n) <= h*(n) for all n.
    """

    def __init__(self, model, rover_gpt5_heuristic_7: Optional[dict] = None):
        super().__init__(model)
        self._init_internal_caches()

    # ------------------------- Public API -------------------------
    def evaluate(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        h = 0

        # Soil goals
        for wp, comm_fact in self._soil_goal_comm_facts.items():
            if self._is_true(state_bitwise, comm_fact.global_id):
                continue  # already achieved
            have_mask = self._soil_have_masks.get(wp, 0)
            if state_bitwise & have_mask:
                # Have analysis already -> one communicate
                h += 1
            else:
                # Need to sample + communicate (navigation ignored)
                h += 2

        # Rock goals
        for wp, comm_fact in self._rock_goal_comm_facts.items():
            if self._is_true(state_bitwise, comm_fact.global_id):
                continue
            have_mask = self._rock_have_masks.get(wp, 0)
            if state_bitwise & have_mask:
                h += 1
            else:
                h += 2

        # Image goals
        for (obj, mode), comm_fact in self._image_goal_comm_facts.items():
            if self._is_true(state_bitwise, comm_fact.global_id):
                continue
            have_mask = self._image_have_masks.get((obj, mode), 0)
            if state_bitwise & have_mask:
                # Image already taken -> one communicate
                h += 1
            else:
                # If any camera supporting this mode is already calibrated, we can skip calibration
                cal_mask = self._calibrated_masks_by_mode.get(mode, 0)
                if cal_mask and (state_bitwise & cal_mask):
                    # take_image + communicate
                    h += 2
                else:
                    # calibrate + take_image + communicate
                    h += 3

        return h

    def __call__(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        return self.evaluate(state_bitwise, task_network)

    # ---------------------- Internal utilities ----------------------
    def _init_internal_caches(self):
        # Index facts by predicate and args for quick lookup
        self._facts_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self._predicate_index: Dict[str, List[object]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._facts_by_key[key] = f
            self._predicate_index.setdefault(pred, []).append(f)

        # Precompute supported cameras per mode and on-board relations
        self._supports: Set[Tuple[str, str]] = set()      # (camera, mode)
        self._on_board: Set[Tuple[str, str]] = set()      # (camera, rover)
        self._calibrated_facts: Dict[Tuple[str, str], object] = {}  # (camera, rover) -> Fact
        for f in self._predicate_index.get('supports', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self._supports.add((args[0], args[1]))
        for f in self._predicate_index.get('on_board', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self._on_board.add((args[0], args[1]))
        for f in self._predicate_index.get('calibrated', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self._calibrated_facts[(args[0], args[1])] = f

        # Build per-mode calibrated bitmasks: any (camera, rover) that supports the mode and is on-board
        self._calibrated_masks_by_mode: Dict[str, int] = {}
        for cam, m in self._supports:
            for cam2, r in self._on_board:
                if cam2 != cam:
                    continue
                f = self._calibrated_facts.get((cam, r), None)
                if f is None:
                    continue
                self._calibrated_masks_by_mode[m] = self._calibrated_masks_by_mode.get(m, 0) | (1 << f.global_id)

        # Precompute have_* bitmasks per goal argument for fast checks
        self._soil_have_masks: Dict[str, int] = {}  # waypoint -> bitmask of any have_soil_analysis(*, waypoint)
        for f in self._predicate_index.get('have_soil_analysis', []):
            _, args = self._parse_fact_name(f.name)  # (rover, waypoint)
            if len(args) == 2:
                wp = args[1]
                self._soil_have_masks[wp] = self._soil_have_masks.get(wp, 0) | (1 << f.global_id)

        self._rock_have_masks: Dict[str, int] = {}  # waypoint -> bitmask of any have_rock_analysis(*, waypoint)
        for f in self._predicate_index.get('have_rock_analysis', []):
            _, args = self._parse_fact_name(f.name)  # (rover, waypoint)
            if len(args) == 2:
                wp = args[1]
                self._rock_have_masks[wp] = self._rock_have_masks.get(wp, 0) | (1 << f.global_id)

        self._image_have_masks: Dict[Tuple[str, str], int] = {}  # (objective, mode) -> bitmask of any have_image(*, objective, mode)
        for f in self._predicate_index.get('have_image', []):
            _, args = self._parse_fact_name(f.name)  # (rover, objective, mode)
            if len(args) == 3:
                obj, mode = args[1], args[2]
                key = (obj, mode)
                self._image_have_masks[key] = self._image_have_masks.get(key, 0) | (1 << f.global_id)

        # Parse goal facts into typed groups for O(1) checking during evaluation
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._soil_goal_comm_facts: Dict[str, object] = {}   # waypoint -> Fact(communicated_soil_data(wp))
        self._rock_goal_comm_facts: Dict[str, object] = {}   # waypoint -> Fact(communicated_rock_data(wp))
        self._image_goal_comm_facts: Dict[Tuple[str, str], object] = {}  # (objective, mode) -> Fact

        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            fact = self._facts_by_key.get(key, None)
            if fact is None:
                # Defensive: skip if not found (shouldn't happen)
                continue
            if pred == 'communicated_soil_data' and len(args) == 1:
                self._soil_goal_comm_facts[args[0]] = fact
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self._rock_goal_comm_facts[args[0]] = fact
            elif pred == 'communicated_image_data' and len(args) == 2:
                self._image_goal_comm_facts[(args[0], args[1])] = fact
            else:
                # Other goal facts (if any) are ignored by this heuristic
                pass

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith('+') or name.startswith('-'):
            name = name[1:]
            bracket_idx = name.find('[')
            if bracket_idx > 0 and name.endswith(']'):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) != 0


# Alias for compatibility if the framework expects a factory function/class name
def rover_gpt5_heuristic_7(model, config: Optional[dict] = None):
    return RoverGPT5Heuristic7(model, rover_gpt5_heuristic_7=config)
