from typing import Dict, List, Tuple, Set, Optional, Any

try:
    # Heuristic base is provided by the Pytrich framework
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub in case of offline checking; framework provides the real class at runtime
    class Heuristic:
        def __init__(self, model=None):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network: List[Any]):
            return 0


class RoverGPT5Heuristic14(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain (Pytrich Planner).

    Design rationale (lower-bound, consistent guidance):
    - We compute a goal-based relaxed cost that ignores movement, channel contention, and
      other delete effects beyond what the domain minimally imposes, yielding an admissible
      lower bound:
        * communicated_soil_data(p):
            - 0 if already true
            - 1 if any have_soil_analysis(*, p) is true (just one communication needed)
            - 2 otherwise (at least sample_soil + communicate)
        * communicated_rock_data(p): analogous to soil
        * communicated_image_data(o, m):
            - 0 if already true
            - 1 if any have_image(*, o, m) is true (just one communication needed)
            - 2 if there exists a suitable (camera c on rover x) already calibrated for o
                (i.e., calibrated(c, x) true) so we just need take_image + communicate
            - 3 otherwise (calibrate + take_image + communicate)
      This leverages domain structure: images require calibration, taking, and communication;
      soil/rock require sampling (with empty store) and communication. Navigation is hard to
      price cheaply and remains ignored to keep the bound admissible and cheap to compute.

    - We also compute a task-network-based cost that accounts for the remaining network:
        * Count 1 for each primitive operator still in the task network (must be executed).
        * For abstract tasks, add a minimal domain-based lower bound ignoring navigation:
            - get_soil_data: 2 (sample + communicate)
            - get_rock_data: 2 (sample + communicate)
            - get_image_data: 3 (calibrate + take_image + communicate)
            - send_*: 1 (single communication action)
            - do_calibrate: 1 (a single calibrate action in the best case)
            - empty_store: 0 (best case already empty -> NOP)
            - do_navigate*: 0 (can decompose to NOP if already at destination)
          For any other/unknown abstract task names: 0 (safe lower bound).

      This part provides guidance even when there are no explicit goal facts left but the
      HTN is not yet refined to primitives. Because goals and task-network can overlap in
      their contributions (e.g., a get_* task eventually achieves a communicated_* goal),
      we combine both estimates using max(h_goals, h_tn) to avoid double counting. The
      final heuristic remains admissible and tends to be informative on both small and large
      instances, while remaining very fast using bitwise checks and pre-indexed facts.

    Implementation notes:
    - We pre-index relevant facts by predicate/arguments for O(1) checks during evaluate():
        communicated_* (by arg), have_* (grouped to check existence), have_image (by (o, m)),
        calibrated (by (c, x)), and static mappings (on_board, supports, calibration_target)
        to quickly identify valid (camera, rover) candidates per (objective, mode).
    - We only use fast bitwise membership checks ((state >> fact_id) & 1) for performance.
    - We do not attempt to estimate navigation steps, store drops, or channel contention to
      keep the heuristic admissible, simple, and fast.
    """

    def __init__(self, model, rover_gpt5_heuristic_14: bool = True):
        try:
            super().__init__(model)
        except Exception:
            self.model = model

        # Fast access
        self.facts = self.model.facts

        # Indexes for quick lookups
        self.comm_soil_by_wp: Dict[str, int] = {}               # wp -> fact_id for +communicated_soil_data[wp]
        self.comm_rock_by_wp: Dict[str, int] = {}               # wp -> fact_id for +communicated_rock_data[wp]
        self.comm_image_by_om: Dict[Tuple[str, str], int] = {}  # (objective, mode) -> fact_id for +communicated_image_data[obj,mode]

        self.have_soil_by_wp: Dict[str, List[int]] = {}         # wp -> [fact_id of +have_soil_analysis[*,wp]]
        self.have_rock_by_wp: Dict[str, List[int]] = {}         # wp -> [fact_id of +have_rock_analysis[*,wp]]
        self.have_image_by_om: Dict[Tuple[str, str], List[int]] = {}  # (objective, mode) -> [fact_id of +have_image[*,obj,mode]]

        self.calibrated_id: Dict[Tuple[str, str], int] = {}     # (camera, rover) -> fact_id for +calibrated[camera,rover]
        self.on_board: Dict[str, str] = {}                      # camera -> rover
        self.supports: Dict[str, Set[str]] = {}                 # camera -> set(modes)
        self.calibration_target: Dict[str, str] = {}            # camera -> objective

        # Candidates per (objective, mode): list of (camera, rover, calibrated_fact_id or None)
        self.candidates_by_om: Dict[Tuple[str, str], List[Tuple[str, str, Optional[int]]]] = {}

        # Optional: for task-network checks
        self.empty_store_fact: Dict[str, int] = {}              # store -> fact_id for +empty[store]
        self.at_fact: Dict[Tuple[str, str], int] = {}           # (rover, waypoint) -> fact_id for +at[rover,waypoint]

        # Build all indexes from the model facts
        self._build_indexes()

        # Precompute goal facts as a compact list of (pred, args, fact_id)
        self.goal_facts: List[Tuple[str, Tuple[str, ...], int]] = []
        goals_bits = self.model.goals
        bit_pos = 0
        while goals_bits:
            if goals_bits & 1:
                # Fact with this global_id is a goal fact
                fact = self.facts[bit_pos]
                pred, args = self._parse_fact_name(fact.name)
                if pred is not None:
                    self.goal_facts.append((pred, tuple(args), fact.global_id))
            goals_bits >>= 1
            bit_pos += 1

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """Parse grounded fact name: +predicate[arg1,arg2] or -predicate[arg1,arg2]."""
        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s2 = s[1:]
            b = s2.find("[")
            if b > 0 and s2.endswith("]"):
                predicate = s2[:b]
                args_str = s2[b + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _build_indexes(self):
        # First pass: index dynamic and static predicates we care about
        for fact in self.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == 'communicated_soil_data' and len(args) == 1:
                self.comm_soil_by_wp[args[0]] = fact.global_id
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.comm_rock_by_wp[args[0]] = fact.global_id
            elif pred == 'communicated_image_data' and len(args) == 2:
                self.comm_image_by_om[(args[0], args[1])] = fact.global_id

            elif pred == 'have_soil_analysis' and len(args) == 2:
                wp = args[1]
                self.have_soil_by_wp.setdefault(wp, []).append(fact.global_id)
            elif pred == 'have_rock_analysis' and len(args) == 2:
                wp = args[1]
                self.have_rock_by_wp.setdefault(wp, []).append(fact.global_id)
            elif pred == 'have_image' and len(args) == 3:
                obj, mode = args[1], args[2]
                self.have_image_by_om.setdefault((obj, mode), []).append(fact.global_id)

            elif pred == 'calibrated' and len(args) == 2:
                cam, rov = args[0], args[1]
                self.calibrated_id[(cam, rov)] = fact.global_id
            elif pred == 'on_board' and len(args) == 2:
                cam, rov = args[0], args[1]
                self.on_board[cam] = rov
            elif pred == 'supports' and len(args) == 2:
                cam, mode = args[0], args[1]
                self.supports.setdefault(cam, set()).add(mode)
            elif pred == 'calibration_target' and len(args) == 2:
                cam, obj = args[0], args[1]
                self.calibration_target[cam] = obj

            elif pred == 'empty' and len(args) == 1:
                self.empty_store_fact[args[0]] = fact.global_id
            elif pred == 'at' and len(args) == 2:
                self.at_fact[(args[0], args[1])] = fact.global_id

        # Second pass: build candidate (camera, rover) pairs per (objective, mode)
        # Only cameras that have on_board rover, a calibration_target matching objective,
        # and support the mode are usable.
        for cam, rov in self.on_board.items():
            if cam not in self.calibration_target:
                continue
            obj = self.calibration_target[cam]
            cam_modes = self.supports.get(cam, set())
            for mode in cam_modes:
                key = (obj, mode)
                cal_fid = self.calibrated_id.get((cam, rov))
                self.candidates_by_om.setdefault(key, []).append((cam, rov, cal_fid))

    def _is_true(self, state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) != 0

    def _goal_cost_soil(self, state_bitwise: int, wp: str) -> int:
        # 0 if already communicated
        fact_id = self.comm_soil_by_wp.get(wp)
        if self._is_true(state_bitwise, fact_id):
            return 0
        # 1 if analysis already available (just communicate)
        for fid in self.have_soil_by_wp.get(wp, []):
            if self._is_true(state_bitwise, fid):
                return 1
        # Else need at least sample + communicate
        return 2

    def _goal_cost_rock(self, state_bitwise: int, wp: str) -> int:
        fact_id = self.comm_rock_by_wp.get(wp)
        if self._is_true(state_bitwise, fact_id):
            return 0
        for fid in self.have_rock_by_wp.get(wp, []):
            if self._is_true(state_bitwise, fid):
                return 1
        return 2

    def _goal_cost_image(self, state_bitwise: int, obj: str, mode: str) -> int:
        fact_id = self.comm_image_by_om.get((obj, mode))
        if self._is_true(state_bitwise, fact_id):
            return 0
        # Already have image: only communication needed
        for fid in self.have_image_by_om.get((obj, mode), []):
            if self._is_true(state_bitwise, fid):
                return 1
        # Check if a suitable (camera, rover) is already calibrated
        for cam, rov, cal_fid in self.candidates_by_om.get((obj, mode), []):
            if self._is_true(state_bitwise, cal_fid):
                # take_image + communicate
                return 2
        # Else, must (calibrate + take_image + communicate)
        return 3

    def _estimate_task_network_cost(self, state_bitwise: int, task_network: List[Any]) -> int:
        if not task_network:
            return 0
        cost = 0
        for t in task_network:
            # Primitive operator still in the network -> must cost at least 1
            if hasattr(t, 'apply') and hasattr(t, 'cost'):
                cost += 1
                continue
            # Abstract task: use name-based lower bound
            name = getattr(t, 'name', '')
            lname = name.lower()
            # Domain-specific lower bounds (ignore navigation to stay admissible)
            if 'get_soil_data' in lname:
                cost += 2
            elif 'get_rock_data' in lname:
                cost += 2
            elif 'get_image_data' in lname:
                cost += 3
            elif 'send_image_data' in lname:
                cost += 1
            elif 'send_soil_data' in lname:
                cost += 1
            elif 'send_rock_data' in lname:
                cost += 1
            elif 'do_calibrate' in lname:
                # Best case: calibrate is one action; could already be satisfied but staying at 1
                # here gives guidance for unresolved calibration tasks. To keep stricter
                # admissibility, this could be 0; however, 1 still remains a lower bound in the HTN
                # setting since do_calibrate typically requires at least one calibrate action
                # (unless a NOP method is applicable). To be maximally conservative, set to 1.
                cost += 1
            elif 'empty_store' in lname:
                # Could be NOP if store is already empty
                cost += 0
            elif 'do_navigate' in lname:
                # Could be NOP if already at destination
                cost += 0
            else:
                # Unknown abstract task; safest admissible lower bound is 0
                cost += 0
        return cost

    def evaluate(self, state_bitwise: int, task_network: List[Any]) -> int:
        # 1) Goal-based relaxed estimate
        h_goals = 0
        for pred, args, _fid in self.goal_facts:
            if pred == 'communicated_soil_data' and len(args) == 1:
                h_goals += self._goal_cost_soil(state_bitwise, args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1:
                h_goals += self._goal_cost_rock(state_bitwise, args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                h_goals += self._goal_cost_image(state_bitwise, args[0], args[1])
            else:
                # If other goal facts appear (not expected in Rover), they are already
                # satisfied by design or require domain-specific rules; default to 0.
                h_goals += 0

        # 2) Task-network-based estimate
        h_tn = self._estimate_task_network_cost(state_bitwise, task_network or [])

        # Combine without double counting overlap: lower bound is max of both
        return max(h_goals, h_tn)


# Factory alias for Pytrich configuration loaders (optional convenience)
def rover_gpt5_heuristic_14(model):
    return RoverGPT5Heuristic14(model)
