from typing import Dict, List, Tuple, Set, Any

# Attempt to import Heuristic from common locations used by Pytrich
try:
    from Pytrich.Heuristics.Heuristic import Heuristic  # type: ignore
except Exception:  # pragma: no cover
    try:
        from Pytrich.Heuristics.heuristic import Heuristic  # type: ignore
    except Exception:  # pragma: no cover
        # Minimal fallback to allow static analysis if the real Heuristic isn't importable in this environment
        class Heuristic:  # type: ignore
            def __init__(self, model):
                self.model = model


class RoverGPT5Heuristic6(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (admissible lower bound via domain relaxation):
    - Goals in Rover are communicated_* facts (soil/rock/image). Achieving each requires a small pipeline:
      * Soil: sample_soil (1) + communicate_soil (1) => 2 actions if not yet sampled; if already have_soil_analysis, only 1 (communicate).
      * Rock: sample_rock (1) + communicate_rock (1) => 2 actions if not yet sampled; if already have_rock_analysis, only 1 (communicate).
      * Image: calibrate (1) + take_image (1) + communicate_image (1) => 3 actions if not yet have_image; if already have_image, only 1 (communicate).
    - This ignores navigation, store capacity and visibility constraints (delete-relaxation), which yields a safe admissible lower bound.

    Hierarchical (HTN) awareness:
    - In addition to goal-based lower bound, we compute a second lower bound from the current task network (TN):
      * For each primitive operator remaining in TN: minimal cost >= 1 (must be executed).
      * For each abstract task we use the cheapest possible decomposition under the same relaxation:
        - get_soil_data: 2 (sample + communicate)
        - get_rock_data: 2 (sample + communicate)
        - get_image_data: 3 (calibrate + take_image + communicate)
        - send_*_data: 1 (communicate)
        - do_calibrate: 1 (calibrate)
        - do_navigate1/do_navigate2: 0 (can be a NOP if already satisfied)
        - empty_store: 0 (can be a NOP if already empty)
    - The final heuristic is the maximum of the goal-based and TN-based lower bounds, still admissible because the remaining cost must be at least both.

    Implementation notes:
    - We use bitwise checks for state membership: (state >> fact.global_id) & 1 (fast, as required).
    - We pre-index facts for O(1) lookup of communicated_* facts and partial achievements (have_* facts).
    - Robust handling of task network entries works both for component ids and direct objects via model.get_component.

    This heuristic is designed to be light-weight (fast) yet informative, and scales to large Rover instances.
    """

    def __init__(self, model, rover_gpt5_heuristic_6: bool = True):
        super().__init__(model)
        self.enabled = rover_gpt5_heuristic_6

        # Predicates we care about
        self.P_COMM_SOIL = 'communicated_soil_data'
        self.P_COMM_ROCK = 'communicated_rock_data'
        self.P_COMM_IMG = 'communicated_image_data'
        self.P_HAVE_SOIL = 'have_soil_analysis'
        self.P_HAVE_ROCK = 'have_rock_analysis'
        self.P_HAVE_IMG = 'have_image'

        # Index facts by (predicate, args)
        self.fact_by_pred_args: Dict[Tuple[str, Tuple[str, ...]], Any] = {}

        # Index partial achievements for quick membership tests
        self.have_soil_by_wp: Dict[str, List[Any]] = {}
        self.have_rock_by_wp: Dict[str, List[Any]] = {}
        self.have_img_by_om: Dict[Tuple[str, str], List[Any]] = {}

        # Build indices over all facts
        self._build_fact_indices()

        # Parse and cache goal structure (which communicated_* are actually in the problem goals)
        self.goal_comm_soil_facts: Dict[str, Any] = {}   # waypoint -> Fact
        self.goal_comm_rock_facts: Dict[str, Any] = {}   # waypoint -> Fact
        self.goal_comm_img_facts: Dict[Tuple[str, str], Any] = {}  # (objective, mode) -> Fact
        self._extract_goal_facts()

        # Minimal cost estimates per abstract task (by substring match on task name)
        # These are conservative lower bounds under the same relaxation used above.
        self.task_min_cost: Dict[str, int] = {
            'get_soil_data': 2,
            'get_rock_data': 2,
            'get_image_data': 3,
            'send_soil_data': 1,
            'send_rock_data': 1,
            'send_image_data': 1,
            'do_calibrate': 1,
            'do_navigate1': 0,
            'do_navigate2': 0,
            'empty_store': 0,
        }

    # ---------- Public API ----------
    def __call__(self, state_bitwise: int, task_network: List[Any]) -> int:
        if not self.enabled:
            return 0
        return self._heuristic(state_bitwise, task_network)

    # Backward compatibility for some planners
    def h(self, state_bitwise: int, task_network: List[Any]) -> int:
        return self.__call__(state_bitwise, task_network)

    # ---------- Core Heuristic ----------
    def _heuristic(self, state_bitwise: int, task_network: List[Any]) -> int:
        # 1) Goal-based relaxed lower bound
        goal_lb = self._goal_lower_bound(state_bitwise)

        # 2) Task-network-based relaxed lower bound
        tn_lb = self._task_network_lower_bound(task_network)

        # Admissible combination: take max of independent LBs
        return max(goal_lb, tn_lb)

    # ---------- Goal-based LB ----------
    def _goal_lower_bound(self, state_bitwise: int) -> int:
        lb = 0

        # Soil goals
        for wp, comm_fact in self.goal_comm_soil_facts.items():
            if self._is_true(state_bitwise, comm_fact):
                continue
            # If data already sampled by any rover, only 1 (communicate)
            if any(self._is_true(state_bitwise, f) for f in self.have_soil_by_wp.get(wp, [])):
                lb += 1
            else:
                # Need sample + communicate (navigation ignored): 2
                lb += 2

        # Rock goals
        for wp, comm_fact in self.goal_comm_rock_facts.items():
            if self._is_true(state_bitwise, comm_fact):
                continue
            if any(self._is_true(state_bitwise, f) for f in self.have_rock_by_wp.get(wp, [])):
                lb += 1
            else:
                lb += 2

        # Image goals
        for (obj, mode), comm_fact in self.goal_comm_img_facts.items():
            if self._is_true(state_bitwise, comm_fact):
                continue
            if any(self._is_true(state_bitwise, f) for f in self.have_img_by_om.get((obj, mode), [])):
                lb += 1  # only communicate left
            else:
                lb += 3  # calibrate + take_image + communicate

        return lb

    # ---------- TN-based LB ----------
    def _task_network_lower_bound(self, task_network: List[Any]) -> int:
        if not task_network:
            return 0
        total = 0

        for t in task_network:
            comp = t
            # Task entries may be component IDs or direct objects
            if isinstance(t, int):
                comp = self.model.get_component(t)

            # Distinguish by global id range if possible (operators vs abstract tasks)
            try:
                gid = comp.global_id  # should exist for all components
            except Exception:
                # Fallback: use name-based matching if no id
                name = getattr(comp, 'name', '')
                total += self._min_cost_for_task_name(name)
                continue

            if self.model.iop_init <= gid <= self.model.iop_end:
                # Primitive operator remaining: at least 1 step
                total += 1
                continue

            if self.model.iabt_init <= gid <= self.model.iabt_end:
                # Abstract task: use cheap mapping by name
                name = getattr(comp, 'name', '')
                total += self._min_cost_for_task_name(name)
                continue

            # Decompositions are not tasks to execute; ignore

        return total

    def _min_cost_for_task_name(self, name: str) -> int:
        if not name:
            return 0
        # Substring matching is robust across common namings
        for key, val in self.task_min_cost.items():
            if key in name:
                return val
        return 0

    # ---------- Indices & Utilities ----------
    def _build_fact_indices(self):
        # Build a mapping from (predicate, args) to Fact; and partial achievement indices
        for fact in self.model.facts:
            if not isinstance(getattr(fact, 'name', ''), str):
                continue
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self.fact_by_pred_args[key] = fact

            # Partial achievements
            if pred == self.P_HAVE_SOIL and len(args) == 2:
                wp = args[1]
                self.have_soil_by_wp.setdefault(wp, []).append(fact)
            elif pred == self.P_HAVE_ROCK and len(args) == 2:
                wp = args[1]
                self.have_rock_by_wp.setdefault(wp, []).append(fact)
            elif pred == self.P_HAVE_IMG and len(args) == 3:
                obj, mode = args[1], args[2]
                self.have_img_by_om.setdefault((obj, mode), []).append(fact)

    def _extract_goal_facts(self):
        # model.goals is bitwise; obtain explicit list once
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            if pred == self.P_COMM_SOIL and len(args) == 1:
                key = (pred, tuple(args))
                f = self.fact_by_pred_args.get(key, None)
                if f is not None:
                    self.goal_comm_soil_facts[args[0]] = f
            elif pred == self.P_COMM_ROCK and len(args) == 1:
                key = (pred, tuple(args))
                f = self.fact_by_pred_args.get(key, None)
                if f is not None:
                    self.goal_comm_rock_facts[args[0]] = f
            elif pred == self.P_COMM_IMG and len(args) == 2:
                key = (pred, tuple(args))
                f = self.fact_by_pred_args.get(key, None)
                if f is not None:
                    self.goal_comm_img_facts[(args[0], args[1])] = f

    @staticmethod
    def _parse_fact_name(name: str):
        """
        Parse facts of the form "+predicate[arg1,arg2,...]" as specified.
        Returns (predicate, [args]) or (None, []).
        """
        try:
            s = name.strip()
            if s.startswith('+') or s.startswith('-'):
                s = s[1:]
            bracket_idx = s.find('[')
            if bracket_idx > 0 and s.endswith(']'):
                predicate = s[:bracket_idx]
                args_str = s[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
            return None, []
        except Exception:  # pragma: no cover
            return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact) -> bool:
        gid = getattr(fact, 'global_id', None)
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1


# Factory function hook if the framework expects a creator
def rover_gpt5_heuristic_6(model):
    return RoverGPT5Heuristic6(model)
