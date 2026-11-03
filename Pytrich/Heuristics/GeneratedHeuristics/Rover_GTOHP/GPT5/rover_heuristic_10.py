from typing import Dict, List, Tuple, Set, Optional

try:
    # Expected location in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Lightweight fallback to ease static checking if framework import is unavailable at generation time
    class Heuristic:
        def __init__(self, model, *args, **kwargs):
            self.model = model
        def h(self, state_bitwise: int, task_network: List) -> int:
            return 0


class RoverGPT5Heuristic10(Heuristic):
    """
    Rover domain-dependent heuristic (admissible, fast) for Pytrich.

    Design rationale and domain insights
    ------------------------------------
    The Rover HDDL domain has three families of top-level goals in the provided training set:
      - communicated_soil_data[waypoint]
      - communicated_rock_data[waypoint]
      - communicated_image_data[objective,mode]

    For each goal instance, there are domain-mandatory primitive actions:
      - Soil: to achieve communicated_soil_data[p], one must (i) possess have_soil_analysis[x,p] for some rover x
               (achieved by a single sample_soil action per location) and (ii) perform exactly one communicate action.
               Therefore a safe (admissible) lower bound per pending soil goal is:
                 1 (communication) + (0 if any have_soil_analysis[* , p] is already true, else 1 for sample)
      - Rock: analogous to soil.
      - Image: to achieve communicated_image_data[o,m], one must (i) have the image have_image[x,o,m]
                (requires exactly one calibrate and one take_image for the chosen camera/rover; the action take_image
                deletes calibrated, so calibration cannot be re-used across multiple images), then (ii) communicate it.
                Therefore a safe (admissible) lower bound per pending image goal is:
                 - If any have_image[* , o, m] is already true: 1 (communication)
                 - Otherwise: 3 (1 calibration + 1 take_image + 1 communication)

    We purposely ignore navigation/drop costs in this heuristic to preserve admissibility across multiple goals, because
    navigation steps can often be shared among goals and summing them would risk overestimation. The counts above only
    add actions that are strictly goal-specific and cannot be shared across different goal instances.

    This yields a cheap, scalable, yet informative heuristic:
      h(state) = sum_over_pending_soil  [1 + (have_soil? 0 : 1)]
               + sum_over_pending_rock  [1 + (have_rock? 0 : 1)]
               + sum_over_pending_image [1 if have_image else 3]

    Implementation notes
    --------------------
    - Uses bitwise checks against fact.global_id for O(1) membership.
    - Precomputes indices from the model once in __init__.
    - Parses goal facts from model.goals using model.state_explicit_repr(self.model.goals), which returns strings like
      '+predicate[arg1,arg2]'.
    - Avoids calling model.state_explicit_repr(state) during heuristic evaluation (too slow on large instances).

    This heuristic is designed to be:
      - Domain-specific and informative
      - Admissible (never overestimates)
      - Fast to compute on very large instances
    """

    def __init__(self, model, rover_gpt5_heuristic_10: Optional[Dict] = None):
        super().__init__(model)

        # Internal index maps
        self._fact_name_to_id: Dict[str, int] = {}
        self._have_soil_by_wp: Dict[str, List[int]] = {}
        self._have_rock_by_wp: Dict[str, List[int]] = {}
        self._have_image_by_om: Dict[Tuple[str, str], List[int]] = {}

        # Goal targets (and their fact ids) extracted once from model.goals
        self._goal_comm_soil: Dict[str, int] = {}            # wp -> fact_id
        self._goal_comm_rock: Dict[str, int] = {}            # wp -> fact_id
        self._goal_comm_image: Dict[Tuple[str, str], int] = {}  # (o, m) -> fact_id

        # Build global fact-name -> id map and pre-index relevant facts
        self._build_fact_indexes()
        self._extract_goal_targets()

    # -------------------------------
    # Heuristic core
    # -------------------------------
    def h(self, state_bitwise: int, task_network: List) -> int:
        # If already solved, return 0 quickly
        if self.model.goal_reached(state_bitwise, task_network):
            return 0

        h_val = 0

        # Soil goals
        for wp, fid in self._goal_comm_soil.items():
            if not self._bit_true(state_bitwise, fid):
                # At least one communication action is required
                cost = 1
                # If have_soil_analysis for this waypoint is not yet true by any rover, add cost for one sample
                if not self._any_fact_true(state_bitwise, self._have_soil_by_wp.get(wp, [])):
                    cost += 1
                h_val += cost

        # Rock goals
        for wp, fid in self._goal_comm_rock.items():
            if not self._bit_true(state_bitwise, fid):
                # At least one communication action is required
                cost = 1
                # If have_rock_analysis for this waypoint is not yet true by any rover, add cost for one sample
                if not self._any_fact_true(state_bitwise, self._have_rock_by_wp.get(wp, [])):
                    cost += 1
                h_val += cost

        # Image goals
        for (obj, mode), fid in self._goal_comm_image.items():
            if not self._bit_true(state_bitwise, fid):
                # If we already have the image by some rover, only 1 communication is needed
                if self._any_fact_true(state_bitwise, self._have_image_by_om.get((obj, mode), [])):
                    h_val += 1
                else:
                    # Otherwise, we need at least: 1 calibrate + 1 take_image + 1 communicate = 3
                    h_val += 3

        return h_val

    # -------------------------------
    # Internal utilities
    # -------------------------------
    def _build_fact_indexes(self) -> None:
        """
        Build indices for fast bitwise membership testing at evaluation time.
        """
        for f in self.model.facts:
            self._fact_name_to_id[f.name] = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue

            # Group facts we will need to check quickly
            if pred == 'have_soil_analysis' and len(args) == 2:
                # have_soil_analysis[x, wp]
                wp = args[1]
                self._have_soil_by_wp.setdefault(wp, []).append(f.global_id)

            elif pred == 'have_rock_analysis' and len(args) == 2:
                # have_rock_analysis[x, wp]
                wp = args[1]
                self._have_rock_by_wp.setdefault(wp, []).append(f.global_id)

            elif pred == 'have_image' and len(args) == 3:
                # have_image[x, o, m]
                o, m = args[1], args[2]
                self._have_image_by_om.setdefault((o, m), []).append(f.global_id)

    def _extract_goal_targets(self) -> None:
        """
        Read model.goals (bitwise) as explicit names once and store their IDs and arguments by predicate type.
        """
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            fid = self._fact_name_to_id.get(name, None)
            if fid is None:
                # Should not happen; defensively skip if unknown
                continue

            if pred == 'communicated_soil_data' and len(args) == 1:
                self._goal_comm_soil[args[0]] = fid
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self._goal_comm_rock[args[0]] = fid
            elif pred == 'communicated_image_data' and len(args) == 2:
                self._goal_comm_image[(args[0], args[1])] = fid

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names in the Pytrich representation:
          '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'.
        Returns (predicate, [args...]) or (None, []).
        """
        n = name.strip()
        if n.startswith("+") or n.startswith("-"):
            n = n[1:]
            bracket_idx = n.find("[")
            if bracket_idx > 0 and n.endswith("]"):
                predicate = n[:bracket_idx]
                args_str = n[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _bit_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _any_fact_true(self, state_bitwise: int, fact_ids: List[int]) -> bool:
        for fid in fact_ids:
            if self._bit_true(state_bitwise, fid):
                return True
        return False


# Optional factory alias (some runners expect a callable to obtain the heuristic)
DefHeuristic = RoverGPT5Heuristic10
