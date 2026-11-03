from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# NOTE: Pytrich provides Heuristic base in its environment
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimalist base to avoid import errors if inspected standalone.
    class Heuristic:  # type: ignore
        def __init__(self, model, *args, **kwargs):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network=None) -> int:
            return 0
        def __call__(self, state_bitwise: int, task_network=None) -> int:
            return self.evaluate(state_bitwise, task_network)


class RoverGPT5Heuristic3(Heuristic):
    """
    Domain-dependent heuristic for the Rover HDDL domain.

    Key ideas (admissible lower-bound landmarks, no delete accounting, no navigation accounted):
    - For each goal communicated_soil_data[p]:
        * If already true => 0
        * Else if any have_soil_analysis[x,p] is true => at least one communicate action remains => +1
        * Else => need to (sample_soil + communicate) ignoring navigation => +2
    - For each goal communicated_rock_data[p]: analogous to soil => +1 if have_rock_analysis exists, else +2
    - For each goal communicated_image_data[o,m]:
        * If already true => 0
        * Else if any have_image[x,o,m] is true => at least one communicate action => +1
        * Else if any calibrated[c,x] currently true for a camera c that supports m => need (take_image + communicate) => +2
        * Else => need (calibrate + take_image + communicate) => +3

    Rationale:
    - We ignore navigation and visibility constraints to keep a safe (admissible) lower-bound.
    - We leverage domain structure (mandatory sampling/communication; imaging requires calibration before shooting) to count mandatory remaining primitive actions.
    - We use bitwise checks only for state membership to remain efficient on large problems.

    This heuristic is consistent and inexpensive to compute:
    - All checks are O(#goals + #calibrated-facts) via precomputed indices.
    - No string-contains checks on state; only bitwise membership via fact.global_id.

    Parameter naming note:
    - The __init__ parameter name rover_gpt5_heuristic_3 is present to satisfy the task requirements; it is not otherwise used.
    """

    def __init__(self, model, rover_gpt5_heuristic_3: bool = True):
        super().__init__(model)
        self.model = model

        # Predicate -> {(args...): Fact}
        self._pred_arg_to_fact: Dict[str, Dict[Tuple[str, ...], object]] = defaultdict(dict)

        # Fast access to all Facts by predicate
        self._facts_by_pred: Dict[str, List[object]] = defaultdict(list)

        # Build indices over all grounded facts
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = tuple(a.strip() for a in args)
            self._pred_arg_to_fact[pred][key] = f
            self._facts_by_pred[pred].append(f)

        # Extract and index goal facts once
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._soil_goal_entries: List[Dict] = []            # Each: { 'comm_id': int, 'have_ids': List[int] }
        self._rock_goal_entries: List[Dict] = []            # Each: { 'comm_id': int, 'have_ids': List[int] }
        self._image_goal_entries: List[Dict] = []           # Each: { 'comm_id': int, 'have_ids': List[int], 'mode': str }

        # Precompute for soil/rock: mapping waypoint -> [have_*_analysis fact ids]
        self._have_soil_by_wp: Dict[str, List[int]] = defaultdict(list)  # p -> list of fact ids
        self._have_rock_by_wp: Dict[str, List[int]] = defaultdict(list)  # p -> list of fact ids
        for f in self._facts_by_pred.get('have_soil_analysis', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                # (x, p)
                p = args[1].strip()
                self._have_soil_by_wp[p].append(f.global_id)
        for f in self._facts_by_pred.get('have_rock_analysis', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                p = args[1].strip()
                self._have_rock_by_wp[p].append(f.global_id)

        # Precompute for imaging: have_image by (o,m)
        self._have_image_by_om: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        for f in self._facts_by_pred.get('have_image', []):
            _, args = self._parse_fact_name(f.name)
            # have_image ?x ?o ?m
            if len(args) == 3:
                o = args[1].strip()
                m = args[2].strip()
                self._have_image_by_om[(o, m)].append(f.global_id)

        # supports camera by mode: m -> set(cameras)
        self._cameras_by_mode: Dict[str, Set[str]] = defaultdict(set)
        for f in self._facts_by_pred.get('supports', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                c = args[0].strip()
                m = args[1].strip()
                self._cameras_by_mode[m].add(c)

        # calibrated facts by camera: c -> list of global_ids of calibrated[c,x], for all rovers x
        self._calibrated_ids_by_camera: Dict[str, List[int]] = defaultdict(list)
        for f in self._facts_by_pred.get('calibrated', []):
            _, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                c = args[0].strip()
                self._calibrated_ids_by_camera[c].append(f.global_id)

        # Precompute: for each mode m, collect all calibrated fact ids whose camera supports m
        self._calibrated_ids_by_mode: Dict[str, List[int]] = defaultdict(list)
        for m, cameras in self._cameras_by_mode.items():
            ids: List[int] = []
            for c in cameras:
                ids.extend(self._calibrated_ids_by_camera.get(c, []))
            self._calibrated_ids_by_mode[m] = ids

        # Build per-goal entries for fast evaluation
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            key = tuple(a.strip() for a in args)

            if pred == 'communicated_soil_data' and len(key) == 1:
                p = key[0]
                comm_fact = self._pred_arg_to_fact[pred].get((p,))
                if comm_fact is None:
                    continue
                have_ids = self._have_soil_by_wp.get(p, [])
                self._soil_goal_entries.append({
                    'comm_id': comm_fact.global_id,
                    'have_ids': have_ids,
                })

            elif pred == 'communicated_rock_data' and len(key) == 1:
                p = key[0]
                comm_fact = self._pred_arg_to_fact[pred].get((p,))
                if comm_fact is None:
                    continue
                have_ids = self._have_rock_by_wp.get(p, [])
                self._rock_goal_entries.append({
                    'comm_id': comm_fact.global_id,
                    'have_ids': have_ids,
                })

            elif pred == 'communicated_image_data' and len(key) == 2:
                o, m = key[0], key[1]
                comm_fact = self._pred_arg_to_fact[pred].get((o, m))
                if comm_fact is None:
                    continue
                have_ids = self._have_image_by_om.get((o, m), [])
                self._image_goal_entries.append({
                    'comm_id': comm_fact.global_id,
                    'have_ids': have_ids,
                    'mode': m,
                })
            else:
                # Ignore any other potential goals (not expected in Rover domain spec)
                pass

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """Parse grounded fact name in +predicate[arg1,arg2] format into (predicate, [args])."""
        if not name:
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return predicate, args
        return None, []

    @staticmethod
    def _bit_true(state_bitwise: int, gid: int) -> bool:
        return (state_bitwise >> gid) & 1 == 1

    @staticmethod
    def _any_true(state_bitwise: int, gids: List[int]) -> bool:
        for gid in gids:
            if (state_bitwise >> gid) & 1 == 1:
                return True
        return False

    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        h = 0

        # Soil goals
        for entry in self._soil_goal_entries:
            comm_id = entry['comm_id']
            if self._bit_true(state_bitwise, comm_id):
                continue  # already satisfied
            # If any have_soil_analysis[x,p] is already true => only 1 send remains
            if entry['have_ids'] and self._any_true(state_bitwise, entry['have_ids']):
                h += 1
            else:
                # Need at least (sample + send) ignoring navigation => 2
                h += 2

        # Rock goals
        for entry in self._rock_goal_entries:
            comm_id = entry['comm_id']
            if self._bit_true(state_bitwise, comm_id):
                continue
            if entry['have_ids'] and self._any_true(state_bitwise, entry['have_ids']):
                h += 1
            else:
                h += 2

        # Image goals
        for entry in self._image_goal_entries:
            comm_id = entry['comm_id']
            if self._bit_true(state_bitwise, comm_id):
                continue
            have_ids = entry['have_ids']
            mode = entry['mode']
            if have_ids and self._any_true(state_bitwise, have_ids):
                # Have image already, need to communicate
                h += 1
            else:
                # If any camera calibrated that supports mode => need (take + communicate)
                calib_ids = self._calibrated_ids_by_mode.get(mode, [])
                if calib_ids and self._any_true(state_bitwise, calib_ids):
                    h += 2
                else:
                    # Need (calibrate + take + communicate)
                    h += 3

        return h

    def __call__(self, state_bitwise: int, task_network=None) -> int:
        return self.evaluate(state_bitwise, task_network)
