from typing import Dict, List, Tuple, Set, Optional

# Fallback base class if the framework import path differs
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:  # pragma: no cover
    class Heuristic:  # type: ignore
        pass


class RoverGPT5Heuristic17(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (lower-bound, domain-aware, fast):
    - We estimate the remaining cost to achieve the explicit goal facts in the Rover domain.
    - The domain goals are communicated_* facts for soil, rock, and image data. We ignore navigation
      distance (best-case assumption: already at a suitable place) to keep admissibility and speed.
    - For each pending goal fact we add the minimal number of primitive actions under optimistic
      assumptions about navigation/visibility and resource availability. This yields a safe lower bound.

    Cost model per pending goal fact (admissible lower bounds):
      - communicated_soil_data[p]:
          * If any have_soil_analysis[any_rover,p] holds in the current state: cost >= 1 (one communicate action)
          * Else: cost >= 2 (one sample + one communicate). We optimistically assume store is empty and visibility OK.
      - communicated_rock_data[p]: same scheme as soil.
      - communicated_image_data[o,m]:
          * If any have_image[any_rover,o,m] holds: cost >= 1 (one communicate action)
          * Else if there exists a calibrated[c,rover] already true for a camera/rover combo
            usable for (o,m): cost >= 2 (take_image + communicate)
          * Else if a usable camera/rover combo for (o,m) exists (based on static on_board/supports/calibration_target
            and rover equipped_for_imaging): cost >= 3 (calibrate + take_image + communicate)
          * Else: cost >= 0 (goals unsatisfiable under current static constraints; return 0 to keep admissibility)

    Implementation notes:
      - Uses bitwise checks for state membership to be fast (no string membership on states at runtime).
      - Pre-indexes goal-relevant facts (have_*_analysis, have_image, calibrated) and static relations
        (on_board, supports, calibration_target, equipped_for_imaging) once in __init__.
      - Iterates only over pending goal bits each call and sums minimal costs.

    This heuristic is consistent/admissible under the chosen cost model and reduces expansions by
    giving informative non-zero estimates once samples/images are available.
    """

    def __init__(self, model, rover_gpt5_heuristic_17: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.name = "RoverGPT5Heuristic17"
        self.cfg = rover_gpt5_heuristic_17 or {}

        # Indexes
        self._fact_pred_args: Dict[int, Tuple[str, Tuple[str, ...]]] = {}
        self._have_soil_by_wp: Dict[str, List[int]] = {}
        self._have_rock_by_wp: Dict[str, List[int]] = {}
        self._have_image_by_om: Dict[Tuple[str, str], List[int]] = {}
        self._calibrated_fid_by_cx: Dict[Tuple[str, str], int] = {}

        # Static predicates from initial state
        self._on_board: Set[Tuple[str, str]] = set()           # (camera, rover)
        self._supports: Set[Tuple[str, str]] = set()            # (camera, mode)
        self._calib_target: Set[Tuple[str, str]] = set()        # (camera, objective)
        self._imaging_rovers: Set[str] = set()

        # Precomputation for images
        self._any_camera_for_om: Set[Tuple[str, str]] = set()   # (objective, mode) combos feasible statically
        self._calibrated_ids_for_om: Dict[Tuple[str, str], List[int]] = {}

        # Build all indexes once
        self._build_indexes()

    # -------------------- Core Heuristic API --------------------
    def __call__(self, state_bitwise: int, task_network=None) -> int:
        # pending_bits: bits set in goals but not yet set in state
        pending_bits = self.model.goals & (~state_bitwise)
        if pending_bits == 0:
            # All goal facts reached; we return 0 to keep admissibility even if tasks remain
            return 0

        h = 0
        for bit in self._iterate_bits(pending_bits):
            fact = self.model.get_component(bit)  # Fact
            pred, args = self._fact_pred_args[fact.global_id]

            # communicated_soil_data[wp]
            if pred == "communicated_soil_data":
                wp = args[0]
                if self._any_fact_true(state_bitwise, self._have_soil_by_wp.get(wp, [])):
                    h += 1  # just communicate
                else:
                    h += 2  # sample + communicate (optimistic nav and empty store)

            # communicated_rock_data[wp]
            elif pred == "communicated_rock_data":
                wp = args[0]
                if self._any_fact_true(state_bitwise, self._have_rock_by_wp.get(wp, [])):
                    h += 1
                else:
                    h += 2

            # communicated_image_data[obj, mode]
            elif pred == "communicated_image_data":
                obj, mode = args[0], args[1]
                # If image already taken by any rover
                if self._any_fact_true(state_bitwise, self._have_image_by_om.get((obj, mode), [])):
                    h += 1  # only communication remains
                else:
                    # If currently calibrated for a usable (camera, rover) combo for (obj, mode)
                    calib_ids = self._calibrated_ids_for_om.get((obj, mode), [])
                    if self._any_fact_true(state_bitwise, calib_ids):
                        h += 2  # take_image + communicate
                    else:
                        # If any feasible camera exists statically for (obj, mode)
                        if (obj, mode) in self._any_camera_for_om:
                            h += 3  # calibrate + take_image + communicate (optimistic nav)
                        else:
                            # No feasible camera known; return 0 to keep admissibility (dead-end or static mismatch)
                            h += 0
            else:
                # Any other goal fact type (not expected in Rover problems) -> contribute 0 to keep admissibility
                h += 0

        return h

    # -------------------- Precomputation --------------------
    def _build_indexes(self):
        # Initial state (for static predicates)
        init = self.model.initial_state

        # 1) Parse and classify all FACTS once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            self._fact_pred_args[f.global_id] = (pred, tuple(args))

        # 2) Collect static facts from initial state
        for f in self.model.facts:
            gid = f.global_id
            if not self._is_true(init, gid):
                continue
            pred, args = self._fact_pred_args[gid]
            if pred == "on_board":
                # (camera, rover)
                if len(args) == 2:
                    self._on_board.add((args[0], args[1]))
            elif pred == "supports":
                # (camera, mode)
                if len(args) == 2:
                    self._supports.add((args[0], args[1]))
            elif pred == "calibration_target":
                # (camera, objective)
                if len(args) == 2:
                    self._calib_target.add((args[0], args[1]))
            elif pred == "equipped_for_imaging":
                # (rover)
                if len(args) == 1:
                    self._imaging_rovers.add(args[0])

        # 3) Index dynamic goal-relevant facts across the full fact universe
        for f in self.model.facts:
            pred, args = self._fact_pred_args[f.global_id]
            if pred == "have_soil_analysis" and len(args) == 2:
                # (rover, waypoint)
                wp = args[1]
                self._have_soil_by_wp.setdefault(wp, []).append(f.global_id)
            elif pred == "have_rock_analysis" and len(args) == 2:
                # (rover, waypoint)
                wp = args[1]
                self._have_rock_by_wp.setdefault(wp, []).append(f.global_id)
            elif pred == "have_image" and len(args) == 3:
                # (rover, objective, mode) -> index by (objective, mode)
                om = (args[1], args[2])
                self._have_image_by_om.setdefault(om, []).append(f.global_id)
            elif pred == "calibrated" and len(args) == 2:
                # (camera, rover)
                self._calibrated_fid_by_cx[(args[0], args[1])] = f.global_id

        # 4) Precompute feasibility of (objective, mode) and the calibrated ids usable for each (o,m)
        for cam, rov in self._on_board:
            if rov not in self._imaging_rovers:
                continue
            # For this (camera, rover) pair, find all supported modes and target objectives
            supported_modes = [m for (c, m) in self._supports if c == cam]
            target_objs = [o for (c, o) in self._calib_target if c == cam]
            if not supported_modes or not target_objs:
                continue

            # Register feasibility
            for o in target_objs:
                for m in supported_modes:
                    self._any_camera_for_om.add((o, m))

            # Map calibrated facts for (o,m) via this (cam, rov) pair
            calib_gid = self._calibrated_fid_by_cx.get((cam, rov))
            if calib_gid is not None:
                for o in target_objs:
                    for m in supported_modes:
                        self._calibrated_ids_for_om.setdefault((o, m), []).append(calib_gid)

    # -------------------- Bitwise utilities --------------------
    @staticmethod
    def _iterate_bits(bitset: int):
        """Yield set bit positions in bitset."""
        i = 0
        b = bitset
        while b:
            if b & 1:
                yield i
            b >>= 1
            i += 1

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _any_fact_true(self, state_bitwise: int, fact_ids: List[int]) -> bool:
        for fid in fact_ids:
            if self._is_true(state_bitwise, fid):
                return True
        return False

    # -------------------- Name parsing --------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        """
        Parse grounded element name of the form "+predicate[arg1,arg2]" or "-predicate[arg1,arg2]".
        Returns (predicate, (args...)) or (None, ()).
        """
        n = name.strip()
        if n.startswith("+") or n.startswith("-"):
            n = n[1:]
        bracket_idx = n.find("[")
        if bracket_idx > 0 and n.endswith("]"):
            predicate = n[:bracket_idx]
            args_str = n[bracket_idx + 1:-1]
            args = tuple(a.strip() for a in args_str.split(",")) if args_str else tuple()
            return predicate, args
        return None, tuple()
