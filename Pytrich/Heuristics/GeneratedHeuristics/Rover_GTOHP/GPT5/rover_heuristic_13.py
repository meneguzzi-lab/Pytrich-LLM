from typing import Dict, List, Tuple, Set, Optional

# Try to import the Heuristic base from common Pytrich locations
try:
    from Pytrich.Heuristics.heuristic import Heuristic  # type: ignore
except Exception:
    try:
        from heuristic import Heuristic  # type: ignore
    except Exception:
        # Fallback stub (in case of isolated unit tests). The real framework provides Heuristic.
        class Heuristic:  # type: ignore
            def __init__(self, model):
                self.model = model


class RoverGPT5Heuristic13(Heuristic):
    """
    RoverGPT5Heuristic13: A domain-dependent HTN heuristic for the Rover domain.

    Design principles and domain insights:
    - Target goals are always of three kinds: communicated_soil_data[p], communicated_rock_data[p],
      communicated_image_data[o,m]. Achieving these requires the following minimum primitive steps
      (ignoring navigation, store handling, and channel contention for admissibility):
        * Soil goal @ waypoint p:
            - If communicated_soil_data[p] is already true: cost 0.
            - Else if have_soil_analysis[*,p] is true for some rover: only communication is needed: cost 1.
            - Else: need sampling + communication: cost 2.
        * Rock goal @ waypoint p:
            - Analogue of soil: 0 if already communicated; 1 if we already have rock analysis; otherwise 2.
        * Image goal @ (objective o, mode m):
            - If communicated_image_data[o,m] is already true: cost 0.
            - Else if have_image[*,o,m] is true: only send is needed: cost 1.
            - Else if a camera already calibrated (calibrated[c,x]) for objective o, and supports[c,m] holds:
                take_image + send are needed: cost 2 (still ignoring navigation).
            - Else: calibrate + take_image + send: cost 3.

    - This is a lower bound (admissible): we intentionally ignore necessary navigation, store-emptying actions,
      visit/unvisit, and channel availability; these can only add cost in real plans, never reduce it below the
      counts above.

    - Implementation details:
      * We use the bitwise state representation exclusively to check truth of facts quickly (no explicit repr on states),
        in accordance with the framework's performance requirements.
      * We pre-index the relevant facts into compact bitmasks to allow O(1) checks per goal during evaluation:
          - For soil/rock: per waypoint masks of have_*_analysis facts.
          - For image: per (objective,mode) masks of have_image facts and of calibrated facts on cameras that target the
            objective and support the mode.
      * Goals are fetched from model.goals via model.state_explicit_repr(model.goals) and mapped back to Fact objects.

    - Computational efficiency:
      * All heavy parsing and indexing happens once in __init__.
      * Per-state heuristic evaluation is a few bit operations and additions.

    - Consistency:
      * The heuristic is consistent under uniform action cost assumptions, as each action application cannot reduce
        the lower bound by more than the action cost itself. We never overestimate, and we only reduce the count when
        a corresponding fact becomes true.

    Usage:
      h = RoverGPT5Heuristic13(model, rover_gpt5_heuristic_13=True)
      value = h.evaluate(state_bitwise, task_network)

    Notes:
      - This heuristic is intentionally optimistic; it underestimates real costs because it ignores navigation and
        resource-handling subtasks. This preserves admissibility while still providing meaningful guidance that scales
        to large Rover instances.
    """

    def __init__(self, model, rover_gpt5_heuristic_13: bool = True):
        super().__init__(model)
        self.enabled = rover_gpt5_heuristic_13

        # Map from full grounded fact name (e.g., '+communicated_soil_data[wp0]') to Fact object
        self._fact_by_name: Dict[str, object] = {}

        # Goals (Fact objects) and typed goal info entries for fast evaluation
        # Each entry is a dict with keys depending on type ('soil', 'rock', 'image'):
        #  common: 'goal_id' (int)
        #  soil/rock: 'have_mask' (int bitmask of have_*_analysis facts)
        #  image: 'haveimg_mask' (int), 'calib_mask' (int)
        self._goal_entries: List[Dict[str, object]] = []

        # Indices for quick checks
        self._have_soil_mask_by_wp: Dict[str, int] = {}
        self._have_rock_mask_by_wp: Dict[str, int] = {}
        self._have_img_mask_by_obj_mode: Dict[Tuple[str, str], int] = {}
        self._calibrated_mask_by_obj_mode: Dict[Tuple[str, str], int] = {}

        # Static mappings (built from static facts):
        #   camera -> objective (its calibration target)
        self._camera_to_objective: Dict[str, str] = {}
        #   camera -> set of supported modes
        self._camera_to_modes: Dict[str, Set[str]] = {}

        # Build indices
        self._index_facts()
        self._index_goals()

    # ------------------------ Internal setup and parsing helpers ------------------------

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact names in format "+predicate[a,b]" or "-predicate[a,b]" to (predicate, [args]).
        Returns (None, []) if the format does not match.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _index_facts(self) -> None:
        """
        Pre-index all facts we will need:
        - Build name->Fact map
        - Build camera_to_objective and camera_to_modes from static facts
        - Build masks for have_soil_analysis (by waypoint), have_rock_analysis (by waypoint)
        - Build masks for have_image (by (objective, mode))
        - Build masks for calibrated facts by (objective, mode) using camera_to_objective and camera_to_modes
        """
        # 1) Map fact names to Fact objects
        for f in self.model.facts:
            self._fact_by_name[f.name] = f

        # Temporary collectors for building masks
        have_soil_by_wp_list: Dict[str, List[int]] = {}
        have_rock_by_wp_list: Dict[str, List[int]] = {}
        have_img_by_obj_mode_list: Dict[Tuple[str, str], List[int]] = {}
        calibrated_facts: List[Tuple[str, str, int]] = []  # (camera, rover, fact_id)

        # 2) Scan facts, collect static mappings and relevant fact IDs
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            # Static facts for cameras
            if pred == "calibration_target" and len(args) == 2:
                cam, obj = args[0], args[1]
                self._camera_to_objective[cam] = obj
            elif pred == "supports" and len(args) == 2:
                cam, mode = args[0], args[1]
                self._camera_to_modes.setdefault(cam, set()).add(mode)

            # Dynamic/relevant facts
            elif pred == "have_soil_analysis" and len(args) == 2:
                # have_soil_analysis[rover, waypoint]
                wp = args[1]
                have_soil_by_wp_list.setdefault(wp, []).append(f.global_id)
            elif pred == "have_rock_analysis" and len(args) == 2:
                # have_rock_analysis[rover, waypoint]
                wp = args[1]
                have_rock_by_wp_list.setdefault(wp, []).append(f.global_id)
            elif pred == "have_image" and len(args) == 3:
                # have_image[rover, objective, mode]
                obj, mode = args[1], args[2]
                have_img_by_obj_mode_list.setdefault((obj, mode), []).append(f.global_id)
            elif pred == "calibrated" and len(args) == 2:
                # calibrated[camera, rover]
                cam, rov = args[0], args[1]
                calibrated_facts.append((cam, rov, f.global_id))

        # 3) Build bitmasks for soil/rock HAVE facts
        for wp, id_list in have_soil_by_wp_list.items():
            mask = 0
            for fid in id_list:
                mask |= (1 << fid)
            self._have_soil_mask_by_wp[wp] = mask

        for wp, id_list in have_rock_by_wp_list.items():
            mask = 0
            for fid in id_list:
                mask |= (1 << fid)
            self._have_rock_mask_by_wp[wp] = mask

        # 4) Build bitmasks for have_image by (objective, mode)
        for key, id_list in have_img_by_obj_mode_list.items():
            mask = 0
            for fid in id_list:
                mask |= (1 << fid)
            self._have_img_mask_by_obj_mode[key] = mask

        # 5) Build calibrated masks by (objective, mode).
        #    We include only calibrated facts for cameras whose calibration_target is the objective and
        #    that support the given mode (tighter but still admissible).
        calib_masks: Dict[Tuple[str, str], int] = {}
        for cam, rov, fid in calibrated_facts:
            obj = self._camera_to_objective.get(cam)
            if not obj:
                continue
            modes = self._camera_to_modes.get(cam, set())
            for mode in modes:
                key = (obj, mode)
                calib_masks[key] = calib_masks.get(key, 0) | (1 << fid)
        self._calibrated_mask_by_obj_mode = calib_masks

    def _index_goals(self) -> None:
        """
        Fetch goal facts from the model, and for each goal build a compact entry used at evaluation time.
        """
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            fact_obj = self._fact_by_name.get(gname)
            if fact_obj is None:
                # Should not happen, but be robust.
                pred, _ = self._parse_fact_name(gname)
                # If we cannot map it, skip silently (no contribution to heuristic)
                continue

            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue

            if pred == "communicated_soil_data" and len(args) == 1:
                wp = args[0]
                entry = {
                    "type": "soil",
                    "goal_id": fact_obj.global_id,
                    "have_mask": self._have_soil_mask_by_wp.get(wp, 0),
                }
                self._goal_entries.append(entry)

            elif pred == "communicated_rock_data" and len(args) == 1:
                wp = args[0]
                entry = {
                    "type": "rock",
                    "goal_id": fact_obj.global_id,
                    "have_mask": self._have_rock_mask_by_wp.get(wp, 0),
                }
                self._goal_entries.append(entry)

            elif pred == "communicated_image_data" and len(args) == 2:
                obj, mode = args[0], args[1]
                entry = {
                    "type": "image",
                    "goal_id": fact_obj.global_id,
                    "haveimg_mask": self._have_img_mask_by_obj_mode.get((obj, mode), 0),
                    "calib_mask": self._calibrated_mask_by_obj_mode.get((obj, mode), 0),
                }
                self._goal_entries.append(entry)
            else:
                # Unknown/unsupported goal predicate -> ignore (safe lower bound)
                continue

    # ------------------------ Heuristic API ------------------------

    def evaluate(self, state_bitwise: int, task_network: List[object]) -> int:
        """
        Compute the admissible lower bound described in the class docstring.

        - No iteration over explicit state; only bitwise checks are used.
        - For each unsatisfied goal, we add:
            soil: 1 if any have_soil_analysis[* , wp], else 2
            rock: 1 if any have_rock_analysis[* , wp], else 2
            image: 1 if any have_image[* , obj, mode]; else 2 if any calibrated for (obj,mode); else 3
        """
        if not self.enabled:
            return 0

        h = 0
        for entry in self._goal_entries:
            goal_id = entry["goal_id"]
            # Goal already true? Skip
            if (state_bitwise >> goal_id) & 1:
                continue

            etype = entry["type"]
            if etype == "soil":
                have_mask = entry.get("have_mask", 0)
                if have_mask and (state_bitwise & have_mask):
                    h += 1
                else:
                    h += 2

            elif etype == "rock":
                have_mask = entry.get("have_mask", 0)
                if have_mask and (state_bitwise & have_mask):
                    h += 1
                else:
                    h += 2

            elif etype == "image":
                haveimg_mask = entry.get("haveimg_mask", 0)
                calib_mask = entry.get("calib_mask", 0)
                if haveimg_mask and (state_bitwise & haveimg_mask):
                    h += 1
                elif calib_mask and (state_bitwise & calib_mask):
                    h += 2
                else:
                    h += 3
            else:
                # Unknown type -> contribute 0 (safe)
                h += 0

        return h

    # Some planners call the heuristic as a function
    __call__ = evaluate

    def __repr__(self) -> str:
        return f"<RoverGPT5Heuristic13 enabled={self.enabled}>"

