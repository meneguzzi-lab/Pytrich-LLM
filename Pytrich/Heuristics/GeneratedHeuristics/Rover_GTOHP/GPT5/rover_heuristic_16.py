from typing import Dict, List, Set, Tuple, Optional

# Pytrich base class (provided by the framework)
from Pytrich.Heuristics.heuristic import Heuristic


class RoverGPT5Heuristic16(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Rover domain.

    Key idea: Each goal in Rover is a "communication" goal that is achieved by a
    pipeline of primitive actions. We compute, for each unmet goal, a safe
    (admissible) lower bound on the minimal number of primitive actions still
    required, leveraging domain structure:
      - For soil and rock data: sample (if not already analyzed) + communicate,
        with at least one navigate if clearly unavoidable. We also consider
        store emptiness for sampling.
      - For image data: (calibrate if needed) + take_image + communicate,
        with at least one navigate if clearly unavoidable. We reason about
        available cameras (on-board, supports mode, correct calibration target)
        and whether current rover positions are already at a waypoint suitable
        for visibility to the objective and/or the lander.

    We keep the heuristic admissible by only adding navigation cost when it is
    provably necessary in the current state (e.g., no rover currently at any
    suitable waypoint), otherwise we assume zero navigation (best-case lower bound).

    IMPORTANT IMPLEMENTATION NOTES (Pytrich):
    - Facts are accessed via bitwise checks using their global_id.
    - Goals are stored as a bitwise integer; we parse their explicit string repr once.
    - We pre-index static relations (visible, visible_from, at_lander, on_board,
      supports, calibration_target) and dynamic fact templates for quick checks.

    This heuristic ignores the current task network to avoid double counting with
    goal pipelines. Counting primitive operators already in the TN alongside goal
    pipelines risks overestimation (inadmissibility). The pipeline-based estimate
    alone is a robust lower bound.
    """

    def __init__(self, model, rover_gpt5_heuristic_16: bool = True):
        super().__init__(model)
        self.model = model

        # --- Internal indices and caches ---
        # Maps for quick lookup of Fact objects by predicate/arguments.
        self._facts_by_pred: Dict[str, List] = {}
        self._facts_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}

        # Static domain structures (filled in _build_static_indices)
        self.lander_waypoints: Set[str] = set()
        self.visible_pairs: Set[Tuple[str, str]] = set()  # (wp1, wp2)
        self.vis_to_lander: Set[str] = set()              # wp s.t. visible(wp, lander_wp)
        self.obj_visible_wps: Dict[str, Set[str]] = {}    # objective -> set of wps
        self.onboard_cams_by_rover: Dict[str, Set[str]] = {}  # rover -> set(cameras)
        self.supports_cm: Set[Tuple[str, str]] = set()    # (camera, mode)
        self.calib_target: Dict[str, str] = {}            # camera -> objective

        # Dynamic template indices for efficient state queries
        self.at_facts_by_rover: Dict[str, List[Tuple[str, object]]] = {}  # rover -> list of (wp, fact)
        self.have_soil_by_p: Dict[str, List[Tuple[str, object]]] = {}     # wp -> list of (rover, fact)
        self.have_rock_by_p: Dict[str, List[Tuple[str, object]]] = {}     # wp -> list of (rover, fact)
        self.have_image_by_om: Dict[Tuple[str, str], List[Tuple[str, object]]] = {}  # (obj,mode) -> list of (rover, fact)
        self.calibrated_fact_by_cx: Dict[Tuple[str, str], object] = {}     # (camera, rover) -> fact

        self.eq_soil: Set[str] = set()
        self.eq_rock: Set[str] = set()
        self.eq_imaging: Set[str] = set()

        self.store_of: Dict[str, str] = {}                 # store -> rover
        self.stores_by_rover: Dict[str, List[str]] = {}    # rover -> list(stores)
        self.empty_fact_by_store: Dict[str, object] = {}   # store -> fact

        # Preparse and index facts once (static wrt model)
        self._index_facts()
        self._build_static_indices()
        self._build_dynamic_templates()

        # Parse goal facts and extract targets
        self.goal_soils: Set[str] = set()
        self.goal_rocks: Set[str] = set()
        self.goal_images: Set[Tuple[str, str]] = set()  # (objective, mode)
        self._parse_goals()

    # ---------------------- Utility parsing/index helpers ----------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bidx = name_.find("[")
            if bidx > 0 and name_.endswith("]"):
                pred = name_[:bidx]
                args_str = name_[bidx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return pred, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_obj) -> bool:
        return ((state_bitwise >> fact_obj.global_id) & 1) == 1

    def _index_facts(self):
        # Build: predicate -> list of Fact; and (pred, args_tuple) -> Fact
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append(f)
            self._facts_by_key[(pred, tuple(args))] = f

    def _build_static_indices(self):
        # visible pairs (static)
        for f in self._facts_by_pred.get("visible", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                self.visible_pairs.add((args[0], args[1]))

        # at_lander: collect waypoints with landers
        for f in self._facts_by_pred.get("at_lander", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                lander_wp = args[1]
                self.lander_waypoints.add(lander_wp)

        # Compute waypoints from which a lander is visible
        self.vis_to_lander = set()
        for (w1, w2) in self.visible_pairs:
            if w2 in self.lander_waypoints:
                self.vis_to_lander.add(w1)

        # visible_from (objective, waypoint)
        for f in self._facts_by_pred.get("visible_from", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                o, w = args[0], args[1]
                self.obj_visible_wps.setdefault(o, set()).add(w)

        # on_board (camera, rover)
        for f in self._facts_by_pred.get("on_board", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, rov = args[0], args[1]
                self.onboard_cams_by_rover.setdefault(rov, set()).add(cam)

        # supports (camera, mode)
        for f in self._facts_by_pred.get("supports", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, mode = args[0], args[1]
                self.supports_cm.add((cam, mode))

        # calibration_target (camera, objective)
        for f in self._facts_by_pred.get("calibration_target", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, obj = args[0], args[1]
                self.calib_target[cam] = obj

        # equipment predicates (static per rover)
        for f in self._facts_by_pred.get("equipped_for_soil_analysis", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self.eq_soil.add(args[0])
        for f in self._facts_by_pred.get("equipped_for_rock_analysis", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self.eq_rock.add(args[0])
        for f in self._facts_by_pred.get("equipped_for_imaging", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                self.eq_imaging.add(args[0])

        # store_of and empty/full
        for f in self._facts_by_pred.get("store_of", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                store, rov = args[0], args[1]
                self.store_of[store] = rov
                self.stores_by_rover.setdefault(rov, []).append(store)
        for f in self._facts_by_pred.get("empty", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 1:
                store = args[0]
                self.empty_fact_by_store[store] = f

    def _build_dynamic_templates(self):
        # Build per-rover lists of at-facts
        for f in self._facts_by_pred.get("at", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                rov, wp = args[0], args[1]
                self.at_facts_by_rover.setdefault(rov, []).append((wp, f))

        # have_soil_analysis (rover, waypoint)
        for f in self._facts_by_pred.get("have_soil_analysis", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                rov, wp = args[0], args[1]
                self.have_soil_by_p.setdefault(wp, []).append((rov, f))

        # have_rock_analysis (rover, waypoint)
        for f in self._facts_by_pred.get("have_rock_analysis", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                rov, wp = args[0], args[1]
                self.have_rock_by_p.setdefault(wp, []).append((rov, f))

        # have_image (rover, objective, mode)
        for f in self._facts_by_pred.get("have_image", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 3:
                rov, obj, mode = args[0], args[1], args[2]
                self.have_image_by_om.setdefault((obj, mode), []).append((rov, f))

        # calibrated (camera, rover)
        for f in self._facts_by_pred.get("calibrated", []):
            pred, args = self._parse_fact_name(f.name)
            if len(args) == 2:
                cam, rov = args[0], args[1]
                self.calibrated_fact_by_cx[(cam, rov)] = f

    def _parse_goals(self):
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for g in goal_names:
            pred, args = self._parse_fact_name(g)
            if pred == "communicated_soil_data" and len(args) == 1:
                self.goal_soils.add(args[0])
            elif pred == "communicated_rock_data" and len(args) == 1:
                self.goal_rocks.add(args[0])
            elif pred == "communicated_image_data" and len(args) == 2:
                self.goal_images.add((args[0], args[1]))

    # ----------------------------- State helpers -------------------------------
    def _rover_at(self, state: int, rover: str) -> Optional[str]:
        # returns current waypoint of rover if determinable, else None
        for wp, f in self.at_facts_by_rover.get(rover, []):
            if self._is_true(state, f):
                return wp
        return None

    def _any_rover_at_any(self, state: int, waypoints: Set[str]) -> bool:
        # Check if any rover is currently at any waypoint in the set
        # (coarse but fast enough)
        for rov, lst in self.at_facts_by_rover.items():
            for wp, f in lst:
                if wp in waypoints and self._is_true(state, f):
                    return True
        return False

    def _any_of_these_rovers_at_any(self, state: int, rovers: Set[str], waypoints: Set[str]) -> bool:
        for rov in rovers:
            for wp, f in self.at_facts_by_rover.get(rov, []):
                if wp in waypoints and self._is_true(state, f):
                    return True
        return False

    def _rovers_at_vis_to_lander(self, state: int, rovers: Optional[Set[str]] = None) -> bool:
        if rovers is None:
            # any rover
            return self._any_rover_at_any(state, self.vis_to_lander)
        else:
            return self._any_of_these_rovers_at_any(state, rovers, self.vis_to_lander)

    def _any_analysis_present(self, state: int, p: str, soil: bool) -> Tuple[bool, Set[str]]:
        holders: Set[str] = set()
        lst = self.have_soil_by_p.get(p, []) if soil else self.have_rock_by_p.get(p, [])
        exists = False
        for rov, f in lst:
            if self._is_true(state, f):
                exists = True
                holders.add(rov)
        return exists, holders

    def _any_image_present(self, state: int, obj: str, mode: str) -> Tuple[bool, Set[str]]:
        holders: Set[str] = set()
        lst = self.have_image_by_om.get((obj, mode), [])
        exists = False
        for rov, f in lst:
            if self._is_true(state, f):
                exists = True
                holders.add(rov)
        return exists, holders

    def _any_empty_store_available(self, state: int, candidate_rovers: Set[str]) -> bool:
        # A rover can sample only if some store_of(s, rov) and empty(s)
        for rov in candidate_rovers:
            for s in self.stores_by_rover.get(rov, []):
                f_empty = self.empty_fact_by_store.get(s)
                if f_empty is not None and self._is_true(state, f_empty):
                    return True
        return False

    def _exists_rover_at_wp(self, state: int, rov: str, target_wp: str) -> bool:
        for wp, f in self.at_facts_by_rover.get(rov, []):
            if wp == target_wp and self._is_true(state, f):
                return True
        return False

    # ---------------------------- Cost components -----------------------------
    def _soil_or_rock_goal_cost(self, state: int, p: str, is_soil: bool) -> int:
        """Lower bound for a single communicated_soil_data/communicated_rock_data at waypoint p."""
        # If already satisfied
        pred = "communicated_soil_data" if is_soil else "communicated_rock_data"
        goal_fact = self._facts_by_key.get((pred, (p,)))
        if goal_fact is not None and self._is_true(state, goal_fact):
            return 0

        # Base: 1 communicate action is ultimately required
        base_comm = 1

        # Do we already have analysis?
        have_analysis, holders = self._any_analysis_present(state, p, soil=is_soil)
        if have_analysis:
            # If any holder rover is already at a waypoint with line of sight to lander, no nav is strictly needed for comm
            nav_comm = 0
            if not self._rovers_at_vis_to_lander(state, holders):
                # Not provably at a comm point now => at least one navigate may be needed
                nav_comm = 1
            return base_comm + nav_comm

        # Otherwise we need to sample
        # Choose rovers that can sample this type
        capable = self.eq_soil if is_soil else self.eq_rock
        if not capable:
            # No capable rover in domain model (degenerate). Return 0 as a safe lower bound (unsolvable states get pruned by search).
            return 0

        nav_to_p = 1
        empty_needed = 1

        # If any capable rover already at p, one mandatory navigate can be 0 (we're already there)
        at_set = {p}
        if self._any_of_these_rovers_at_any(state, capable, at_set):
            nav_to_p = 0

        # Store emptiness: if any capable rover has an empty store now, we don't need a drop action
        if self._any_empty_store_available(state, capable):
            empty_needed = 0

        sample_cost = 1 + nav_to_p + empty_needed  # (sample) + possible nav + possible drop

        # After sampling, we will be at p. If p is visible to a lander waypoint, we can communicate from there (no extra nav)
        comm_nav = 0
        visible_to_any_lander = any((p, lw) in self.visible_pairs for lw in self.lander_waypoints)
        if not visible_to_any_lander:
            comm_nav = 1

        return base_comm + sample_cost + comm_nav

    def _image_goal_cost(self, state: int, obj: str, mode: str) -> int:
        """Lower bound for a communicated_image_data(obj, mode) goal."""
        goal_fact = self._facts_by_key.get(("communicated_image_data", (obj, mode)))
        if goal_fact is not None and self._is_true(state, goal_fact):
            return 0

        base_comm = 1

        # If we already have an image on some rover, we only need to navigate (maybe) to a comm point and communicate
        have_img, img_holders = self._any_image_present(state, obj, mode)
        if have_img:
            nav_comm = 0
            if not self._rovers_at_vis_to_lander(state, img_holders):
                nav_comm = 1
            return base_comm + nav_comm

        # Otherwise we need to produce an image: choose the best (rover, camera)
        best_produce = None  # minimal cost to produce (excluding communication), lower bound

        # Waypoints from which the objective is visible
        obj_wps = self.obj_visible_wps.get(obj, set())
        # If we can also ensure comm without extra nav after taking the picture
        # check if there exists a waypoint suitable for both viewing the objective and lander comm
        inter_vis = {w for w in obj_wps if w in self.vis_to_lander}
        can_merge_take_and_comm = len(inter_vis) > 0

        # Iterate eligible (rover, camera)
        for rov, cams in self.onboard_cams_by_rover.items():
            if rov not in self.eq_imaging:
                continue  # cannot take images
            for cam in cams:
                # camera must be for this objective and support the requested mode
                if self.calib_target.get(cam) != obj:
                    continue
                if (cam, mode) not in self.supports_cm:
                    continue

                # Is camera already calibrated on this rover?
                cal_fact = self.calibrated_fact_by_cx.get((cam, rov))
                is_calibrated = self._is_true(state, cal_fact) if cal_fact is not None else False

                # Where is this rover now?
                cur_wp = self._rover_at(state, rov)
                at_obj_wp = cur_wp in obj_wps if cur_wp is not None else False

                # Cost to produce the image (take + optional calibrate + at least one nav if not at a view wp)
                if at_obj_wp:
                    # Already at a valid viewing waypoint
                    produce_cost = 1 if is_calibrated else 2  # take or (calibrate + take)
                else:
                    # Need at least one navigation to some valid viewing waypoint
                    if is_calibrated:
                        produce_cost = 1 + 1  # 1 nav + 1 take
                    else:
                        produce_cost = 1 + 2  # 1 nav + (calibrate + take)

                if (best_produce is None) or (produce_cost < best_produce):
                    best_produce = produce_cost

        if best_produce is None:
            # No possible (rover, camera) combo to satisfy this image goal right now.
            # Return 0 as a safe lower bound; search will handle infeasibility.
            return 0

        # After producing, we need to communicate. If we can plan to take the picture at a waypoint
        # that is also visible to a lander, we can assume no extra nav is strictly necessary for comm (lower bound 0).
        post_comm_nav = 0 if can_merge_take_and_comm else 1

        return best_produce + base_comm + post_comm_nav

    # --------------------------- Public evaluate API ---------------------------
    def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
        """
        Compute admissible lower bound on remaining cost (number of primitive actions) to
        reach a complete solution from the given state.

        We sum independent lower bounds for each unsatisfied communicated_* goal using
        domain-specific pipelines and current-state checks to add at least-one-navigation
        when it is provably required.
        """
        h = 0

        # Soil goals
        for p in self.goal_soils:
            h += self._soil_or_rock_goal_cost(state_bitwise, p, is_soil=True)

        # Rock goals
        for p in self.goal_rocks:
            h += self._soil_or_rock_goal_cost(state_bitwise, p, is_soil=False)

        # Image goals
        for (obj, mode) in self.goal_images:
            h += self._image_goal_cost(state_bitwise, obj, mode)

        return h

    # Some planners might call the heuristic as a function
    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.evaluate(state_bitwise, task_network)
