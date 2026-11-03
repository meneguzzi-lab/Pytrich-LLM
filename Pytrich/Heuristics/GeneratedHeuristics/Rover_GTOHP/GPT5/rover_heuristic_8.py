from typing import Dict, List, Tuple, Optional

# Attempt to import Heuristic base from common locations
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from .Heuristic import Heuristic
    except Exception:
        # Minimal fallback to allow static checking; the planner provides the real base class.
        class Heuristic:
            def __init__(self, model):
                self.model = model
            def evaluate(self, state_bitwise: int, task_network: List) -> int:
                return 0


class RoverGPT5Heuristic8(Heuristic):
    """
    RoverGPT5Heuristic8: Domain-dependent admissible heuristic for HDDL Rover domain (Pytrich Planner).

    Design rationale
    ----------------
    The domain revolves around producing and communicating three kinds of scientific data:
      - Soil analysis data for waypoint p: communicated_soil_data[p]
      - Rock analysis data for waypoint p: communicated_rock_data[p]
      - Image data for objective o and mode m: communicated_image_data[o,m]

    A communicated_* fact can only be achieved after (at least) the following primitive steps:
      - Soil/Rock:
         * If a rover already has the analysis (have_*_analysis[x,p]): 1 action (a communicate_* action)
         * Otherwise: 2 actions (sample_* + communicate_*)
      - Image:
         * If a rover already has the image (have_image[x,o,m]): 1 action (communicate_image_data)
         * Otherwise: 3 actions (calibrate + take_image + communicate_image_data)

    We deliberately do not add navigation, emptying store, or other conditional steps. This keeps the
    heuristic admissible (it never overestimates) while still providing strong guidance: the numbers above
    are true lower bounds on the number of remaining primitive actions per goal atom given the Rover domain
    methods and actions. These bounds hold because in the best case, a rover may already be at a suitable
    location (or already have empty store), so the minimum action counts above are achievable in some states.

    Implementation details
    ----------------------
    - We parse and index all facts once in __init__, creating direct mappings from each goal fact to the set
      of corresponding "have_*" facts that enable immediate communication.
    - During evaluate(), for each unsatisfied communicated_* goal we check whether any enabling have_* fact is
      already true in the current state (bitwise checks with fact.global_id). Based on that, we add a lower-bound
      contribution of 1, 2, or 3 as described above.
    - We only use fast bitwise membership checks; we never reconstruct explicit state representations inside
      evaluate().

    Admissibility and efficiency
    ----------------------------
    - The heuristic is admissible: it only sums action-count lower bounds per goal and never adds costs for
      potentially unnecessary steps (e.g., navigation), so it cannot overestimate the true cost-to-go.
    - The heuristic is efficient: all heavy parsing is done once; per-node evaluation uses constant-time bitwise
      checks over small precomputed lists.

    Notes
    -----
    - The heuristic ignores the current (hierarchical) task network structure when counting cost. This is safe in
      this domain because decomposition itself has no action cost; only primitive actions contribute to plan cost.
      The goals given in the provided problems correspond to the high-level tasks.
    - The __init__ parameter is named 'rover_gpt5_heuristic_8' to satisfy the naming requirement.
    """

    def __init__(self, model, rover_gpt5_heuristic_8: bool = True):
        super().__init__(model)
        self.rover_gpt5_heuristic_8 = rover_gpt5_heuristic_8

        # name -> fact_id mapping (for quick lookup)
        self._fact_name_to_id: Dict[str, int] = {}

        # Parsed indices for enabling facts
        # waypoint -> [fact_ids of have_soil_analysis[x, waypoint]]
        self._have_soil_by_wp: Dict[str, List[int]] = {}
        # waypoint -> [fact_ids of have_rock_analysis[x, waypoint]]
        self._have_rock_by_wp: Dict[str, List[int]] = {}
        # (objective, mode) -> [fact_ids of have_image[x, objective, mode]]
        self._have_image_by_om: Dict[Tuple[str, str], List[int]] = {}

        # Precomputed goal entries: list of tuples describing each communicated_* goal
        #   ('soil', goal_fact_id, [have_soil_fact_ids for that waypoint])
        #   ('rock', goal_fact_id, [have_rock_fact_ids for that waypoint])
        #   ('image', goal_fact_id, [have_image_fact_ids for that (obj,mode)])
        self._goal_entries: List[Tuple[str, int, List[int]]] = []

        # Build indices
        self._precompute_indices()
        self._index_goals()

    # ---------- Internal helpers ----------

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name in the format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        Returns: (predicate, [args]) or (None, []) if parsing fails.
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _precompute_indices(self):
        # Map names to ids and gather enabling facts for have_* predicates
        for f in self.model.facts:
            self._fact_name_to_id[f.name] = f.global_id

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue

            if pred == 'have_soil_analysis':
                # have_soil_analysis[x, waypoint]
                if len(args) == 2:
                    wp = args[1]
                    self._have_soil_by_wp.setdefault(wp, []).append(f.global_id)
            elif pred == 'have_rock_analysis':
                # have_rock_analysis[x, waypoint]
                if len(args) == 2:
                    wp = args[1]
                    self._have_rock_by_wp.setdefault(wp, []).append(f.global_id)
            elif pred == 'have_image':
                # have_image[x, objective, mode]
                if len(args) == 3:
                    o, m = args[1], args[2]
                    self._have_image_by_om.setdefault((o, m), []).append(f.global_id)

    def _index_goals(self):
        # Build goal entries from model.goals bitset
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if not pred:
                continue
            gid = self._fact_name_to_id.get(gname, None)
            if gid is None:
                continue

            if pred == 'communicated_soil_data' and len(args) == 1:
                wp = args[0]
                have_list = self._have_soil_by_wp.get(wp, [])
                self._goal_entries.append(('soil', gid, have_list))
            elif pred == 'communicated_rock_data' and len(args) == 1:
                wp = args[0]
                have_list = self._have_rock_by_wp.get(wp, [])
                self._goal_entries.append(('rock', gid, have_list))
            elif pred == 'communicated_image_data' and len(args) == 2:
                o, m = args[0], args[1]
                have_list = self._have_image_by_om.get((o, m), [])
                self._goal_entries.append(('image', gid, have_list))
            else:
                # Other goal facts (if any) are not part of the standard Rover instances;
                # we conservatively ignore them (admissible: add 0) to avoid overestimation.
                pass

    # ---------- Heuristic ----------

    def evaluate(self, state_bitwise: int, task_network: List) -> int:
        """
        Compute admissible estimate of remaining primitive actions as the sum of per-goal lower bounds:
          - Soil/Rock goal g: 0 if g true; else +1 if some have_*_analysis is true; else +2
          - Image goal g:     0 if g true; else +1 if some have_image is true; else +3
        """
        h = 0
        for kind, goal_fact_id, enabling_fact_ids in self._goal_entries:
            # If goal already achieved in the current state, contributes 0
            if ((state_bitwise >> goal_fact_id) & 1) != 0:
                continue

            # Check whether we already have the corresponding data (have_* fact)
            has_data = False
            for fid in enabling_fact_ids:
                if ((state_bitwise >> fid) & 1) != 0:
                    has_data = True
                    break

            # Accumulate lower-bound contribution per goal kind
            if kind == 'soil' or kind == 'rock':
                # communicated_*: 1 if have_*_analysis, else 2 (sample + communicate)
                h += 1 if has_data else 2
            elif kind == 'image':
                # communicated_image_data: 1 if have_image, else 3 (calibrate + take_image + communicate)
                h += 1 if has_data else 3
            else:
                # Unknown goal type: be safe and add 0 (admissible)
                h += 0

        return h


# Alias to fit potential dynamic loader expectations
HeuristicClass = RoverGPT5Heuristic8
