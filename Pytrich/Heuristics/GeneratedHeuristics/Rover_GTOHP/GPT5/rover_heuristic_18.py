from typing import Dict, List, Tuple, Set, Optional, Union

# Fallback base Heuristic if import path differs in the execution environment
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    class Heuristic:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, state_bitwise:int, task_network:List):
            return 0


class RoverGPT5Heuristic18(Heuristic):
    """
    Domain-dependent heuristic for HDDL Rover domain.

    Design overview
    ----------------
    This heuristic exploits the Rover domain structure and returns a fast, admissible
    lower bound on the remaining effort. It combines two complementary lower bounds
    and returns their maximum:

    1) Goal-fact lower bound (h_goals):
       - For each outstanding goal fact (communicated_soil_data, communicated_rock_data,
         communicated_image_data), we compute the minimal number of primitive actions
         still required from the current state:
         * Soil/Rock data for a waypoint p:
           - If communicated_*[p] already true -> 0
           - Else if any have_*_analysis[x,p] true -> at least 1 action (a single communication)
           - Else -> at least 2 actions (one sample_* + one communicate_*), ignoring movement
         * Image data for (objective o, mode m):
           - If communicated_image_data[o,m] true -> 0
           - Else if any have_image[x,o,m] true -> 1 (just communication)
           - Else if there exists a calibrated camera on some rover x at a waypoint p s.t.
             visible_from[o,p] holds and at[x,p] holds -> 2 actions (take_image + communicate)
           - Else -> 3 actions (calibrate + take_image + communicate), ignoring movement
       This is an admissible lower bound because we relax movement/visibility logistics
       and count only the irreducible action cores.

    2) Task-network lower bound (h_tasks):
       - The planner must also refine the task network to empty. We associate a minimal
         primitive-cost per abstract task (again under a domain-specific relaxation):
           * get_soil_data -> 2 (sample + communicate)
           * get_rock_data -> 2 (sample + communicate)
           * get_image_data -> 3 (calibrate + take_image + communicate)
           * send_*_data   -> 1 (one communication action)
           * do_calibrate  -> 1 (one calibrate action)
           * empty_store, do_navigate1, do_navigate2 -> 0 (can be a no-op under methods)
         For primitive operators remaining in the TN, we add their action cost (defaults to 1).
       Summing these per-item lower bounds yields a lower bound on the remaining number of
       steps to empty the task network (ignoring ordering and preconditions).

    The final heuristic is the maximum of the above two lower bounds:
        h(state, tn) = max(h_goals(state), h_tasks(tn))
    This preserves admissibility since both components are themselves admissible.

    Efficiency notes
    ----------------
    - All state-membership checks are done via bitwise tests on fact global_ids.
    - The initialization pre-indexes only the necessary fact IDs and static relations
      to make per-node evaluation very fast, even for large instances.

    Usage contract
    --------------
    - __init__ takes the model via the parameter name `rover_gpt5_heuristic_18` as required.
    - __call__(state_bitwise, task_network) returns a non-negative integer heuristic value.
    """

    # ---------------------- Construction ----------------------
    def __init__(self, rover_gpt5_heuristic_18):
        super().__init__()
        self.model = rover_gpt5_heuristic_18

        # Pre-indexed data structures
        self._pred_cache: Dict[str, List[Tuple[int, Tuple[str, ...]]]] = {}

        # Goal targets parsed from model.goals (bitset) -> required communicated facts
        self.goal_communicated_soil: Set[str] = set()          # set of waypoint strings
        self.goal_communicated_rock: Set[str] = set()          # set of waypoint strings
        self.goal_communicated_image: Set[Tuple[str, str]] = set()  # set of (objective, mode)

        # Fact-ID indexes for fast bitwise membership tests
        self.comm_soil_fid_by_wp: Dict[str, int] = {}
        self.comm_rock_fid_by_wp: Dict[str, int] = {}
        self.comm_image_fid_by_om: Dict[Tuple[str, str], int] = {}

        self.have_soil_fids_by_wp: Dict[str, List[int]] = {}
        self.have_rock_fids_by_wp: Dict[str, List[int]] = {}
        self.have_image_fids_by_om: Dict[Tuple[str, str], List[int]] = {}

        # Static relations (assumed not deleted by any operator): on_board, visible_from
        self.on_board_pairs: Set[Tuple[str, str]] = set()        # (camera, rover) that are true
        self.calibrated_fid_by_cx: Dict[Tuple[str, str], int] = {}  # (camera, rover) -> fact id

        # at[x,p] fact ids for quick position checks
        self.at_fid_by_xp: Dict[Tuple[str, str], int] = {}

        # Visibility map: objective -> set(waypoints) s.t. visible_from[objective, waypoint]
        self.visible_from_p_by_obj: Dict[str, Set[str]] = {}

        # Pre-index all facts we care about
        self._index_all_facts()

        # Parse the goal bitset to know which communicated facts are actually required
        self._parse_goals()

        # Minimal abstract-task expansion costs used for h_tasks lower bound
        self._abstract_task_min_cost: Dict[str, int] = {
            'get_soil_data': 2,
            'get_rock_data': 2,
            'get_image_data': 3,
            'send_soil_data': 1,
            'send_rock_data': 1,
            'send_image_data': 1,
            'do_calibrate': 1,
            # Navigation and empty-store can be 0 in best case (no-op methods)
            'do_navigate1': 0,
            'do_navigate2': 0,
            'empty_store': 0,
        }

    # ---------------------- Public API ----------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[object]]) -> int:
        h_g = self._goals_lower_bound(state_bitwise)
        h_t = self._tasks_lower_bound(task_network)
        return max(h_g, h_t)

    # ---------------------- Parsing and indexing ----------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse grounded fact name in the format: +predicate[arg1,arg2,...] or -predicate[arg1,...]
        Returns (predicate, [args]). On failure returns (None, []).
        """
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            core = name[1:]
            lb = core.find('[')
            if lb > 0 and core.endswith(']'):
                pred = core[:lb]
                args_str = core[lb + 1:-1]
                args = args_str.split(',') if args_str else []
                return pred, args
        return None, []

    def _index_all_facts(self):
        """
        Build fast lookup tables for all predicates we use in the heuristic.
        """
        # Pre-scan facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            # Cache all appearances per predicate name for potential later use
            self._pred_cache.setdefault(pred, []).append((fact.global_id, tuple(args)))

            # Identify communicated_* fact IDs
            if pred == 'communicated_soil_data' and len(args) == 1:
                wp = args[0]
                self.comm_soil_fid_by_wp[wp] = fact.global_id
            elif pred == 'communicated_rock_data' and len(args) == 1:
                wp = args[0]
                self.comm_rock_fid_by_wp[wp] = fact.global_id
            elif pred == 'communicated_image_data' and len(args) == 2:
                o, m = args
                self.comm_image_fid_by_om[(o, m)] = fact.global_id

            # have_* facts
            elif pred == 'have_soil_analysis' and len(args) == 2:
                x, wp = args
                self.have_soil_fids_by_wp.setdefault(wp, []).append(fact.global_id)
            elif pred == 'have_rock_analysis' and len(args) == 2:
                x, wp = args
                self.have_rock_fids_by_wp.setdefault(wp, []).append(fact.global_id)
            elif pred == 'have_image' and len(args) == 3:
                x, o, m = args
                self.have_image_fids_by_om.setdefault((o, m), []).append(fact.global_id)

            # at[x,p]
            elif pred == 'at' and len(args) == 2:
                x, p = args
                self.at_fid_by_xp[(x, p)] = fact.global_id

            # on_board[c,x]
            elif pred == 'on_board' and len(args) == 2:
                c, x = args
                # on_board is static; detect if it is true in initial state
                if self._is_true(self.model.initial_state, fact.global_id):
                    self.on_board_pairs.add((c, x))

            # calibrated[c,x] -> store fact id (relevant only if (c,x) on_board)
            elif pred == 'calibrated' and len(args) == 2:
                c, x = args
                self.calibrated_fid_by_cx[(c, x)] = fact.global_id

            # visible_from[o,p] (static visibility map)
            elif pred == 'visible_from' and len(args) == 2:
                o, p = args
                self.visible_from_p_by_obj.setdefault(o, set()).add(p)

    def _parse_goals(self):
        """
        Read the bitwise goal set and extract the concrete communicated targets.
        """
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if not pred:
                continue
            if pred == 'communicated_soil_data' and len(args) == 1:
                self.goal_communicated_soil.add(args[0])
            elif pred == 'communicated_rock_data' and len(args) == 1:
                self.goal_communicated_rock.add(args[0])
            elif pred == 'communicated_image_data' and len(args) == 2:
                self.goal_communicated_image.add((args[0], args[1]))

    # ---------------------- Lower bounds ----------------------
    def _goals_lower_bound(self, state_bitwise: int) -> int:
        """
        Compute h_goals: sum of minimal action cores still required to achieve
        all outstanding communicated_* goal facts from the current state.
        """
        h = 0

        # Soil
        for wp in self.goal_communicated_soil:
            fid_comm = self.comm_soil_fid_by_wp.get(wp)
            if fid_comm is None:
                continue  # safety
            if self._is_true(state_bitwise, fid_comm):
                continue  # already achieved
            # If have_soil_analysis for this waypoint already in state, just 1 (communicate)
            if any(self._is_true(state_bitwise, fid) for fid in self.have_soil_fids_by_wp.get(wp, [])):
                h += 1
            else:
                # sample_soil + communicate -> at least 2
                h += 2

        # Rock
        for wp in self.goal_communicated_rock:
            fid_comm = self.comm_rock_fid_by_wp.get(wp)
            if fid_comm is None:
                continue
            if self._is_true(state_bitwise, fid_comm):
                continue
            if any(self._is_true(state_bitwise, fid) for fid in self.have_rock_fids_by_wp.get(wp, [])):
                h += 1
            else:
                h += 2

        # Images
        for (o, m) in self.goal_communicated_image:
            fid_comm = self.comm_image_fid_by_om.get((o, m))
            if fid_comm is None:
                continue
            if self._is_true(state_bitwise, fid_comm):
                continue
            # If have_image already true somewhere: only need to communicate
            if any(self._is_true(state_bitwise, fid) for fid in self.have_image_fids_by_om.get((o, m), [])):
                h += 1
            else:
                # Check if we can do take_image immediately (calibrated & at a visible waypoint)
                if self._can_take_image_now(state_bitwise, o):
                    # take_image + communicate
                    h += 2
                else:
                    # calibrate + take_image + communicate
                    h += 3

        return h

    def _tasks_lower_bound(self, task_network: List[Union[object]]) -> int:
        """
        Compute h_tasks: lower bound on steps required to refine the task network to empty,
        by summing minimal expansion costs per abstract task and counting remaining operators.
        """
        if not task_network:
            return 0

        total = 0
        # Identify operators vs. abstract tasks using the global_id ranges
        op_lo, op_hi = self.model.iop_init, self.model.iop_end
        abt_lo, abt_hi = self.model.iabt_init, self.model.iabt_end

        for comp in task_network:
            try:
                gid = comp.global_id
            except Exception:
                # Unknown component type: be conservative -> add 0
                continue

            # Primitive operator
            if gid >= op_lo and gid <= op_hi:
                # Use provided operator cost if available, else 1
                cost = getattr(comp, 'cost', 1)
                if isinstance(cost, int) and cost >= 0:
                    total += cost
                else:
                    total += 1

            # Abstract task
            elif gid >= abt_lo and gid <= abt_hi:
                tname = getattr(comp, 'name', '')
                # Extract the task head name (drop parameters if present)
                head = tname.split('(')[0] if '(' in tname else tname
                total += self._abstract_task_min_cost.get(head, 0)

            else:
                # Decomposition nodes or unknown: contribute 0 to remain admissible
                total += 0

        return total

    # ---------------------- Helpers ----------------------
    def _is_true(self, state_bitwise: int, fact_gid: int) -> bool:
        return ((state_bitwise >> fact_gid) & 1) == 1

    def _can_take_image_now(self, state_bitwise: int, objective: str) -> bool:
        """
        Check if there exists any (camera c, rover x) currently calibrated, and some waypoint p
        such that at[x,p] is true and visible_from[objective,p] holds. We intentionally ignore
        'supports' and other static checks here to keep this a (safe) lower bound.
        Returns True if a single take_image action could be applied immediately.
        """
        # For efficiency: gather all (c,x) that are on_board and currently calibrated
        for (c, x) in self.on_board_pairs:
            fid_cal = self.calibrated_fid_by_cx.get((c, x))
            if fid_cal is None:
                continue
            if not self._is_true(state_bitwise, fid_cal):
                continue
            # Rover x position must be at some waypoint p where objective is visible
            for p in self.visible_from_p_by_obj.get(objective, set()):
                fid_at = self.at_fid_by_xp.get((x, p))
                if fid_at is not None and self._is_true(state_bitwise, fid_at):
                    return True
        return False


# Alias (if the framework loads by module-level variable name)
rover_gpt5_heuristic_18 = RoverGPT5Heuristic18
