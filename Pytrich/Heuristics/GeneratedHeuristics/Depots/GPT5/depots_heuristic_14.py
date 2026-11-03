from typing import Dict, List, Tuple, Optional, Any


class Heuristic:
    """
    Minimal base class placeholder. In Pytrich, a Heuristic base class is provided.
    This placeholder allows the heuristic to be imported even if the base class is
    not explicitly available in this snippet. When integrated in Pytrich, the real
    Heuristic will be used and this class will be ignored.
    """
    def __init__(self, model, name: str = "Heuristic"):
        self.model = model
        self.name = name

    def __call__(self, state: int, task_network: List[Any]):
        return 0

    def compute(self, state: int, task_network: List[Any]):
        return self.__call__(state, task_network)


class DepotsGPT5Heuristic14(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depot domain.

    Design rationale and domain insights:
    - Goals are of the form on(crate, surface). Each unsatisfied goal ultimately
      requires a Drop action for that crate onto its target surface. A single Drop
      achieves exactly one such goal, so counting one per unsatisfied goal is an
      admissible lower bound.
    - Additional crate-specific actions are often required:
        * If crate and target surface are already co-located at the same place
          and the crate is resting on some support, the minimal path is Lift + Drop (2 actions).
        * If the crate is already being lifted by a hoist at the same place as the target
          surface, only a Drop is needed (1 action). If lifted at a different place, the
          crate must be transferred via a truck: Load + Unload + Drop (3 crate-specific actions),
          ignoring the drive count to remain a safe lower bound across multiple goals.
        * If the crate is in a truck, the minimal path is Unload + Drop (2 crate-specific actions)
          regardless of location (drive moves are non-crate-specific and may be shared across goals;
          we conservatively do not sum drives across goals to keep the heuristic admissible).
        * If the crate and the target surface are at different places and the crate is not in a truck
          nor being lifted, the minimal crate-specific path is Lift + Load + Unload + Drop (4 actions),
          excluding truck drives.
      We explicitly exclude drive (truck) moves from the per-goal sum to avoid over-counting shared
      actions across multiple goals, thus keeping the heuristic admissible and effective at scale.

    - Hierarchical task network (HTN) guidance:
        * The search must also empty the current task network. We therefore add a separate lower bound
          for the tasks currently in the task network. To remain admissible when combined with the goal
          estimate (which might overlap in actions), we take the maximum of (goal-based LB, task-based LB).
        * For AbstractTasks, we use conservative yet informative lower bounds:
            - do_put_on(c, s): at least 1 action (the final Drop). If we can parse parameters, we reuse
              the same per-goal estimation to get a stronger bound.
            - do_load_truck: at least 2 actions (Lift + Load).
            - do_unload_truck: at least 2 actions (Unload + Drop on some surface); we conservatively
              count only Unload + Drop = 2, which is a clear LB for depositing the crate onto a surface.
            - do_lift_crate: at least 1 action (Lift) in the best case.
            - do_get_truck: at least 1 action (Drive) in the best case when the truck is not already there;
              if we can parse parameters and the truck is already at the place, the LB is 0.
            - do_clear: at least 0..1 actions; when parameters are parseable and the surface is already clear,
              LB=0, otherwise LB=1 (must Lift something off).
        * For primitive operators already in the task network, we add 1 per operator (they must be executed).
      Finally, to remain admissible, the overall heuristic value is:
          h(state, tn) = max( sum_over_goals(crate-specific LBs excluding drives),
                              sum_over_task_network(min per-task LBs) )

    Efficiency considerations:
    - All state membership checks use bitwise operations over precomputed global fact ids.
    - We build indices for predicates: at, on, in, lifting, clear, and optionally available.
    - Per-state evaluation is O(#goals + |task_network|) with small per-item constant factors.

    This heuristic has been designed to be robust across both small and very large Depot instances.
    """

    def __init__(self, model, depots_gpt5_heuristic_14: bool = True):
        super().__init__(model, name="DepotsGPT5Heuristic14")

        # Indices: predicate -> lookup helpers
        self.on_index: Dict[Tuple[str, str], int] = {}
        self.in_index: Dict[Tuple[str, str], int] = {}
        self.in_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        self.lifting_index: Dict[Tuple[str, str], int] = {}
        self.lifting_by_crate: Dict[str, List[Tuple[str, int]]] = {}
        self.clear_index: Dict[str, int] = {}
        self.available_index: Dict[str, int] = {}
        self.at_by_obj: Dict[str, List[Tuple[str, int]]] = {}

        # Parse and index all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                self.on_index[(c, s)] = fact.global_id
            elif pred == 'in' and len(args) == 2:
                c, t = args[0], args[1]
                self.in_index[(c, t)] = fact.global_id
                self.in_by_crate.setdefault(c, []).append((t, fact.global_id))
            elif pred == 'lifting' and len(args) == 2:
                h, c = args[0], args[1]
                self.lifting_index[(h, c)] = fact.global_id
                self.lifting_by_crate.setdefault(c, []).append((h, fact.global_id))
            elif pred == 'clear' and len(args) == 1:
                self.clear_index[args[0]] = fact.global_id
            elif pred == 'available' and len(args) == 1:
                self.available_index[args[0]] = fact.global_id
            elif pred == 'at' and len(args) == 2:
                obj, place = args[0], args[1]
                self.at_by_obj.setdefault(obj, []).append((place, fact.global_id))

        # Extract goal on-facts once (as pairs (crate, surface))
        self.goal_on_pairs: List[Tuple[str, str]] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
            for g in goal_fact_names:
                pred, args = self._parse_fact_name(g)
                if pred == 'on' and len(args) == 2:
                    self.goal_on_pairs.append((args[0], args[1]))
        except Exception:
            # If goals are not available for any reason, keep empty (heuristic then uses only task-network bound)
            self.goal_on_pairs = []

    # ---------------------------- Public API ---------------------------- #
    def __call__(self, state: int, task_network: List[Any] = None) -> int:
        if task_network is None:
            task_network = []

        # 1) Goal-based lower bound (sum of crate-specific actions excluding drives)
        goals_lb = 0
        for (c, s2) in self.goal_on_pairs:
            goals_lb += self._per_goal_lower_bound(state, c, s2)

        # 2) Task-network lower bound
        tn_lb = 0
        for comp in task_network:
            tn_lb += self._task_component_lb(state, comp)

        # 3) Combine admissibly
        return max(goals_lb, tn_lb)

    def compute(self, state: int, task_network: List[Any]) -> int:
        return self.__call__(state, task_network)

    # ---------------------------- Core helpers ---------------------------- #
    def _per_goal_lower_bound(self, state: int, crate: str, surface: str) -> int:
        """
        Per-goal lower bound counting only crate-specific actions (no drives), ensuring
        admissibility when summing across multiple goals:
        - If on(crate, surface): 0
        - If lifting(h, crate):
            * if place(h) == place(surface): 1 (Drop)
            * else: 3 (Load + Unload + Drop), excluding drives
            * if places unknown: optimistic 1
        - Else if in(crate, truck): 2 (Unload + Drop), independent of place for LB
        - Else (crate resting somewhere):
            * if place(crate) == place(surface): 2 (Lift + Drop)
            * else: 4 (Lift + Load + Unload + Drop), excluding drives
            * if places unknown: optimistic 2
        """
        # Goal already satisfied?
        on_id = self.on_index.get((crate, surface))
        if on_id is not None and self._is_true(state, on_id):
            return 0

        # Is crate currently lifted?
        lifted = self._find_lifting_hoist(state, crate)
        if lifted is not None:
            h = lifted
            p_h = self._place_of(state, h)
            p_s = self._place_of(state, surface)
            if p_h is not None and p_s is not None:
                return 1 if p_h == p_s else 3
            # Unknown placement -> take optimistic minimal LB
            return 1

        # Is crate in a truck?
        in_info = self._find_in_truck(state, crate)
        if in_info is not None:
            # Unload + Drop
            return 2

        # Otherwise, crate is at some place (on a support or just at a place)
        p_c = self._place_of(state, crate)
        p_s = self._place_of(state, surface)
        if p_c is not None and p_s is not None:
            if p_c == p_s:
                return 2  # Lift + Drop
            else:
                return 4  # Lift + Load + Unload + Drop (excluding drives)
        # Unknown placement -> optimistic minimal LB when co-located
        return 2

    def _task_component_lb(self, state: int, comp: Any) -> int:
        """
        Lower bound contribution for a task-network component.
        - Primitive operator: at least 1 action.
        - AbstractTask: domain-dependent constants or dynamic checks when possible.
        """
        # Identify component type by duck typing
        cls_name = comp.__class__.__name__

        # Primitive operator
        if cls_name == 'Operator':
            return 1

        # Abstract task
        if cls_name == 'AbstractTask':
            name = getattr(comp, 'name', '')
            head, args = self._parse_task_name(name)
            head = head or ''

            # do_put_on(c, s): reuse per-goal LB if possible
            if 'do_put_on' in head:
                if len(args) >= 2:
                    c, s = args[0], args[1]
                    return self._per_goal_lower_bound(state, c, s)
                return 1  # generic LB: needs at least the final Drop

            # do_load_truck(c, s, p, t): Lift + Load in the best case
            if 'do_load_truck' in head:
                return 2

            # do_unload_truck(c, s, p, t): Unload + Drop in the best case
            if 'do_unload_truck' in head:
                return 2

            # do_lift_crate(c, p, h): at least one Lift action in the best case
            if 'do_lift_crate' in head:
                return 1

            # do_get_truck(t, p): at least one Drive unless the truck is already there
            if 'do_get_truck' in head:
                if len(args) >= 2:
                    t, p = args[0], args[1]
                    at_ids = self.at_by_obj.get(t, [])
                    for place, fid in at_ids:
                        if place == p and self._is_true(state, fid):
                            return 0
                return 1

            # do_clear(s, p): 0 if already clear, else at least 1 Lift
            if 'do_clear' in head:
                if len(args) >= 1:
                    s = args[0]
                    clear_id = self.clear_index.get(s)
                    if clear_id is not None and self._is_true(state, clear_id):
                        return 0
                return 1

            # Unknown abstract task -> conservative minimal contribution
            return 0

        # Decomposition nodes should not be directly in the current task network in most configurations;
        # return 0 if encountered.
        return 0

    # ---------------------------- Parsing helpers ---------------------------- #
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse facts with format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        Returns (predicate, [args...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse operator/task-like strings, trying formats like:
          head(arg1,arg2,...)  or  head[arg1,arg2,...]
        Returns (head, [args...]) or (None, []). Robust against missing args.
        """
        if not name:
            return None, []
        s = name.strip()
        for lbr, rbr in [("(", ")"), ("[", "]")]:
            li = s.find(lbr)
            ri = s.rfind(rbr)
            if li != -1 and ri != -1 and ri > li:
                head = s[:li].strip()
                inner = s[li + 1:ri].strip()
                if not inner:
                    return head, []
                # split by comma, else whitespace
                args = [a.strip() for a in inner.split(",")]
                if len(args) == 1 and ' ' in args[0]:
                    args = [a.strip() for a in args[0].split() if a.strip()]
                return head, args
        # No brackets/parentheses detected -> treat whole as head
        return s, []

    # ---------------------------- State query helpers ---------------------------- #
    @staticmethod
    def _is_true(state_bitwise: int, fact_global_id: int) -> bool:
        return (state_bitwise >> fact_global_id) & 1 == 1

    def _place_of(self, state: int, obj: str) -> Optional[str]:
        """
        Returns the place name where obj is currently located (from +at[obj,place])
        or None if unknown/not present in the state (e.g., being lifted).
        """
        for place, fid in self.at_by_obj.get(obj, []):
            if self._is_true(state, fid):
                return place
        return None

    def _find_lifting_hoist(self, state: int, crate: str) -> Optional[str]:
        """
        If some hoist is lifting the given crate, return that hoist's name, else None.
        """
        for h, fid in self.lifting_by_crate.get(crate, []):
            if self._is_true(state, fid):
                return h
        return None

    def _find_in_truck(self, state: int, crate: str) -> Optional[str]:
        """
        If the crate is in some truck, return that truck's name, else None.
        """
        for t, fid in self.in_by_crate.get(crate, []):
            if self._is_true(state, fid):
                return t
        return None

