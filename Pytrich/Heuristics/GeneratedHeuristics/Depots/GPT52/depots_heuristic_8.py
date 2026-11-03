from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic8(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Key domain insights
    -------------------
    Depots is a logistics + stacking domain with these relevant constraints:

    * A crate can be:
        - on a surface (pallet or crate)
        - in a truck
        - being lifted (lifting hoist crate)
      The goal we care about in provided instances is almost exclusively `on(c, s)`.

    * Achieving `on(c, s)` typically requires:
        - both crate and target surface at same place
        - target surface must be clear
        - crate must be clear to lift
        - a hoist at that place
        - if crate not at place, it must travel via truck (load/drive/unload)

    Heuristic design
    ----------------
    We compute a fast lower bound by summing (with max-aggregation to stay safe)
    per-goal estimates that reflect necessary *action types*:

    * If goal `on(c, s)` already true -> 0.
    * Otherwise, at least one `Drop` is needed to make `on(c,s)` true.
    * If crate is currently `in` a truck -> an `Unload` is necessary before `Drop`.
    * If crate is currently `on` some surface (or equivalently `at(c,p)` holds):
        - if crate not clear -> must clear the crate (move something off it) => >= 1
        - if target surface not clear -> must clear the target => >= 1
        - if crate and target surface not co-located -> must move either crate or target.
          Moving a surface (pallet) is impossible; moving a crate requires truck travel.
          We lower-bound this by requiring at least one `Drive` if places differ.
          (In a fully-connected map, min drive count is 1 when different.)
        - If crate at place but target at different -> must at least Load+Drive+Unload
          in addition to Drop.

    Aggregation across multiple `on` goals:
    * We use `max` over goal estimates rather than sum, because many goals can share
      the same driving/clearing actions; summing would wildly overestimate and harm
      admissibility. `max` is a safe lower bound on the total remaining cost.
    * Additionally, we add a small extra bound based on the count of unsatisfied `on`
      goals: at least one `Drop` per *new* `on` fact, but again shared actions exist.
      To remain admissible we cannot sum Drops across all goals; however, we can use
      a *unit* lower bound: if there is at least one unsatisfied on-goal, cost >= 1.

    This heuristic is intentionally simple and bitwise-fast to scale to very large
    instances (hundreds of crates).
    """

    def __init__(self, model, depots_gpt52_heuristic_8: str = "depots_gpt52_heuristic_8"):
        super().__init__(model, depots_gpt52_heuristic_8)
        self.model = model

        # Cache goal facts as parsed structures
        self._goal_on: List[Tuple[int, str, str]] = []  # (fact_gid, crate, surface)

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                gid = self._fact_gid_by_name(name)
                if gid is not None:
                    self._goal_on.append((gid, args[0], args[1]))

        # Precompute indices for fast lookup from state bits
        # For each object, we will need to find:
        #   location of locatables: at[x,p]
        #   support relation for crates: on[c,s]
        #   in relation for crates: in[c,t]
        #   clear status: clear[x]
        # We build dictionaries mapping keys -> list of fact global_ids to query.
        self._at_facts_by_obj: Dict[str, List[Tuple[int, str]]] = {}  # obj -> [(gid, place)]
        self._on_facts_by_crate: Dict[str, List[Tuple[int, str]]] = {}  # crate -> [(gid, surface)]
        self._in_facts_by_crate: Dict[str, List[Tuple[int, str]]] = {}  # crate -> [(gid, truck)]
        self._clear_fact_by_surface: Dict[str, int] = {}  # surface -> gid

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at" and len(args) == 2:
                obj, place = args
                self._at_facts_by_obj.setdefault(obj, []).append((f.global_id, place))
            elif pred == "on" and len(args) == 2:
                c, s = args
                self._on_facts_by_crate.setdefault(c, []).append((f.global_id, s))
            elif pred == "in" and len(args) == 2:
                c, t = args
                self._in_facts_by_crate.setdefault(c, []).append((f.global_id, t))
            elif pred == "clear" and len(args) == 1:
                self._clear_fact_by_surface[args[0]] = f.global_id

    # --- Core heuristic interface ---
    def __call__(self, parent_node, node) -> int:
        state = node.state

        # If no relevant on-goals found, return 0.
        if not self._goal_on:
            return 0

        max_goal_lb = 0
        any_unsat = False

        # Compute a per-goal lower bound; take max across goals (safe, informative).
        for (on_gid, c, s) in self._goal_on:
            if ((state >> on_gid) & 1) == 1:
                continue
            any_unsat = True
            lb = self._lb_to_achieve_on(state, c, s)
            if lb > max_goal_lb:
                max_goal_lb = lb

        # A tiny extra bound: if at least one on-goal is unsatisfied, at least one
        # primitive step remains.
        if any_unsat:
            return max(1, max_goal_lb)
        return 0

    # --- Lower bound computation for one goal on(c,s) ---
    def _lb_to_achieve_on(self, state: int, crate: str, surface: str) -> int:
        """Admissible, domain-informed lower bound for achieving on(crate,surface)."""

        # Need a Drop to make on(crate,surface) true (unless already true).
        lb = 1

        # Determine where crate is / how it's held.
        crate_in_truck = self._get_true_arg(state, self._in_facts_by_crate.get(crate, []))
        crate_place = self._get_obj_place(state, crate)
        surface_place = self._get_obj_place(state, surface)

        # If crate in truck -> must Unload before Drop.
        if crate_in_truck is not None:
            lb += 1  # Unload
            # Also ensure truck is (or will be) at the target place; if we know target place
            # and can locate the truck, then if different -> need a Drive.
            # If we cannot determine places, we don't add anything.
            truck_place = self._get_obj_place(state, crate_in_truck)
            if surface_place is not None and truck_place is not None and surface_place != truck_place:
                lb += 1  # at least one Drive
            # Target must be clear to Drop.
            if not self._is_clear(state, surface):
                lb += 1
            return lb

        # Otherwise, crate is not in truck. It might be on some surface at a place.
        # If crate place and surface place both known and differ, we need to transport crate.
        if crate_place is not None and surface_place is not None and crate_place != surface_place:
            # Minimal transport pattern: Load + Drive + Unload in addition to Drop.
            lb += 3

        # Preconditions to Lift/Drop require crate clear and target clear.
        if not self._is_clear(state, crate):
            lb += 1
        if not self._is_clear(state, surface):
            lb += 1

        return lb

    # --- Bitwise helpers ---
    def _is_clear(self, state: int, surface: str) -> bool:
        gid = self._clear_fact_by_surface.get(surface)
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _get_obj_place(self, state: int, obj: str) -> Optional[str]:
        """Return the unique place p such that at(obj,p) is true, else None."""
        candidates = self._at_facts_by_obj.get(obj)
        if not candidates:
            return None
        for gid, place in candidates:
            if ((state >> gid) & 1) == 1:
                return place
        return None

    def _get_true_arg(self, state: int, gid_arg_list: List[Tuple[int, str]]) -> Optional[str]:
        """From [(gid,arg)], return arg for the first true gid in state."""
        for gid, arg in gid_arg_list:
            if ((state >> gid) & 1) == 1:
                return arg
        return None

    # --- Fact parsing and lookup ---
    def _parse_fact_name(self, name: str):
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

    def _fact_gid_by_name(self, fact_name: str) -> Optional[int]:
        # Linear scan over facts is OK during init only.
        # (Runtime must avoid explicit repr scans; we do.)
        for f in self.model.facts:
            if f.name == fact_name:
                return f.global_id
        return None
