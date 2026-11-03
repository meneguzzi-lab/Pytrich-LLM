from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic6(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale (domain insights -> heuristic design)
    ---------------------------------------------
    The Depots domain moves crates between stacks/pallets/trucks using hoists and trucks.
    The HTN top-level tasks in the benchmark are typically many `(do_put_on crateX surfaceY)`.

    Key structural challenges:
    1) **Stack clearing dominates cost**: to move a crate you often must clear the crate itself
       (make it `clear`) and clear the destination surface (`clear dest`). Clearing recursively
       requires lifting/loading blocking crates.
    2) **Cross-location moves require logistics**: if crate and destination are at different places,
       the domain forces a load + drive + unload sequence.
    3) **Hoist/truck availability is local**: lifting requires a hoist at the place; loading requires
       a truck at the place; unloading requires truck and hoist co-located.

    Heuristic strategy:
    - We compute a *fast lower bound* per unsatisfied goal `+on[c,s]` by estimating the minimum
      number of primitive actions still needed under a relaxed view.
    - We include:
        (a) minimal actions to create `on[c,s]` from current mode (already on, lifting, in-truck,
            on-other-surface, just at-place),
        (b) a lightweight estimate of extra actions needed to clear the moving crate and/or target.
    - We use only **bitwise checks** against facts for speed.

    This heuristic is **inadmissibility-safe?**
    - It is intended as a *lower bound* by using optimistic assumptions (e.g., existence of a hoist
      and truck, ignoring competition, choosing cheapest case). It can still overcount if we
      double-charge shared clearing across multiple goals; to avoid that, we count clearing costs
      with conservative min(1, ...) style terms and do not recurse deeply.

    Performance considerations:
    - Pre-index facts by predicate and argument positions.
    - For clearing, use precomputed adjacency lists from `on[crate, surface]` facts.
    - Do not call `state_explicit_repr(state)` during search.
    """

    def __init__(self, model, depots_gpt52_heuristic_6: str = "depots_gpt52_heuristic_6"):
        super().__init__(model, depots_gpt52_heuristic_6)
        self.model = model

        # --- Parse and index all facts ---
        self._facts_by_pred: Dict[str, List[Tuple[int, List[str]]]] = {}
        # Useful direct indexes
        self._on_facts: List[Tuple[int, str, str]] = []          # (gid, crate, surface)
        self._at_facts: List[Tuple[int, str, str]] = []          # (gid, obj, place)
        self._in_facts: List[Tuple[int, str, str]] = []          # (gid, crate, truck)
        self._lifting_facts: List[Tuple[int, str, str]] = []     # (gid, hoist, crate)
        self._clear_facts: List[Tuple[int, str]] = []            # (gid, surface)
        self._available_facts: List[Tuple[int, str]] = []        # (gid, hoist)

        # Map for fast lookups of possible relations (static lists of fact ids)
        self._on_by_crate: Dict[str, List[Tuple[int, str]]] = {}
        self._on_by_surface: Dict[str, List[Tuple[int, str]]] = {}
        self._at_by_obj: Dict[str, List[Tuple[int, str]]] = {}
        self._in_by_crate: Dict[str, List[Tuple[int, str]]] = {}
        self._lifting_by_crate: Dict[str, List[Tuple[int, str]]] = {}

        # Collect all hoists/trucks (from facts only; sufficient for heuristic existence checks)
        self._all_hoists: Set[str] = set()
        self._all_trucks: Set[str] = set()

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((fact.global_id, args))

            if pred == "on" and len(args) == 2:
                c, s = args
                self._on_facts.append((fact.global_id, c, s))
                self._on_by_crate.setdefault(c, []).append((fact.global_id, s))
                self._on_by_surface.setdefault(s, []).append((fact.global_id, c))

            elif pred == "at" and len(args) == 2:
                o, p = args
                self._at_facts.append((fact.global_id, o, p))
                self._at_by_obj.setdefault(o, []).append((fact.global_id, p))

            elif pred == "in" and len(args) == 2:
                c, t = args
                self._in_facts.append((fact.global_id, c, t))
                self._in_by_crate.setdefault(c, []).append((fact.global_id, t))
                self._all_trucks.add(t)

            elif pred == "lifting" and len(args) == 2:
                h, c = args
                self._lifting_facts.append((fact.global_id, h, c))
                self._lifting_by_crate.setdefault(c, []).append((fact.global_id, h))
                self._all_hoists.add(h)

            elif pred == "clear" and len(args) == 1:
                s = args[0]
                self._clear_facts.append((fact.global_id, s))

            elif pred == "available" and len(args) == 1:
                h = args[0]
                self._available_facts.append((fact.global_id, h))
                self._all_hoists.add(h)

        # --- Precompute goal on-facts ---
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_on: List[Tuple[str, str, Optional[int]]] = []  # (crate, surface, gid_of_goal_fact)

        # Build a map from (pred,args_tuple) to global_id for fast checking of goal facts
        self._fact_gid_by_key: Dict[Tuple[str, Tuple[str, ...], str], int] = {}
        # key includes sign '+'/'-' because facts are signed in name.
        for fact in self.model.facts:
            sign = fact.name.strip()[0] if fact.name and fact.name[0] in "+-" else "+"
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            self._fact_gid_by_key[(pred, tuple(args), sign)] = fact.global_id

        for fn in goal_fact_names:
            sign = fn.strip()[0] if fn and fn[0] in "+-" else "+"
            pred, args = self._parse_fact_name(fn)
            if sign == "+" and pred == "on" and len(args) == 2:
                c, s = args
                gid = self._fact_gid_by_key.get(("on", (c, s), "+"), None)
                self._goal_on.append((c, s, gid))

        # If there are no explicit on-goals, fallback to generic unsatisfied goal count.
        self._has_on_goals = len(self._goal_on) > 0

    # ------------------------ Core heuristic API ------------------------

    def __call__(self, node) -> int:
        state = node.state

        # If goal reached in terms of state facts, return 0 (HTN task network completion
        # is handled by the planner; this still helps as a heuristic).
        if (state & self.model.goals) == self.model.goals:
            return 0

        # Primary: sum of per-goal optimistic costs for +on goals.
        if self._has_on_goals:
            h = 0
            for c, s, gid in self._goal_on:
                # If goal fact id is known, use bitwise check; else reconstruct via on relation.
                if gid is not None and ((state >> gid) & 1):
                    continue
                # If unknown gid (should be rare), check via any matching on fact.
                if gid is None and self._is_on(state, c, s):
                    continue

                h += self._estimate_put_on_cost(state, c, s)

            # Small strengthening: also count any other unsatisfied goal facts (non-on) as 1.
            # This is conservative and helps if instances include other goals.
            other_missing = self._count_other_missing_goals(state)
            return h + other_missing

        # Fallback: count missing goal facts.
        return self._count_missing_goals(state)

    # ------------------------ Estimation components ------------------------

    def _estimate_put_on_cost(self, state: int, crate: str, dest: str) -> int:
        """Optimistic lower bound for achieving on[crate,dest]."""

        # If already lifting crate: only need Drop provided dest clear (ignoring preconditions).
        if self._is_lifting_any(state, crate):
            base = 1  # Drop
            base += self._need_clear_surface_lb(state, dest)
            base += self._need_at_same_place_lb(state, crate, dest)  # optimistic 0/1
            return base

        # If crate is in some truck: Unload + Drop (and ensure dest clear), plus maybe drive.
        if self._is_in_any_truck(state, crate):
            # Minimal: Unload + Drop = 2
            base = 2
            base += self._need_clear_surface_lb(state, dest)
            base += self._need_drive_to_dest_place_lb(state, crate, dest)
            return base

        # If crate is on some surface currently (typical): Lift + Drop, plus clearing and maybe transport.
        if self._is_on_any_surface(state, crate):
            base = 2  # Lift + Drop
            base += self._need_clear_crate_lb(state, crate)
            base += self._need_clear_surface_lb(state, dest)
            base += self._need_transport_between_places_lb(state, crate, dest)
            return base

        # If crate is just at a place (not on, not in, not lifting): we still need to Lift from some surface
        # but such a state is uncommon; optimistic assume just Drop after being lifted via 1 action.
        base = 2
        base += self._need_clear_surface_lb(state, dest)
        base += self._need_transport_between_places_lb(state, crate, dest)
        return base

    # ------------------------ Lower-bound helpers (0/1/2 etc.) ------------------------

    def _need_clear_crate_lb(self, state: int, crate: str) -> int:
        """Return 0 if crate is clear, else 1.

        Being non-clear implies something on top; at least one Lift+Load cycle is needed to clear it.
        We use 1 as a conservative action lower bound contribution (not full cost) to avoid
        overcounting across goals.
        """
        gid = self._fact_gid_by_key.get(("clear", (crate,), "+"), None)
        if gid is not None:
            return 0 if ((state >> gid) & 1) else 1
        # If no clear fact for this object exists, return 0 (optimistic).
        return 0

    def _need_clear_surface_lb(self, state: int, surface: str) -> int:
        """Return 0 if surface is clear, else 1.

        To drop onto a surface, it must be clear.
        """
        gid = self._fact_gid_by_key.get(("clear", (surface,), "+"), None)
        if gid is not None:
            return 0 if ((state >> gid) & 1) else 1
        return 0

    def _need_at_same_place_lb(self, state: int, obj1: str, obj2: str) -> int:
        """0 if we can prove they are at same place now, else 1 (optimistic).

        Drop requires both hoist and destination surface at place; crate must also be at that place
        (or lifting at that place). We approximate with a weak 0/1.
        """
        p1 = self._get_any_place_of(state, obj1)
        p2 = self._get_any_place_of(state, obj2)
        if p1 is not None and p2 is not None and p1 == p2:
            return 0
        return 1

    def _need_transport_between_places_lb(self, state: int, crate: str, dest: str) -> int:
        """Lower bound on needing truck transport.

        If crate and dest are known at different places => at least 3 actions: Load, Drive, Unload.
        We return 0 if same place known, else 1 if unknown/different (weak) to remain conservative.
        """
        pc = self._get_any_place_of(state, crate)
        pd = self._get_any_place_of(state, dest)
        if pc is not None and pd is not None:
            return 0 if pc == pd else 1
        return 0

    def _need_drive_to_dest_place_lb(self, state: int, crate: str, dest: str) -> int:
        """If crate is in truck, we might need to drive that truck to dest place.

        Weak LB: if truck place known and dest place known and differ => 1 else 0.
        """
        # Find a truck containing crate (any one)
        t = self._get_any_truck_of(state, crate)
        if t is None:
            return 0
        pt = self._get_any_place_of(state, t)
        pd = self._get_any_place_of(state, dest)
        if pt is not None and pd is not None and pt != pd:
            return 1
        return 0

    # ------------------------ State query (bitwise) ------------------------

    def _is_lifting_any(self, state: int, crate: str) -> bool:
        for gid, _h in self._lifting_by_crate.get(crate, []):
            if (state >> gid) & 1:
                return True
        return False

    def _is_in_any_truck(self, state: int, crate: str) -> bool:
        for gid, _t in self._in_by_crate.get(crate, []):
            if (state >> gid) & 1:
                return True
        return False

    def _is_on_any_surface(self, state: int, crate: str) -> bool:
        for gid, _s in self._on_by_crate.get(crate, []):
            if (state >> gid) & 1:
                return True
        return False

    def _is_on(self, state: int, crate: str, surface: str) -> bool:
        gid = self._fact_gid_by_key.get(("on", (crate, surface), "+"), None)
        if gid is None:
            return False
        return bool((state >> gid) & 1)

    def _get_any_place_of(self, state: int, obj: str) -> Optional[str]:
        # Direct at[obj,place]
        for gid, p in self._at_by_obj.get(obj, []):
            if (state >> gid) & 1:
                return p

        # If obj is a crate and is on some surface, infer place from at[surface,place]
        for on_gid, surf in self._on_by_crate.get(obj, []):
            if (state >> on_gid) & 1:
                for at_gid, p in self._at_by_obj.get(surf, []):
                    if (state >> at_gid) & 1:
                        return p

        # If obj is a crate and is in some truck, infer place from at[truck,place]
        for in_gid, t in self._in_by_crate.get(obj, []):
            if (state >> in_gid) & 1:
                for at_gid, p in self._at_by_obj.get(t, []):
                    if (state >> at_gid) & 1:
                        return p

        return None

    def _get_any_truck_of(self, state: int, crate: str) -> Optional[str]:
        for gid, t in self._in_by_crate.get(crate, []):
            if (state >> gid) & 1:
                return t
        return None

    # ------------------------ Goal counting helpers ------------------------

    def _count_missing_goals(self, state: int) -> int:
        missing_bits = self.model.goals & ~state
        # popcount
        return missing_bits.bit_count()

    def _count_other_missing_goals(self, state: int) -> int:
        """Count missing goal facts that are not +on[*,*]."""
        missing_bits = self.model.goals & ~state
        if missing_bits == 0:
            return 0

        count = 0
        # Iterate over set bits in missing_bits
        mb = missing_bits
        while mb:
            lsb = mb & -mb
            gid = (lsb.bit_length() - 1)
            fact = self.model.facts[gid]
            pred, args = self._parse_fact_name(fact.name)
            sign = fact.name.strip()[0] if fact.name and fact.name[0] in "+-" else "+"
            if not (sign == "+" and pred == "on" and len(args) == 2):
                count += 1
            mb ^= lsb
        return count

    # ------------------------ Fact name parsing ------------------------

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
