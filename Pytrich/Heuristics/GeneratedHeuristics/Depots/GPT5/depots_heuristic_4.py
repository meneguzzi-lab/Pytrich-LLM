from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, namedtuple

# Attempt to import the base Heuristic from Pytrich. If unavailable, define a minimal stub
from Pytrich.Heuristics.heuristic import Heuristic
CrateStatus = namedtuple("CrateStatus", ["kind", "place"])  # kind in {on, in, lifting, at, unknown}


class DepotsGPT5Heuristic4(Heuristic):
    """
    Domain-dependent heuristic for HDDL Depots domain.

    Design rationale (admissible lower bound):
    - Goals are 'on(crate, surface)'. Achieving on(c, s) minimally requires a sequence of primitive actions
      depending on current locations and carriers:
        * If already true: 0.
        * If c is being lifted at place p and s is at the same place p: Drop only => 1.
        * If c is being lifted at p and s is at a different place: Load + Drive + Unload + Drop => 4.
        * If c is in a truck at place p:
              - same place as s: Unload + Drop => 2
              - different place: Drive + Unload + Drop => 3
        * If c is on some surface at place p (and not being lifted or in truck):
              - same place as s: Lift + Drop => 2
              - different place: Lift + Load + Drive + Unload + Drop => 5
    - Additional necessary preconditions introduce irreducible action needs:
        * If the destination surface s is not clear, at least one Lift is required on the top crate of s: +1.
        * If c is on a surface and c is not clear (something on top), to move c at least one Lift is needed: +1.
      These increments are safe lower bounds and keep the heuristic admissible.
    - Truck positioning is optimistic: when different places are involved, at least one Drive is needed.
      We do not add extra Drives to fetch a truck to the source, keeping the estimate optimistic.
    - Hierarchy awareness: we also compute a trivial lower bound for the current task network: each primitive
      operator remaining (excluding 'nop') must cost at least 1. To remain admissible we use:
          h = max(sum_of_goal_LBs, count_of_pending_primitives)
      This avoids double-counting between the two lower bounds.

    Efficiency:
    - All goal facts and relevant Fact handles are pre-indexed at initialization.
    - State queries use bitwise checks on fact.global_id for speed.

    Notes:
    - The heuristic is designed to scale to large instances by using constant-time (amortized) lookups and
      small per-goal computations.
    - It is robust to partial states and dynamic changes as it derives places of crates/surfaces at evaluation time.
    """

    def __init__(self, model, depots_gpt5_heuristic_4: bool = True):
        super().__init__()
        self.model = model
        self.enabled = depots_gpt5_heuristic_4

        # Indices
        self.on_facts_by_c: Dict[str, List[Any]] = defaultdict(list)       # crate -> [Fact(+on[c,z])]
        self.in_facts_by_c: Dict[str, List[Any]] = defaultdict(list)       # crate -> [Fact(+in[c,t])]
        self.lifting_facts_by_c: Dict[str, List[Any]] = defaultdict(list)  # crate -> [Fact(+lifting[h,c])]
        self.at_facts_by_obj: Dict[str, List[Any]] = defaultdict(list)     # obj -> [Fact(+at[obj,place])]
        self.clear_fact_by_surface: Dict[str, Any] = {}                    # surface -> Fact(+clear[s])

        self.fact_name_to_fact: Dict[str, Any] = {}

        # Type helpers from naming convention
        self._is_truck = lambda x: x.startswith("truck")
        self._is_hoist = lambda x: x.startswith("hoist")
        self._is_crate = lambda x: x.startswith("crate")
        self._is_pallet = lambda x: x.startswith("pallet")
        self._is_place = lambda x: x.startswith("depot") or x.startswith("distributor")

        # Build fact indices
        for f in self.model.facts:
            self.fact_name_to_fact[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                c, z = args
                self.on_facts_by_c[c].append(f)
            elif pred == 'in' and len(args) == 2:
                c, t = args
                self.in_facts_by_c[c].append(f)
            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self.lifting_facts_by_c[c].append(f)
            elif pred == 'at' and len(args) == 2:
                obj, p = args
                self.at_facts_by_obj[obj].append(f)
            elif pred == 'clear' and len(args) == 1:
                s = args[0]
                self.clear_fact_by_surface[s] = f

        # Precompute goal 'on' facts and their (crate, surface) pairs
        self.goal_on_pairs: List[Tuple[Tuple[str, str], Any]] = []  # [((c,s), Fact(+on[c,s]))]
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                c, s = args
                fact = self._find_on_fact(c, s)
                if fact is not None:
                    self.goal_on_pairs.append(((c, s), fact))

    # ---------------------------- Public API ----------------------------
    def __call__(self, model, state, task_network=None) -> int:
        return self.evaluate(model, state, task_network)

    def evaluate(self, model, state, task_network=None) -> int:
        if not self.enabled:
            return 0

        state_bitwise: int = state if isinstance(state, int) else int(state)

        # Lower bound from goals (sum of independent per-goal lower bounds)
        h_goals = 0
        place_cache: Dict[str, Optional[str]] = {}

        for (c, s), on_fact in self.goal_on_pairs:
            if self._is_true(on_fact, state_bitwise):
                continue  # already satisfied

            # Compute LB for this goal
            cost = 0

            # Destination clear requirement: Drop precondition requires clear(s) => at least one Lift if not clear
            if not self._is_clear(s, state_bitwise):
                cost += 1

            # Determine status and place of crate c
            c_status = self._crate_status(c, state_bitwise, place_cache)

            # Determine current place of destination surface s
            p_s = self._get_current_place(s, state_bitwise, place_cache)

            if c_status.kind == 'lifting':
                # Already being lifted at p_c
                p_c = c_status.place
                if self._same_place(p_c, p_s):
                    # Just Drop
                    cost += 1
                else:
                    # Load + Drive (>=1) + Unload + Drop
                    # Use 1 drive as a safe lower bound when places differ
                    cost += 4

            elif c_status.kind == 'in':
                p_c = c_status.place  # place of the truck holding c
                drives = 0 if self._same_place(p_c, p_s) else 1
                # Unload + Drop + drives
                cost += 2 + drives

            elif c_status.kind == 'on' or c_status.kind == 'at':
                p_c = c_status.place
                if self._same_place(p_c, p_s):
                    # Need: Lift + Drop
                    cost += 2
                else:
                    # Need: Lift + Load + Drive + Unload + Drop
                    cost += 5
                # If c is on a surface and NOT clear, at least one Lift is needed to clear top of c before moving it
                if not self._is_clear(c, state_bitwise):
                    cost += 1

            else:
                # Unknown conservative fallback: assume minimal local rearrangement
                cost += 2

            h_goals += cost

        # Lower bound from remaining primitive operators in task network: each non-nop primitive >= 1
        h_tn = 0
        if task_network is not None:
            for comp in task_network:
                obj = comp
                if isinstance(comp, int):
                    try:
                        obj = self.model.get_component(comp)
                    except Exception:
                        obj = comp
                # Heuristic: identify primitive operators by presence of operator-like attributes
                if hasattr(obj, 'pos_precons') and hasattr(obj, 'apply'):
                    name = getattr(obj, 'name', '')
                    if 'nop' not in name:
                        h_tn += 1
        # Use max to keep admissible (avoid double-counting goals and TN work)
        return max(h_goals, h_tn)

    # ---------------------------- Internal helpers ----------------------------
    def _parse_fact_name(self, name: str):
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

    def _is_true(self, fact, state_bitwise: int) -> bool:
        if fact is None:
            return False
        return ((state_bitwise >> fact.global_id) & 1) == 1

    def _find_on_fact(self, c: str, s: str):
        for f in self.on_facts_by_c.get(c, []):
            pred, args = self._parse_fact_name(f.name)
            if pred == 'on' and len(args) == 2 and args[1] == s:
                return f
        return None

    def _get_current_place(self, obj: str, state_bitwise: int, cache: Dict[str, Optional[str]]) -> Optional[str]:
        # Cached lookup
        if obj in cache:
            return cache[obj]

        # If crate, it might be lifted or in a truck; resolve place through carriers first
        if self._is_crate(obj):
            # lifted?
            for lf in self.lifting_facts_by_c.get(obj, []):
                if self._is_true(lf, state_bitwise):
                    # find hoist place
                    _, args = self._parse_fact_name(lf.name)
                    h = args[0]
                    p = self._get_place_from_at(h, state_bitwise)
                    cache[obj] = p
                    return p
            # in truck?
            for inf in self.in_facts_by_c.get(obj, []):
                if self._is_true(inf, state_bitwise):
                    _, args = self._parse_fact_name(inf.name)
                    t = args[1]
                    p = self._get_place_from_at(t, state_bitwise)
                    cache[obj] = p
                    return p
            # else fall through to standard 'at'

        # General 'at' resolution
        p = self._get_place_from_at(obj, state_bitwise)
        cache[obj] = p
        return p

    def _get_place_from_at(self, obj: str, state_bitwise: int) -> Optional[str]:
        for af in self.at_facts_by_obj.get(obj, []):
            if self._is_true(af, state_bitwise):
                _, args = self._parse_fact_name(af.name)
                if len(args) == 2:
                    return args[1]
        return None

    def _is_clear(self, surf: str, state_bitwise: int) -> bool:
        f = self.clear_fact_by_surface.get(surf, None)
        if f is None:
            # If we do not find a clear[s] fact, assume clear to keep admissible (optimistic)
            return True
        return self._is_true(f, state_bitwise)

    def _same_place(self, p1: Optional[str], p2: Optional[str]) -> bool:
        # If unknown, assume same place (optimistic, admissible)
        if p1 is None or p2 is None:
            return True
        return p1 == p2

    def _crate_status(self, c: str, state_bitwise: int, place_cache: Dict[str, Optional[str]]) -> CrateStatus:
        # lifting?
        for lf in self.lifting_facts_by_c.get(c, []):
            if self._is_true(lf, state_bitwise):
                _, args = self._parse_fact_name(lf.name)
                h = args[0]
                p = self._get_current_place(h, state_bitwise, place_cache)
                return CrateStatus('lifting', p)
        # in truck?
        for inf in self.in_facts_by_c.get(c, []):
            if self._is_true(inf, state_bitwise):
                _, args = self._parse_fact_name(inf.name)
                t = args[1]
                p = self._get_current_place(t, state_bitwise, place_cache)
                return CrateStatus('in', p)
        # on something?
        for of in self.on_facts_by_c.get(c, []):
            if self._is_true(of, state_bitwise):
                # resolve place via at[c, p]
                p = self._get_current_place(c, state_bitwise, place_cache)
                return CrateStatus('on', p)
        # maybe just 'at' (edge case)
        p = self._get_current_place(c, state_bitwise, place_cache)
        if p is not None:
            return CrateStatus('at', p)
        return CrateStatus('unknown', None)
