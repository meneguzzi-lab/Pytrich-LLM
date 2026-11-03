from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

# Try to import the planner's base Heuristic; fall back to a minimal stub for static checks
from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic15(Heuristic):
    """
    Domain-dependent heuristic for the Depot (Depots) HDDL domain.

    Key ideas (admissible lower-bound, domain-specific):
    - Every goal on(c, s) ultimately requires a Drop action (unless already satisfied).
    - If crate c is not currently being carried (lifting), it must be picked up first via Lift (if on something) or Unload (if in a truck).
    - If c and its target support s are at different places (depots/distributors), moving c across places minimally requires Load + Unload (= 2 actions). We ignore Drive to keep a safe (admissible) lower-bound that can't be shared across multiple goals.
    - We do not add costs for clearing the target support or source-stack interference (to avoid double counting across multiple goals). This keeps the heuristic admissible.
    - Costs are summed across goal on-facts. Each individual crate requires at least its own pickup/move/place actions; these cannot be shared across different crates.

    The heuristic estimates for each goal on(c, s):
      h_on(c, s) = 0, if already true
                 = 1 (Drop) + X (Lift or Unload) + Y (cross-place transfer), otherwise
      where
        X = 0 if c is already lifting
            1 if c is on a surface (needs Lift)
            1 if c is in a truck (needs Unload)
        Y = 0 if c is in a truck (we can unload at destination) or if c/s are already co-located
            2 if c and s are at different places and c is not in a truck (requires Load + Unload)

    Notes about implementation details:
    - We only use fast bitwise checks for state membership to avoid timeouts on large instances.
    - We pre-index facts by predicate and relevant arguments for O(1) access during heuristic evaluation.
    - We parse the bit-encoded goal state once in __init__ and only consider on-goals.
    - We determine an object's current place using facts available in the current state:
      * Direct at[x, p]
      * If a crate is in a truck: use truck's at
      * If a crate is on a surface: follow the support chain up to a pallet to get the place via at[pallet, p]
      * If a crate is lifting by a hoist: use the hoist's place
    """

    def __init__(self, model, depots_gpt5_heuristic_15: bool = True):
        super().__init__(model=model, name="DepotsGPT5Heuristic15")

        # Internal indices for fast lookups
        self.on_fact_id: Dict[Tuple[str, str], int] = {}
        self.on_by_crate: Dict[str, List[Tuple[str, int]]] = defaultdict(list)       # c -> List[(support, fid)]
        self.on_by_support: Dict[str, List[Tuple[str, int]]] = defaultdict(list)     # support -> List[(crate, fid)]
        self.in_by_crate: Dict[str, List[Tuple[str, int]]] = defaultdict(list)      # c -> List[(truck, fid)]
        self.lifting_by_crate: Dict[str, List[Tuple[str, int]]] = defaultdict(list) # c -> List[(hoist, fid)]
        self.at_by_obj: Dict[str, List[Tuple[str, int]]] = defaultdict(list)        # obj -> List[(place, fid)]
        self.clear_by_surface: Dict[str, int] = {}                                  # surface -> fid (not used in LB)
        self.available_by_hoist: Dict[str, int] = {}                                # hoist -> fid (not used in LB)

        # Sets to identify object kinds by name pattern (robust and cheap)
        self.pallets: Set[str] = set()
        self.crates: Set[str] = set()
        self.trucks: Set[str] = set()
        self.hoists: Set[str] = set()
        self.places: Set[str] = set()  # depots + distributors

        # Build indices over all facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == "on" and len(args) == 2:
                c, s = args
                self.on_fact_id[(c, s)] = fact.global_id
                self.on_by_crate[c].append((s, fact.global_id))
                self.on_by_support[s].append((c, fact.global_id))
                if c.startswith("crate"):
                    self.crates.add(c)
                # surface s could be pallet or crate
                if s.startswith("pallet"):
                    self.pallets.add(s)
                if s.startswith("crate"):
                    self.crates.add(s)
            elif pred == "in" and len(args) == 2:
                c, t = args
                self.in_by_crate[c].append((t, fact.global_id))
                if c.startswith("crate"):
                    self.crates.add(c)
                if t.startswith("truck"):
                    self.trucks.add(t)
            elif pred == "lifting" and len(args) == 2:
                h, c = args
                self.lifting_by_crate[c].append((h, fact.global_id))
                if c.startswith("crate"):
                    self.crates.add(c)
                if h.startswith("hoist"):
                    self.hoists.add(h)
            elif pred == "at" and len(args) == 2:
                x, p = args
                self.at_by_obj[x].append((p, fact.global_id))
                # classify objects by name (p is a place)
                if p.startswith("depot") or p.startswith("distributor"):
                    self.places.add(p)
                if x.startswith("pallet"):
                    self.pallets.add(x)
                elif x.startswith("crate"):
                    self.crates.add(x)
                elif x.startswith("truck"):
                    self.trucks.add(x)
                elif x.startswith("hoist"):
                    self.hoists.add(x)
                elif x.startswith("depot") or x.startswith("distributor"):
                    # Some encodings might use at on places (rare), collect defensively
                    self.places.add(x)
            elif pred == "clear" and len(args) == 1:
                s = args[0]
                self.clear_by_surface[s] = fact.global_id
                if s.startswith("pallet"):
                    self.pallets.add(s)
                if s.startswith("crate"):
                    self.crates.add(s)
            elif pred == "available" and len(args) == 1:
                h = args[0]
                self.available_by_hoist[h] = fact.global_id
                if h.startswith("hoist"):
                    self.hoists.add(h)

        # Parse the bit-encoded goal state once, keep only on-goals
        self.goal_pairs: List[Tuple[str, str]] = []  # List of (crate, support)
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self.goal_pairs.append((args[0], args[1]))

    # ----------------------------------------------------------------------
    # Heuristic public interface
    # ----------------------------------------------------------------------
    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self._estimate(state_bitwise)

    # ----------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------
    def _estimate(self, state_bitwise: int) -> int:
        # Caches per state evaluation to avoid recomputation during loops
        memo_true_support: Dict[str, Optional[str]] = {}
        memo_place: Dict[str, Optional[str]] = {}
        memo_is_lifting: Dict[str, Tuple[bool, Optional[str]]] = {}
        memo_in_truck: Dict[str, Tuple[bool, Optional[str]]] = {}

        h_val = 0
        for c, s_target in self.goal_pairs:
            fid_goal = self.on_fact_id.get((c, s_target))
            if fid_goal is not None and self._is_true(state_bitwise, fid_goal):
                # Goal already satisfied for this pair
                continue

            # Base: one Drop will be needed to achieve on(c, s_target)
            cost = 1

            # Determine c's current mode/location
            is_lift, lift_hoist = self._is_crate_lifting(state_bitwise, c, memo_is_lifting)
            in_truck, truck = self._is_crate_in_truck(state_bitwise, c, memo_in_truck)

            # Determine places (if known)
            target_place = self._get_support_place(state_bitwise, s_target, memo_true_support, memo_place)

            if is_lift:
                # If lifting and target is at a different place, need at least Load + Unload before Drop
                hoist_place = self._get_object_place(state_bitwise, lift_hoist, memo_place) if lift_hoist else None
                if hoist_place is not None and target_place is not None and hoist_place != target_place:
                    cost += 2  # Load + Unload (Drive ignored for LB)
                # No Lift/Unload needed: already lifting
            elif in_truck:
                # Need to Unload before Drop
                cost += 1
                # We assume we can unload at target place (Drive ignored for LB)
            else:
                # Assume c is on some surface or at ground: need a Lift before Drop
                cost += 1
                # If c and s are at different places, we'll need Load+Unload to move c across places
                c_place = self._get_crate_place(state_bitwise, c, memo_true_support, memo_place, memo_in_truck, memo_is_lifting)
                if c_place is not None and target_place is not None and c_place != target_place:
                    cost += 2  # Load + Unload

            # NOTE: We deliberately do NOT add cost for clearing s_target to remain conservative (admissible)
            h_val += cost

        return h_val

    # ----------------------------------------------------------------------
    # Fast bitwise check
    # ----------------------------------------------------------------------
    @staticmethod
    def _is_true(state_bitwise: int, fid: int) -> bool:
        return ((state_bitwise >> fid) & 1) == 1

    # ----------------------------------------------------------------------
    # Parsing helper from the spec
    # ----------------------------------------------------------------------
    @staticmethod
    def _parse_fact_name(name: str):
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

    # ----------------------------------------------------------------------
    # State interpretation utilities (with caching)
    # ----------------------------------------------------------------------
    def _is_crate_lifting(self, state_bitwise: int, crate: str, memo: Dict[str, Tuple[bool, Optional[str]]]) -> Tuple[bool, Optional[str]]:
        if crate in memo:
            return memo[crate]
        for h, fid in self.lifting_by_crate.get(crate, []):
            if self._is_true(state_bitwise, fid):
                memo[crate] = (True, h)
                return memo[crate]
        memo[crate] = (False, None)
        return memo[crate]

    def _is_crate_in_truck(self, state_bitwise: int, crate: str, memo: Dict[str, Tuple[bool, Optional[str]]]) -> Tuple[bool, Optional[str]]:
        if crate in memo:
            return memo[crate]
        for t, fid in self.in_by_crate.get(crate, []):
            if self._is_true(state_bitwise, fid):
                memo[crate] = (True, t)
                return memo[crate]
        memo[crate] = (False, None)
        return memo[crate]

    def _get_true_support_of_crate(self, state_bitwise: int, crate: str, memo_support: Dict[str, Optional[str]]) -> Optional[str]:
        if crate in memo_support:
            return memo_support[crate]
        for s, fid in self.on_by_crate.get(crate, []):
            if self._is_true(state_bitwise, fid):
                memo_support[crate] = s
                return s
        memo_support[crate] = None
        return None

    def _get_object_place(self, state_bitwise: int, obj: Optional[str], memo_place: Dict[str, Optional[str]]) -> Optional[str]:
        if obj is None:
            return None
        if obj in memo_place:
            return memo_place[obj]
        for p, fid in self.at_by_obj.get(obj, []):
            if self._is_true(state_bitwise, fid):
                memo_place[obj] = p
                return p
        memo_place[obj] = None
        return None

    def _get_support_place(self, state_bitwise: int, support: str,
                           memo_support: Dict[str, Optional[str]],
                           memo_place: Dict[str, Optional[str]]) -> Optional[str]:
        """Return the place of a support (pallet or crate) in current state.
        If it's a pallet: directly via at[pallet, p].
        If it's a crate: follow its current support recursively up to a pallet or use in/lifting/at to deduce.
        """
        # If direct at is known (crate or pallet), use it first
        place = self._get_object_place(state_bitwise, support, memo_place)
        if place is not None:
            return place

        # If support is a crate, follow its current support upwards
        if support.startswith("crate"):
            up = self._get_true_support_of_crate(state_bitwise, support, memo_support)
            if up is None:
                # Maybe it's in a truck or being lifted -> use those
                in_truck, t = self._is_crate_in_truck(state_bitwise, support, {})
                if in_truck and t is not None:
                    return self._get_object_place(state_bitwise, t, memo_place)
                is_lift, h = self._is_crate_lifting(state_bitwise, support, {})
                if is_lift and h is not None:
                    return self._get_object_place(state_bitwise, h, memo_place)
                # Unknown
                return None
            # If support above is pallet -> return its place
            if up.startswith("pallet"):
                return self._get_object_place(state_bitwise, up, memo_place)
            # Else: recursively continue up the stack
            return self._get_support_place(state_bitwise, up, memo_support, memo_place)

        # Unknown/unsupported
        return None

    def _get_crate_place(self, state_bitwise: int, crate: str,
                         memo_support: Dict[str, Optional[str]],
                         memo_place: Dict[str, Optional[str]],
                         memo_in: Dict[str, Tuple[bool, Optional[str]]],
                         memo_lift: Dict[str, Tuple[bool, Optional[str]]]) -> Optional[str]:
        # Direct at
        p = self._get_object_place(state_bitwise, crate, memo_place)
        if p is not None:
            return p
        # In truck
        in_truck, t = self._is_crate_in_truck(state_bitwise, crate, memo_in)
        if in_truck and t is not None:
            return self._get_object_place(state_bitwise, t, memo_place)
        # Lifting
        is_lift, h = self._is_crate_lifting(state_bitwise, crate, memo_lift)
        if is_lift and h is not None:
            return self._get_object_place(state_bitwise, h, memo_place)
        # On support -> use support place
        s = self._get_true_support_of_crate(state_bitwise, crate, memo_support)
        if s is not None:
            return self._get_support_place(state_bitwise, s, memo_support, memo_place)
        # Unknown
        return None


# Alias required by some configuration loaders that instantiate by parameter name
def depots_gpt5_heuristic_15(model):
    return DepotsGPT5Heuristic15(model)
