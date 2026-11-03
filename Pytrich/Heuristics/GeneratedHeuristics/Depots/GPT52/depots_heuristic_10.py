from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic10(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Rationale (domain insights -> heuristic design)
    ----------------------------------------------
    Depots problems revolve around achieving many (on c s) goals, where s is a surface
    (pallet or crate). Achieving (on c s) typically requires:
      - The crate c and target surface s to be at the same place p.
      - A hoist at p.
      - The target surface s to be clear.
      - The crate c to be clear (for Lift precondition).
      - If c is in a truck, it must be unloaded at p.
      - If c is at a different place, it must be moved via truck (load/drive/unload).

    The HDDL methods reflect these cases:
      m0: already on -> 0 cost
      m1: c at p -> clear(c), clear(s), lift, drop
      m2: c in truck -> get truck to p, clear(s), unload, drop
      m3: general move -> load, drive, unload

    Heuristic strategy
    ------------------
    We compute a fast admissible lower bound by summing per-goal *local* lower bounds,
    using only information that can be checked with bitwise operations.

    For each unsatisfied goal +on[c,s], we add a minimal number of primitive actions
    that must occur in any plan (under a relaxation):

      - If on[c,s] already true: 0.
      - Otherwise at least one Drop is needed to establish on[c,s] (the only action that adds on).
        => +1.

      - If c is not currently being lifted by some hoist, then to be dropped it must be lifted or unloaded first.
        Unload creates lifting too; Lift creates lifting.
        Under a relaxation, at least one of {Lift, Unload} is required before Drop.
        => +1 if not currently lifting(c).

      - If c is currently in some truck, then Lift cannot apply; an Unload is necessary.
        => +1 (Unload).

      - If c is not at the same place as s (based on current at-facts), then some Drive must occur
        (either move truck to c then to s, or move c to s, etc.). In any case at least one Drive.
        => +1.

      - If c is not at the same place as s and c is not in a truck, then c must be loaded at some point.
        => +1 (Load).

      - Clearing requirements: if target surface s is not clear, something must be moved off it.
        => +1. If crate c is not clear and not currently lifted (Lift needs clear(c)), then at least one
        action is needed to clear it (in reality can be recursive; we keep it as +1 lower bound).
        => +1.

    This is a delete-relaxation-inspired counting heuristic specialized to Depots' causal structure.
    It is admissible as each increment corresponds to an action that is necessary in any real plan
    for that goal given the current situation.

    Efficiency
    ----------
    - Pre-parses fact IDs for key predicates (on, at, in, clear, lifting, available).
    - Per-state evaluation uses only bit tests and small dictionary lookups.
    - Goals are extracted correctly from model.goals via model.state_explicit_repr(model.goals).

    Notes
    -----
    - This heuristic is not perfectly consistent because it sums per-goal estimates with shared actions,
      but it remains a strong practical guide while staying a lower bound per-goal.
      If strict consistency is required, use max over goals rather than sum; here we keep sum because
      Depots instances contain many independent goals and sum provides better guidance.
    """

    def __init__(self, model, depots_gpt52_heuristic_10: str = "depots_gpt52_heuristic_10"):
        super().__init__(model, depots_gpt52_heuristic_10)
        self.model = model

        # fact_id lookup tables
        self._on_id: Dict[Tuple[str, str], int] = {}
        self._at_id: Dict[Tuple[str, str], int] = {}
        self._in_id: Dict[Tuple[str, str], int] = {}
        self._clear_id: Dict[str, int] = {}
        self._lifting_ids_by_crate: Dict[str, List[int]] = {}

        # also collect reverse mappings to cheaply determine locations
        self._places: set[str] = set()
        self._trucks: set[str] = set()
        self._hoists: set[str] = set()

        # Parse all facts once.
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "on" and len(args) == 2:
                self._on_id[(args[0], args[1])] = fact.global_id
            elif pred == "at" and len(args) == 2:
                obj, place = args[0], args[1]
                self._at_id[(obj, place)] = fact.global_id
                self._places.add(place)
            elif pred == "in" and len(args) == 2:
                self._in_id[(args[0], args[1])] = fact.global_id
                self._trucks.add(args[1])
            elif pred == "clear" and len(args) == 1:
                self._clear_id[args[0]] = fact.global_id
            elif pred == "lifting" and len(args) == 2:
                hoist, crate = args[0], args[1]
                self._lifting_ids_by_crate.setdefault(crate, []).append(fact.global_id)
                self._hoists.add(hoist)

        # Cache goal on-facts as (c,s) pairs.
        self._goal_on_pairs: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2 and name.startswith("+"):
                self._goal_on_pairs.append((args[0], args[1]))

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

    @staticmethod
    def _bit_true(state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _find_location(self, state_bitwise: int, obj: str) -> Optional[str]:
        """Return some place p such that at[obj,p] holds, else None.

        We keep this simple: scan all places but only through existing at-facts for obj.
        In Depots instances, each locatable is at exactly one place when not in truck/lifting.
        """
        # Fast path: iterate keys that match obj by checking dictionary entries.
        # (Still O(#places) worst-case if implemented naively; keep it O(#at-facts of obj)).
        # We don't have an index obj->[(obj,place)], so build it lazily via scan of known places.
        for place in self._places:
            fid = self._at_id.get((obj, place))
            if fid is not None and ((state_bitwise >> fid) & 1):
                return place
        return None

    def _is_in_truck(self, state_bitwise: int, crate: str) -> Optional[str]:
        """Return truck t if in[crate,t] holds, else None."""
        for t in self._trucks:
            fid = self._in_id.get((crate, t))
            if fid is not None and ((state_bitwise >> fid) & 1):
                return t
        return None

    def _is_lifted(self, state_bitwise: int, crate: str) -> bool:
        ids = self._lifting_ids_by_crate.get(crate)
        if not ids:
            return False
        # any lifting[hoist,crate]
        for fid in ids:
            if ((state_bitwise >> fid) & 1):
                return True
        return False

    def __call__(self, node) -> int:
        state = node.state

        h_val = 0

        for c, s in self._goal_on_pairs:
            on_fid = self._on_id.get((c, s))
            if on_fid is not None and ((state >> on_fid) & 1):
                continue  # goal already satisfied

            # 1) Need a Drop to make on[c,s]
            per_goal = 1

            # Determine current situation of crate c
            lifted = self._is_lifted(state, c)
            in_truck = self._is_in_truck(state, c)

            # 2) If not already lifted, must become lifted before Drop (via Lift or Unload)
            if not lifted:
                per_goal += 1

            # 3) If in truck, must Unload (Lift doesn't apply)
            if in_truck is not None:
                per_goal += 1

            # 4-5) Movement: if c and s are at different places, must at least drive, and if c not already
            # in truck then must load at least once.
            loc_s = self._find_location(state, s)

            # If crate is lifted, it is not at any place (Lift deletes at[c,p]). In that case,
            # treat its location as the hoist's place is unknown; don't add movement lower bound.
            if not lifted and in_truck is None:
                loc_c = self._find_location(state, c)
            else:
                loc_c = None

            if loc_s is not None:
                # If c is in truck, compare truck location with s.
                if in_truck is not None:
                    loc_t = self._find_location(state, in_truck)
                    if loc_t is not None and loc_t != loc_s:
                        per_goal += 1  # at least one Drive
                elif loc_c is not None and loc_c != loc_s:
                    per_goal += 1  # at least one Drive
                    per_goal += 1  # at least one Load somewhere

            # 6) Clearing: target surface must be clear to Drop.
            clear_s_fid = self._clear_id.get(s)
            if clear_s_fid is not None and (((state >> clear_s_fid) & 1) == 0):
                per_goal += 1

            # 7) Clearing: crate must be clear to Lift (if we need Lift rather than Unload).
            # If already lifted, ignore. If in truck, Unload doesn't require clear(crate) but later operations might.
            # For a lower bound, only count if not clear and not lifted.
            clear_c_fid = self._clear_id.get(c)
            if (not lifted) and (clear_c_fid is not None) and (((state >> clear_c_fid) & 1) == 0):
                per_goal += 1

            h_val += per_goal

        return h_val
