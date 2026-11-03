from typing import Dict, List, Tuple, Optional, Any

# NOTE: This file defines a domain-dependent heuristic for the HDDL Depots domain
# tailored to the Pytrich Planner framework.
#
# Heuristic name/class: DepotsGPT5Heuristic17
# __init__ parameter name: depots_gpt5_heuristic_17 (lowercase, as requested)
#
# Design summary (see class docstring for details):
#   h(s) = sum over unsatisfied on(c,surf) goals of [
#              1 (mandatory Drop to achieve on)
#            + 1 if crate c is not currently lifting (a Lift or Unload is required to reach lifting)
#            + 2 if crate c must be transported between different places (requires Load + Unload)
#                  - This transport cost is added only when it is strictly necessary and unique
#                  - We do NOT count any Drive actions (they can be shared across crates)
#         ]
# This is an admissible lower bound: all counted steps are mandatory and not shareable between distinct goal crates,
# while sharable steps (e.g., Drive) are deliberately omitted to preserve admissibility.
#
# Implementation highlights:
#   - Strict use of bitwise checks for fact truthfulness (fast; required for large instances)
#   - Pre-indexing of facts by predicate/arguments for O(1)/O(k) lookups per object
#   - Goals extracted once from model.goals (bitwise integer) and cached
#   - Robust handling of lifted/in-truck/at-place states for crates; conservative where data is missing
#
# This heuristic provides strong guidance on large Depots instances by recognizing unavoidable action patterns:
#   Drop (to add on), Lift/Unload (to reach lifting), and Load+Unload (to move crates across places).

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Depots HDDL domain.

    Rationale and admissibility argument:
    - Every unsatisfied goal of the form on(c, s) must (in any valid plan) be achieved by exactly one Drop action.
      Therefore, for each unsatisfied goal we count +1.
    - Before Drop, crate c must be in the lifting state (precondition of Drop includes lifting(h, c)).
      If c is not currently lifting, the only ways to achieve lifting are:
        * Lift(h, c, s_cur, p) if c is at some place p (on some surface s_cur). This is one action.
        * Unload(h, c, t, p) if c is inside a truck. This is also one action.
      We conservatively add +1 if c is not lifting right now.
    - If c must change places (from p_c to p_s), then (given that hoists cannot move between places and crates cannot
      be carried without trucks) c necessarily requires at least one Load and one Unload action.
      These are crate-specific and cannot be shared with other crates, hence we can safely add +2 (Load + Unload)
      whenever we can confirm that p_c != p_s and c is not already inside a truck. If c is already lifted at a
      different place than p_s, the same lower bound (+2) applies. If c is in a truck, transport is already accounted
      for by Unload (and possibly later Drive, which we do not count since Drive can be shared and is not crate-unique).

    We do NOT add costs for clearing targets (clear(s)) nor for Drive actions, since these may be shareable across
    multiple goal crates and adding them per goal would risk inadmissibility. The chosen components are specific to
    each goal crate and are thus safe to sum.

    Heuristic formula per unsatisfied goal on(c, s):
      h_goal(c, s) = 1 (Drop)
                   + 1 if c is not lifting (must Lift or Unload to become lifting)
                   + 2 if c must be transported across places (Load + Unload), i.e.:
                         - case A: c is at place p_c, s is at place p_s, p_c != p_s  => +2
                         - case B: c is lifting at place p_lift, s is at place p_s, p_lift != p_s => +2
                         - case C: c is in truck: +0 here (Unload counted in pre-lift; Drive omitted on purpose)

    Implementation efficiency:
    - Uses bitwise membership checks for facts: (state >> fact_id) & 1
    - Pre-indexes all facts by predicate/arguments to avoid scanning the full fact set at runtime
    - Caches goal on(c, s) pairs and their fact IDs on initialization

    Parameters
    - depots_gpt5_heuristic_17: optional flag (unused; reserved for future toggles or debug)
    """

    def __init__(self, model, depots_gpt5_heuristic_17: bool = True):
        super().__init__(model)
        self._build_indexes()
        self.goal_on_pairs: List[Tuple[str, str, Optional[int]]] = self._extract_goal_on_pairs()

    # ------------------------------ Public API ------------------------------
    def __call__(self, state_bitwise: int, task_network: List[Any]) -> int:
        # Compute admissible cost estimate as described in the class docstring
        h_val = 0
        for c, s, on_fact_id in self.goal_on_pairs:
            # If goal on(c, s) already holds, skip
            if on_fact_id is not None and self._is_true(state_bitwise, on_fact_id):
                continue

            # 1) Mandatory Drop for this crate
            h_goal = 1

            # Gather current crate status
            in_truck, in_truck_truck = self._get_in_truck(state_bitwise, c)
            lifting, lifting_hoist, lifting_place = self._get_lifting_place(state_bitwise, c)

            # 2) Must be in lifting state before Drop (if not already lifting => Unload or Lift)
            if not lifting:
                # Either Unload (if in truck) or Lift (if at place)
                h_goal += 1

            # 3) Transport (Load + Unload) if c and target surface s are at different places
            # We only add transport if we can safely confirm that a place-change is necessary.
            target_place = self._get_at_place(state_bitwise, s)

            if target_place is not None:
                if lifting:
                    # If currently lifting at lifting_place and different from target_place => need Load + Unload
                    if lifting_place is not None and lifting_place != target_place:
                        h_goal += 2
                else:
                    if in_truck:
                        # In-truck: we count Unload above (+1); Drive is shared and omitted.
                        pass
                    else:
                        # At place: if at c's place differs from s's place, need Load + Unload
                        c_place = self._get_at_place(state_bitwise, c)
                        if c_place is not None and c_place != target_place:
                            h_goal += 2

            h_val += h_goal

        return h_val

    # ------------------------------ Indexing ------------------------------
    def _build_indexes(self):
        # Parsed fact indexes for fast lookup
        self.on_index: Dict[Tuple[str, str], int] = {}
        self.at_index: Dict[str, List[Tuple[str, int]]] = {}          # obj -> list of (place, fact_id)
        self.in_index: Dict[str, List[Tuple[str, int]]] = {}          # crate -> list of (truck, fact_id)
        self.lifting_index: Dict[str, List[Tuple[str, int]]] = {}     # crate -> list of (hoist, fact_id)
        self.clear_index: Dict[str, int] = {}                         # surface -> fact_id
        self.hoist_at_index: Dict[str, List[Tuple[str, int]]] = {}    # hoist -> list of (place, fact_id)

        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue

            if predicate == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                self.on_index[(c, s)] = fact.global_id
            elif predicate == 'at' and len(args) == 2:
                obj, place = args[0], args[1]
                self.at_index.setdefault(obj, []).append((place, fact.global_id))
                # Track hoist locations too (for lifting place inference)
                if obj.startswith('hoist'):
                    self.hoist_at_index.setdefault(obj, []).append((place, fact.global_id))
            elif predicate == 'in' and len(args) == 2:
                c, t = args[0], args[1]
                self.in_index.setdefault(c, []).append((t, fact.global_id))
            elif predicate == 'lifting' and len(args) == 2:
                h, c = args[0], args[1]
                self.lifting_index.setdefault(c, []).append((h, fact.global_id))
            elif predicate == 'clear' and len(args) == 1:
                s = args[0]
                self.clear_index[s] = fact.global_id
            # 'available' is not needed for this heuristic (ignored by design)

    def _extract_goal_on_pairs(self) -> List[Tuple[str, str, Optional[int]]]:
        pairs: List[Tuple[str, str, Optional[int]]] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            predicate, args = self._parse_fact_name(name)
            if predicate == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                fact_id = self.on_index.get((c, s), None)
                pairs.append((c, s, fact_id))
        return pairs

    # ------------------------------ State queries ------------------------------
    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _get_at_place(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Return the (first) place p with +at[obj,p] true in state, else None
        entries = self.at_index.get(obj, [])
        for p, fid in entries:
            if self._is_true(state_bitwise, fid):
                return p
        return None

    def _get_in_truck(self, state_bitwise: int, c: str) -> Tuple[bool, Optional[str]]:
        # Return (True, truck) if +in[c,truck] true; else (False, None)
        entries = self.in_index.get(c, [])
        for t, fid in entries:
            if self._is_true(state_bitwise, fid):
                return True, t
        return False, None

    def _get_lifting_place(self, state_bitwise: int, c: str) -> Tuple[bool, Optional[str], Optional[str]]:
        # If lifting[h, c] holds, return (True, h, place_of_h)
        entries = self.lifting_index.get(c, [])
        for h, fid in entries:
            if self._is_true(state_bitwise, fid):
                # Find hoist place
                p = self._get_at_place(state_bitwise, h)
                return True, h, p
        return False, None, None

    # ------------------------------ Utilities ------------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        # As specified in the prompt: '+predicate[arg1,arg2]' or '-predicate[arg1,arg2]'
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
            bracket_idx = s.find('[')
            if bracket_idx > 0 and s.endswith(']'):
                predicate = s[:bracket_idx]
                args_str = s[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []


# Alias for potential external references
HeuristicClass = DepotsGPT5Heuristic17
