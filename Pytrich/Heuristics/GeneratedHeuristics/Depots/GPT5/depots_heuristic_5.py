from typing import Dict, List, Tuple, Optional, Set

# NOTE: The Pytrich framework provides this base class.
# If your project uses a different import path, adjust accordingly.
from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic5(Heuristic):
    """
    Domain-dependent heuristic for the Depot (HDDL) domain.

    Design rationale (admissible, fast):
    - Each goal of the form on(crate c, surface s) requires, if not already true, at least:
      - One final Drop action for crate c onto target surface s (unique per goal).
      - One action to make c manipulable at the target place:
        * If c is currently in some truck, you must at least Unload c (unique per crate).
        * Otherwise, you must at least Lift c from its current support before you can Drop (unique per crate unless c is already being lifted).
    - We do NOT charge for Drive or for recursively clearing stacks, because these can be shared across goals,
      and summing them per goal may overestimate (inadmissible). By skipping these, the estimate remains a
      consistent lower bound while still giving guidance based on whether crates are already lifted or inside trucks.

    - Heuristic value h(s) is computed as the sum over all unsatisfied on-goals of:
        1 (Drop) + [1 if c is in a truck else 0 if c is currently lifted else 1]
      That is:
        - If c already lifted: cost = 1 (only the Drop still needed)
        - If c in a truck:     cost = 2 (Unload + Drop)
        - Otherwise:           cost = 2 (Lift + Drop)

    - This remains admissible because we only count per-goal actions that cannot be shared between different goals.
      It is also state-dependent (checks current 'in' and 'lifting') and very fast (bitwise checks only).

    Integration details:
    - Class name: DepotsGPT5Heuristic5
    - __init__ second parameter name: depots_gpt5_heuristic_5 (lowercase, as requested)
    - Uses model.goals via model.state_explicit_repr(model.goals) only once at init for goal parsing.
    - Uses fast bitwise membership checks for all state queries during heuristic evaluation.
    """

    def __init__(self, model, depots_gpt5_heuristic_5=None):
        super().__init__(model)
        self.model = model

        # Internal parsed representation for all facts (indexed by global_id)
        self._fact_pred_args: List[Tuple[Optional[str], List[str]]] = [self._parse_fact_name(f.name) for f in self.model.facts]

        # Parse goal on-pairs once
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self.goal_on_pairs: List[Tuple[str, str]] = []  # list of (crate, surface)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'on' and len(args) == 2:
                self.goal_on_pairs.append((args[0], args[1]))

        # Index facts for quick bitwise checks
        # Map (c, s) -> fact_id for '+on[c,s]' used to check whether a specific goal is already satisfied
        self._on_fact_id: Dict[Tuple[str, str], int] = {}
        # For crates appearing in goals: 'in' and 'lifting' fact-ids per crate
        crates_in_goals: Set[str] = set(c for c, _ in self.goal_on_pairs)
        self._in_fact_ids_by_crate: Dict[str, List[int]] = {c: [] for c in crates_in_goals}
        self._lifting_fact_ids_by_crate: Dict[str, List[int]] = {c: [] for c in crates_in_goals}

        # Build indices by scanning all facts once
        for fid, (pred, args) in enumerate(self._fact_pred_args):
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                # Store only the pairs we might need (all, goal lookups will use these)
                self._on_fact_id[(c, s)] = fid
            elif pred == 'in' and len(args) == 2:
                c = args[0]
                if c in self._in_fact_ids_by_crate:
                    self._in_fact_ids_by_crate[c].append(fid)
            elif pred == 'lifting' and len(args) == 2:
                # (lifting hoist, crate)
                c = args[1]
                if c in self._lifting_fact_ids_by_crate:
                    self._lifting_fact_ids_by_crate[c].append(fid)

        # Precompute the fact ids for goal satisfaction checks; some goals might be missing due to grounding issues
        # Guard against missing: we will treat as not yet satisfied (safe for admissibility)
        self._goal_on_fact_ids: Dict[Tuple[str, str], Optional[int]] = {}
        for c, s in self.goal_on_pairs:
            self._goal_on_fact_ids[(c, s)] = self._on_fact_id.get((c, s), None)

    # ------------- Heuristic API -------------
    def __call__(self, state_bitwise: int, task_network: List) -> int:
        return self.get_value(state_bitwise, task_network)

    def get_value(self, state_bitwise: int, task_network: List) -> int:
        # Sum admissible per-goal lower bounds as described in the rationale.
        h_val = 0
        for c, s in self.goal_on_pairs:
            if self._is_goal_satisfied(state_bitwise, c, s):
                continue
            # Always at least one Drop to achieve on(c, s)
            cost = 1
            # Then either Unload (if in truck) or Lift (if not in truck and not already lifted)
            if self._is_crate_in_any_truck(state_bitwise, c):
                cost += 1  # Unload
            else:
                # If crate is already being lifted, do not count the Lift again
                if not self._is_crate_currently_lifted(state_bitwise, c):
                    cost += 1  # Lift
            h_val += cost
        return h_val

    # ------------- Internal helpers -------------
    def _is_goal_satisfied(self, state_bitwise: int, c: str, s: str) -> bool:
        fid = self._goal_on_fact_ids.get((c, s))
        if fid is None:
            # If we cannot find the fact-id for this goal, play safe: assume not satisfied
            return False
        return self._is_set(state_bitwise, fid)

    def _is_crate_in_any_truck(self, state_bitwise: int, c: str) -> bool:
        fids = self._in_fact_ids_by_crate.get(c)
        if not fids:
            return False
        for fid in fids:
            if self._is_set(state_bitwise, fid):
                return True
        return False

    def _is_crate_currently_lifted(self, state_bitwise: int, c: str) -> bool:
        fids = self._lifting_fact_ids_by_crate.get(c)
        if not fids:
            return False
        for fid in fids:
            if self._is_set(state_bitwise, fid):
                return True
        return False

    @staticmethod
    def _is_set(state_bitwise: int, fact_id: int) -> bool:
        return (state_bitwise >> fact_id) & 1 == 1

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
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


# Alias to satisfy potential external references expecting a factory-like symbol
def depots_gpt5_heuristic_5(model, depots_gpt5_heuristic_5=None):
    return DepotsGPT5Heuristic5(model, depots_gpt5_heuristic_5)
