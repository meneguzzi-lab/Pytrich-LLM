# Domain-dependent heuristic for HDDL Depots domain
# Class name: DepotsGPT5Heuristic12 (inherits from Heuristic)
# __init__ parameter: depots_gpt5_heuristic_12 (lowercase as required)
#
# This heuristic is designed for the Pytrich Planner framework.
# It uses domain knowledge of the Depots domain to create an admissible, consistent, and efficient
# estimate of the cost-to-go based on the current state and the (static) classical goal set.
#
# Core idea (lower bound, admissible):
# For each unsatisfied goal on(c, s2), the following primitive actions are strictly necessary,
# independent across crates (so summation remains a lower bound):
#   - One Drop of crate c onto surface s2 (if goal not yet true)  -> +1
#   - If c is not already in a truck and not already being lifted -> +1 Lift (to detach from its current support)
#   - If c must move across places (place(c) != place(s2)) and c is not already in a truck (or when lifting across places),
#       it must be transported by a truck: at least one Load and one Unload                    -> +2
#   - If c is currently in a truck: to be placed on s2 it needs Unload first, then Drop        -> +1 Unload (Drop already counted)
# We deliberately DO NOT add any Drive actions (truck movements) because drives can be shared among
# multiple crates in an optimal plan; summing drives per-goal would risk inadmissibility. Ignoring
# Drive maintains admissibility and still provides strong guidance from Lift/Load/Unload/Drop counts.
#
# We also avoid counting costs for clearing target surfaces (s2) or clearing crate c when it has a
# blocker on top, because those actions may be shared with other goals (e.g., the blocker crate might
# have its own goal that already requires lifting it). Not counting those preserves admissibility.
#
# Implementation details:
# - We use fast bitwise checks over state_bitwise with each Fact's global_id.
# - We parse model.goals (bitset) once to capture the set of goal on[c, s2] pairs.
# - We pre-index the model's facts for O(1) access to relevant predicates: on, in, at, lifting.
# - Heuristic value is the sum of the per-goal lower bounds above.
#
# Notes on robustness:
# - The code handles arbitrary intermediate states (partial decompositions), uses only the classical
#   goal set (model.goals) for guidance, and remains efficient on large instances (few arithmetic
#   ops and short scans over pre-indexed facts).

from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Depots HDDL domain.

    Admissible lower-bound estimate of remaining actions to achieve all goal facts on(c, s2):
    - Counts only per-crate, non-shareable actions: Lift, Load, Unload, Drop (never Drive), as justified above.
    - Uses fast bitwise checks for state membership.

    Public API:
    - __call__(state_bitwise, task_network=None) -> int
      Also accepts a single 'node' argument if it exposes 'state' and 'task_network' attributes.

    Construction:
    - depots_gpt5_heuristic_12: optional flag for toggling/debugging (unused, maintained for the required name).
    """

    def __init__(self, model, depots_gpt5_heuristic_12: bool = True):
        super().__init__()
        self.model = model
        self.depots_gpt5_heuristic_12 = depots_gpt5_heuristic_12

        # Map from full fact name string to Fact (for quick access if needed)
        self._fact_by_name: Dict[str, object] = {}
        # Pre-indexed structures for fast per-entity lookups in a given state
        # on: direct mapping for goals
        self._on_fact_by_pair: Dict[Tuple[str, str], object] = {}
        # in: per-crate list of (Fact, truck)
        self._in_facts_by_crate: Dict[str, List[Tuple[object, str]]] = {}
        # lifting: per-crate list of (Fact, hoist)
        self._lifting_facts_by_crate: Dict[str, List[Tuple[object, str]]] = {}
        # at: per-entity list of (Fact, place)
        self._at_facts_by_entity: Dict[str, List[Tuple[object, str]]] = {}

        # Parse and index all facts once
        for fact in self.model.facts:
            name = fact.name
            self._fact_by_name[name] = fact
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue
            if pred == 'on' and len(args) == 2:
                c, s = args
                self._on_fact_by_pair[(c, s)] = fact
            elif pred == 'in' and len(args) == 2:
                c, t = args
                self._in_facts_by_crate.setdefault(c, []).append((fact, t))
            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self._lifting_facts_by_crate.setdefault(c, []).append((fact, h))
            elif pred == 'at' and len(args) == 2:
                x, p = args
                self._at_facts_by_entity.setdefault(x, []).append((fact, p))

        # Extract goal pairs (crate, surface)
        self._goal_pairs: List[Tuple[str, str]] = []
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                self._goal_pairs.append((args[0], args[1]))
        # Remove duplicates while preserving order
        seen_g = set()
        unique_pairs = []
        for pair in self._goal_pairs:
            if pair not in seen_g:
                unique_pairs.append(pair)
                seen_g.add(pair)
        self._goal_pairs = unique_pairs

    # =============================
    # Public interface
    # =============================
    def __call__(self, *args, **kwargs) -> int:
        """
        Flexible call interface:
        - __call__(state_bitwise, task_network=None)
        - __call__(node) where node.state and node.task_network exist
        """
        if len(args) == 1 and hasattr(args[0], 'state'):
            node = args[0]
            state_bitwise = getattr(node, 'state', None)
            task_network = getattr(node, 'task_network', None)
            return self.evaluate(state_bitwise, task_network)
        elif len(args) >= 1:
            state_bitwise = args[0]
            task_network = args[1] if len(args) >= 2 else None
            return self.evaluate(state_bitwise, task_network)
        else:
            raise ValueError('DepotsGPT5Heuristic12.__call__ requires (state_bitwise[, task_network]) or a node with .state/.task_network')

    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        """
        Compute admissible heuristic value for the given state.
        task_network is ignored on purpose to keep admissibility (abstract tasks can reduce to nop).
        """
        if state_bitwise is None:
            return 0
        h_val = 0
        for c, s2 in self._goal_pairs:
            h_val += self._per_goal_cost(state_bitwise, c, s2)
        return h_val

    # Alias for compatibility with some planners
    def h(self, state_bitwise: int, task_network=None) -> int:
        return self.evaluate(state_bitwise, task_network)

    # =============================
    # Internal helpers
    # =============================
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse fact names in format: +predicate[arg1,arg2] (or -predicate[arg1,arg2], sign ignored).
        Returns (predicate, [args]).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        idx = s.find('[')
        if idx <= 0 or not s.endswith(']'):
            return None, []
        pred = s[:idx]
        args_str = s[idx + 1:-1]
        args = args_str.split(',') if args_str else []
        return pred, args

    def _is_true(self, state_bitwise: int, fact_obj) -> bool:
        if fact_obj is None:
            return False
        return ((state_bitwise >> fact_obj.global_id) & 1) == 1

    def _on_true(self, state_bitwise: int, c: str, s: str) -> bool:
        fact = self._on_fact_by_pair.get((c, s))
        return self._is_true(state_bitwise, fact)

    def _crate_in_truck(self, state_bitwise: int, c: str) -> Optional[str]:
        """Return the truck name t if in[c,t] holds, else None."""
        for fact, t in self._in_facts_by_crate.get(c, []):
            if self._is_true(state_bitwise, fact):
                return t
        return None

    def _crate_lifting_by(self, state_bitwise: int, c: str) -> Optional[str]:
        """Return hoist name h if lifting[h,c] holds, else None."""
        for fact, h in self._lifting_facts_by_crate.get(c, []):
            if self._is_true(state_bitwise, fact):
                return h
        return None

    def _entity_place(self, state_bitwise: int, x: str) -> Optional[str]:
        """Return place p if at[x,p] holds, else None."""
        for fact, p in self._at_facts_by_entity.get(x, []):
            if self._is_true(state_bitwise, fact):
                return p
        return None

    def _per_goal_cost(self, state_bitwise: int, c: str, s2: str) -> int:
        """
        Compute admissible per-goal lower bound for on(c, s2) from current state.
        Cases:
         - If on[c,s2] already: 0
         - Else always include Drop: +1
         - If in[c,t]: +1 (Unload) [Drop counted]
         - Else if lifting[h,c]: if places differ (hoist place != s2 place) -> +2 (Load+Unload), else 0
         - Else (c on some support): +1 (Lift) and if place(c) != place(s2) -> +2 (Load+Unload)
        """
        # Goal already true
        if self._on_true(state_bitwise, c, s2):
            return 0

        cost = 0
        # Always need a Drop to assert on[c, s2]
        cost += 1  # Drop

        # Check if crate is currently in a truck
        t = self._crate_in_truck(state_bitwise, c)
        if t is not None:
            # Need to unload before drop
            cost += 1  # Unload
            # We ignore Drive (can be shared), keep admissible
            return cost

        # Check if crate is currently being lifted
        h = self._crate_lifting_by(state_bitwise, c)
        if h is not None:
            # If lifting at a different place than s2, we must go via truck -> Load + Unload
            p_h = self._entity_place(state_bitwise, h)
            p_s2 = self._entity_place(state_bitwise, s2)
            if p_h is not None and p_s2 is not None and p_h != p_s2:
                cost += 2  # Load + Unload
            # else same place: just Drop (already counted)
            return cost

        # Otherwise, crate is on some support at a place (not in truck, not lifting)
        # Must lift it off its current support
        cost += 1  # Lift

        # If crate and target surface are at different places, we must Load + Unload via a truck
        p_c = self._entity_place(state_bitwise, c)
        p_s2 = self._entity_place(state_bitwise, s2)
        if p_c is not None and p_s2 is not None and p_c != p_s2:
            cost += 2  # Load + Unload

        # Do not add Drive; keep admissible since drives can be shared
        return cost
