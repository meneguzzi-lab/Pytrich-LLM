from typing import Dict, List, Tuple, Iterable, Optional, Union

# NOTE: This heuristic is designed for the Pytrich Planner framework.
# It uses only fast bitwise checks for state membership and avoids slow explicit state scans.
#
# Heuristic name/class: DepotsGPT5Heuristic7
# __init__ parameter name: depots_gpt5_heuristic_7 (as required)
#
# Core idea (admissible, domain-dependent):
# - Goals are all of the form on(crate, surface). Achieving on(c, x) must include
#   primitive actions that directly manipulate crate c: Lift/Load/Unload/Drop with c as the manipulated crate.
# - To remain admissible when many goals interact (shared surfaces, shared trucks),
#   we count only the minimal number of actions that must directly involve each goal crate c.
#   We deliberately ignore transport drives, and ignore clearing actions that involve other crates,
#   because those can be shared across multiple goals and would risk double-counting.
# - For each unsatisfied on(c, x):
#     * If c is currently being lifted: we need at least one Drop of c => cost 1
#     * Otherwise (c in a truck, or on some surface different from x):
#         we need at least two c-touching actions (either Unload+Drop, or Lift+Drop) => cost 2
#   Summing across all unsatisfied on-goals is admissible because actions counted for different crates are disjoint
#   (they are bound to the y=crate parameter in the primitive operators).
# - Additionally, when the current task network (TN) already contains primitive operators, those must still be
#   executed to finish the plan. We lower-bound the remaining cost by the count of pending primitives in the TN.
#   To avoid double-counting with the goal-based lower bound, we take the maximum of the two estimates.
#
# Result: A fast, scalable, domain-aware and admissible heuristic that provides useful guidance without expensive
# structure-building. It handles small and very large Depots instances.

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic7(Heuristic):
    """
    Domain-dependent admissible heuristic for the Depots (HDDL) domain.

    h(s, tn) = max( sum_{(c,x) in unsatisfied on-goals} cost_c(s),
                    count_primitive_ops_in(tn) )

    Where cost_c(s) (crate-touching lower bound) is computed as:
      - If on[c, x] already true: 0
      - Else if there exists some hoist h with lifting[h, c] true: 1   (need Drop(c, x, ...))
      - Else: 2  (if c is in a truck -> Unload(c) + Drop(c),
                  if c is on some surface -> Lift(c) + Drop(c))

    Notes:
    - We intentionally ignore drive actions and any clearing actions involving other crates (e.g., to clear targets),
      because these can be shared across multiple goals and summing them would risk inadmissibility.
    - We only count primitive actions that must directly manipulate the goal crate c.
    - We also lower-bound the remaining cost by the number of already-present primitive operators in the current TN;
      we then take the maximum of this and the goal-based estimate to avoid double-counting.

    This design keeps evaluation O(#goals + #TN), relying solely on fast bitwise membership checks.
    """

    def __init__(self, model, depots_gpt5_heuristic_7: Optional[dict] = None):
        super().__init__(model)
        self.model = model
        self.depots_gpt5_heuristic_7_cfg = depots_gpt5_heuristic_7 or {}

        # Build a quick predicate index: predicate -> { args_tuple : Fact }
        # Facts use the "+pred[...]" string format; we parse and store global_ids for fast checks.
        self._pred_index: Dict[str, Dict[Tuple[str, ...], object]] = {}
        for fact in getattr(self.model, 'facts', []):
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            self._pred_index.setdefault(pred, {})[tuple(args)] = fact

        # Precompute goal facts and the (crate, surface) pairs for on-goals.
        # We use bit positions of the goal bitset for quick membership checking and to avoid name-based lookups.
        self._goal_fact_ids: List[int] = []                # list of goal fact bit positions (global_ids)
        self._goal_on_pairs: Dict[int, Tuple[str, str]] = {}  # gid -> (crate, surface)

        for gid in self._iter_set_bits(self.model.goals):
            # Model facts are indexed by their global_id
            fact = self.model.facts[gid]
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'on' and len(args) == 2:
                self._goal_fact_ids.append(gid)
                self._goal_on_pairs[gid] = (args[0], args[1])
            else:
                # Non-on goals (if any) are ignored for Depots; if present, you can still benefit from TN-based bound
                # and whatever implicit constraints the domain has. We skip them here to keep this heuristic fast.
                pass

        # Build helper maps for quick checks of lifting[* , c] and in[c, *] for any crate c.
        self._crate_to_lifting_fact_ids: Dict[str, List[int]] = {}
        self._crate_to_in_fact_ids: Dict[str, List[int]] = {}

        lifting_facts = self._pred_index.get('lifting', {})
        for (h, c), f in lifting_facts.items():
            self._crate_to_lifting_fact_ids.setdefault(c, []).append(f.global_id)

        in_facts = self._pred_index.get('in', {})
        for (c, t), f in in_facts.items():
            self._crate_to_in_fact_ids.setdefault(c, []).append(f.global_id)

    # --------------- Public API ---------------

    def __call__(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        return self.evaluate(state_bitwise, task_network)

    def evaluate(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        # Lower bound from goals (sum of crate-touching min actions per unsatisfied on-goal)
        goal_lb = 0
        for gid in self._goal_fact_ids:
            if self._is_true_bit(state_bitwise, gid):
                continue  # this on-goal already satisfied
            c, _x = self._goal_on_pairs[gid]
            goal_lb += self._crate_touching_cost(state_bitwise, c)

        # Lower bound from already-present primitive operators in the current task network
        tn_prim_lb = self._count_pending_primitives(task_network)

        # Use the maximum to avoid double-counting
        return max(goal_lb, tn_prim_lb)

    # --------------- Core Helpers ---------------

    def _crate_touching_cost(self, state_bitwise: int, crate: str) -> int:
        """
        Admissible lower bound on the number of remaining primitive actions that must directly manipulate
        the given crate to achieve its on-goal (ignoring drives and clearing of other crates).

        Returns:
            0 if already on target (handled upstream), else:
            1 if crate is currently being lifted (Drop needed),
            2 otherwise (Unload+Drop if in truck, or Lift+Drop if on another surface).
        """
        # If any lifting[h, crate] is true -> need at least Drop => 1
        if self._is_any_true(state_bitwise, self._crate_to_lifting_fact_ids.get(crate, [])):
            return 1

        # Otherwise, either in[crate, t] or on[crate, s] for some t/s
        # In both cases (goal not yet satisfied), we need at least two actions on crate:
        #   in -> Unload + Drop, on -> Lift + Drop
        return 2

    def _count_pending_primitives(self, task_network: List[Union[int, object]]) -> int:
        count = 0
        if not task_network:
            return 0

        for elem in task_network:
            comp = elem
            # Elements may be stored as ids; if so, retrieve the component
            if isinstance(elem, int):
                try:
                    comp = self.model.get_component(elem)
                except Exception:
                    comp = None
            # Count only primitive operators as remaining single-step costs
            if comp is not None and comp.__class__.__name__ == 'Operator':
                count += 1
        return count

    # --------------- Bitwise and Parsing Utilities ---------------

    @staticmethod
    def _iter_set_bits(bitset: int) -> Iterable[int]:
        i = 0
        value = bitset
        while value:
            if value & 1:
                yield i
            value >>= 1
            i += 1

    @staticmethod
    def _is_true_bit(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    @staticmethod
    def _is_any_true(state_bitwise: int, gids: List[int]) -> bool:
        # Fast check: OR the checks; break early on first match
        for gid in gids:
            if ((state_bitwise >> gid) & 1) == 1:
                return True
        return False

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name = name[1:]
            bracket_idx = name.find('[')
            if bracket_idx > 0 and name.endswith(']'):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []


# Convenience factory for frameworks that instantiate by name
def get_heuristic(model, depots_gpt5_heuristic_7: Optional[dict] = None):
    return DepotsGPT5Heuristic7(model, depots_gpt5_heuristic_7)
