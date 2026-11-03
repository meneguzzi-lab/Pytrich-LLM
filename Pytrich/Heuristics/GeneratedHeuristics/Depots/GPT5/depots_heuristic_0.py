from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Try to import the framework base classes; fall back to minimal stubs if unavailable at import time.
from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic0(Heuristic):
    """
    Domain-dependent heuristic for the Depots HDDL domain.

    Design overview (admissible lower bound):
    - The only action that can make on(c, s) true is Drop(h, c, s, p).
      Therefore, for every goal on(c, s) that is not yet satisfied, we need at least one Drop.
    - Drop requires lifting(h, c) and clear(s). In a delete-relaxation style lower bound,
      we estimate the minimal additional steps to enable Drop:
        * If lifting(h, c) is not true in the current state, we must create it via either Unload (if c is in a truck) or Lift (if c is on a surface and clear(c)).
          => add +1 when lifting(c) is currently false.
        * If c is NOT in any truck AND c is NOT clear(c), then before we can Lift(c, ...), we must first clear c by lifting the crate on top of c.
          => add +1 more in that case (best-case lower bound: one lift clears c; we ignore freeing the hoist to stay admissible).
        * If clear(s) is false, we add +1 to clear s by lifting the top crate from s.
          => add +1 in that case.
    - We purposefully ignore place alignment, truck driving, and hoist availability/location in this bound.
      That keeps the heuristic admissible while remaining computationally cheap. It also ties directly to
      the HTN methods in this domain (do_put_on reduces to clearing, lifting, and finally dropping).

    Final per-goal lower bound if on(c,s) is not yet achieved:
      1 (Drop) + [1 if not lifting(c)] + [1 if (not in(c, any truck) and not clear(c) and not lifting(c))] + [1 if not clear(s)]

    This is summed across all unsatisfied on-goals. Actions for different crates cannot generally serve each other
    (e.g., a lift for c1 does not create lifting(c2)), so summation is safe and remains a valid lower bound.

    Performance considerations:
    - All needed fact IDs are pre-indexed once in __init__ (by scanning facts):
        * clear[x] -> fact_id
        * lifting[h,c] -> grouped by c (crate)
        * in[c,t] -> grouped by c (crate)
      Goal on-facts are extracted directly from the model goal bitset.
    - During evaluation, only bitwise tests are performed; no string parsing.

    Notes:
    - The heuristic is designed for scale (e.g., problem p30). It leverages only O(#goals) checks per state,
      with small constant work per goal (checking a few pre-indexed fact IDs per crate/surface).
    - It is admissible and robust to partial states.
    """

    def __init__(self, model, depots_gpt5_heuristic_0: bool = True):
        super().__init__(model)
        self.depots_gpt5_heuristic_0 = depots_gpt5_heuristic_0

        # Internal indexes
        self.clear_fact_id: Dict[str, int] = {}               # name -> fact_id for clear[name]
        self.lifting_ids_by_crate: Dict[str, List[int]] = defaultdict(list)  # crate -> [fact_ids of lifting[h, crate]]
        self.in_ids_by_crate: Dict[str, List[int]] = defaultdict(list)       # crate -> [fact_ids of in[crate, t]]
        self.goal_items: List[Tuple[int, str, str]] = []      # List of (goal_fact_id, crate_name, surface_name)

        # Build all indices once
        self._build_indexes()

    # ----------------------------
    # Utilities and index builders
    # ----------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse a grounded fact name of the form '+predicate[arg1,arg2]' or '-predicate[arg1]'.
        Returns (predicate, [args...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_wo = name[1:]
            b = name_wo.find("[")
            if b > 0 and name_wo.endswith("]"):
                pred = name_wo[:b]
                args_str = name_wo[b + 1:-1]
                args = args_str.split(",") if args_str else []
                return pred, args
        return None, []

    def _iter_set_bits(self, bitset: int):
        """Yield positions of set bits in a non-negative integer."""
        pos = 0
        bs = bitset
        while bs:
            if bs & 1:
                yield pos
            bs >>= 1
            pos += 1

    def _build_indexes(self):
        # 1) Scan facts once to index clear[], lifting[], in[]
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue
            if pred == 'clear' and len(args) == 1:
                self.clear_fact_id[args[0]] = fact.global_id
            elif pred == 'lifting' and len(args) == 2:
                h, c = args[0], args[1]
                self.lifting_ids_by_crate[c].append(fact.global_id)
            elif pred == 'in' and len(args) == 2:
                c, t = args[0], args[1]
                self.in_ids_by_crate[c].append(fact.global_id)

        # 2) Extract goal on-facts directly from the goal bitset
        for bit_pos in self._iter_set_bits(self.model.goals):
            f = self.model.facts[bit_pos]
            pred, args = self._parse_fact_name(f.name)
            # We only care about on-goals in this domain (provided problem specs)
            if pred == 'on' and len(args) == 2:
                c, s = args[0], args[1]
                self.goal_items.append((f.global_id, c, s))
        # If there are other goal predicates, they are ignored by this heuristic (treated as 0 cost),
        # remaining goals are still enforced by the solver.

    # ----------------------------
    # Bitwise helpers
    # ----------------------------
    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    @staticmethod
    def _any_true(state_bitwise: int, fact_ids: List[int]) -> bool:
        for fid in fact_ids:
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    # ----------------------------
    # Heuristic evaluation
    # ----------------------------
    def __call__(self, state_bitwise: int, task_network: List) -> int:
        """
        Compute the heuristic value h(n) for the given state and current task network.

        Parameters:
        - state_bitwise: int bitset encoding the current state facts (fast bitwise membership checks).
        - task_network: current list of tasks (operators and/or abstract tasks). Not directly used in the
                        lower bound (to avoid double counting), but our per-goal estimate aligns with the
                        decomposition structure of do_put_on tasks.
        """
        h_val = 0

        for goal_fact_id, crate, surface in self.goal_items:
            # If this on-goal is already true, it costs 0.
            if self._is_true(state_bitwise, goal_fact_id):
                continue

            # Base: one Drop is required to add on(crate, surface)
            cost = 1

            # lifting(crate) needed before Drop. Check if any lifting[h, crate] holds.
            lifting_ids = self.lifting_ids_by_crate.get(crate, [])
            crate_being_lifted = self._any_true(state_bitwise, lifting_ids)
            if not crate_being_lifted:
                # Need at least one action to achieve lifting(crate) (either Unload or Lift)
                cost += 1

                # If crate is not in any truck and not clear(crate), we must clear it first (at least one Lift)
                in_ids = self.in_ids_by_crate.get(crate, [])
                crate_in_truck = self._any_true(state_bitwise, in_ids)
                if not crate_in_truck:
                    clear_c_id = self.clear_fact_id.get(crate, None)
                    if clear_c_id is not None and not self._is_true(state_bitwise, clear_c_id):
                        cost += 1

            # Surface must be clear for Drop
            clear_s_id = self.clear_fact_id.get(surface, None)
            if clear_s_id is not None and not self._is_true(state_bitwise, clear_s_id):
                cost += 1

            h_val += cost

        return h_val


# Alias for loader convenience (optional):
HeuristicClass = DepotsGPT5Heuristic0
