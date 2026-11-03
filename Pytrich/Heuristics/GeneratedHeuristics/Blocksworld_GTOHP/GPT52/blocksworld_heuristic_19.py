from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

try:
    # Pytrich typical import location
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback (some setups expose Heuristic at different module paths)
    from heuristic import Heuristic


class BlocksworldGPT52Heuristic19(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Design rationale (domain insights):
    - The only meaningful primitive work in this domain is manipulating blocks via
      {pick-up, put-down, stack, unstack}. HTN methods (do_clear, do_on_table,
      do_move, do_put_on) ultimately realize those.

    - In these instances, the search is guided by the current task network (TN):
      goals are mostly expressed as (do_put_on x y), (do_on_table x), and (do_clear x).
      Estimating remaining effort well requires considering BOTH:
        (1) the current world state (what is already true), and
        (2) the pending abstract tasks and their best-case decompositions.

    Heuristic approach (admissible, fast):
    - For each pending abstract task in the TN, compute a lower bound on the number
      of primitive actions still required to accomplish it *in the current state*.

    - We sum these per-task lower bounds (an additive relaxation). This can overcount
      shared work between tasks, but overcounting makes the heuristic potentially
      inadmissible. Therefore, we apply a conservative aggregation:

        h(state, TN) = max( max_task_lb , ceil(sum_task_lb / 2) )

      Explanation:
      - In blocksworld, most "useful" progress requires at least 2 primitive actions
        per achieved relation when starting from handempty: either
          unstack+stack (2) or pick-up+stack (2) or unstack+put-down (2), etc.
      - Using ceil(sum/2) is a safe dampening that reduces risk of overcounting.
      - Taking max with max_task_lb ensures we still respect a hard single-task lower bound.

    Per-task lower bounds (state-dependent):
    - do_put_on(x,y):
        0 if on(x,y) already.
        else at least 2 actions to place x onto y (pick/unstack + stack), plus
        additional mandatory actions to clear x and y if not clear.
        Clearing a block that is not clear requires (recursively) unstack+put-down
        for each block above it => at least 2 per blocking block.
        Additionally, if y is not on table, method m1 does (do_on_table y) first,
        which costs at least 2 if y is not already ontable.

    - do_on_table(x):
        0 if clear(x) (method m3 is nop). Otherwise need to clear x (handled by do_clear)
        and then if x not ontable, at least unstack+put-down (2).

      Note: In the domain, m3_do_on_table has precondition (clear x) and is nop,
      so achieving ontable is NOT enforced by that abstract task unless m2 triggers.
      We follow the domain semantics: if clear(x), cost 0.

    - do_clear(x):
        0 if clear(x), else 2 * (#blocks above x) as a lower bound.

    Efficiency considerations:
    - We NEVER call model.state_explicit_repr(state) for current states (too slow).
    - We pre-index all relevant facts (+on[a,b], +ontable[a], +clear[a], +handempty, +holding[a])
      into dictionaries mapping to fact global IDs to enable O(1) bit checks.
    - We precompute the "support" relation (which block is directly on which) from state
      using only on-facts that are true, and then compute how many blocks are above a given block
      by following the chain upward. This is O(k) per query where k is stack height; in random
      instances average height is small, and we memoize per evaluation.

    This heuristic is intended for search guidance and scalability on very large instances.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_19: str = "blocksworld_gpt52_heuristic_19"):
        super().__init__(model, blocksworld_gpt52_heuristic_19)
        self.model = model
        self.heuristic_name = blocksworld_gpt52_heuristic_19

        # Fact indices
        self._on_id: Dict[Tuple[str, str], int] = {}
        self._ontable_id: Dict[str, int] = {}
        self._clear_id: Dict[str, int] = {}
        self._holding_id: Dict[str, int] = {}
        self._handempty_id: Optional[int] = None

        # Parse and index facts once
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            gid = fact.global_id
            if pred == "on" and len(args) == 2:
                self._on_id[(args[0], args[1])] = gid
            elif pred == "ontable" and len(args) == 1:
                self._ontable_id[args[0]] = gid
            elif pred == "clear" and len(args) == 1:
                self._clear_id[args[0]] = gid
            elif pred == "holding" and len(args) == 1:
                self._holding_id[args[0]] = gid
            elif pred == "handempty" and len(args) == 0:
                self._handempty_id = gid

        # For fast task-name checks (depends on Pytrich's naming)
        self._task_prefixes = ("do_put_on", "do_on_table", "do_move", "do_clear")

    # ------------------------
    # Core interface
    # ------------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]]):
        """Return heuristic value for (state, task_network)."""
        if not task_network:
            return 0

        # Build direct relations from state: above_of[x]=y if on(x,y)
        above_of: Dict[str, str] = {}
        below_of: Dict[str, str] = {}
        # Iterate only over on-facts we know; check their truth via bit.
        for (x, y), gid in self._on_id.items():
            if (state_bitwise >> gid) & 1:
                above_of[x] = y
                below_of[y] = x

        # Memoize number of blocks above each queried block
        above_count_memo: Dict[str, int] = {}

        def above_count(b: str) -> int:
            if b in above_count_memo:
                return above_count_memo[b]
            cnt = 0
            cur = b
            while cur in below_of:
                cnt += 1
                cur = below_of[cur]
            above_count_memo[b] = cnt
            return cnt

        # Helpers: fast state checks
        def is_true_gid(gid: Optional[int]) -> bool:
            return gid is not None and ((state_bitwise >> gid) & 1) == 1

        def on(x: str, y: str) -> bool:
            return is_true_gid(self._on_id.get((x, y)))

        def clear(x: str) -> bool:
            return is_true_gid(self._clear_id.get(x))

        def ontable(x: str) -> bool:
            return is_true_gid(self._ontable_id.get(x))

        def holding(x: str) -> bool:
            return is_true_gid(self._holding_id.get(x))

        def handempty() -> bool:
            return is_true_gid(self._handempty_id)

        # If holding something, often need at least 1 action (put-down/stack)
        # to proceed with most methods that require handempty.
        hand_penalty = 0 if handempty() else 1

        # Compute per-task lower bounds
        lbs: List[int] = []
        max_lb = 0

        for t in task_network:
            # Primitive operators contribute 1 (they must be executed if present)
            # Abstract tasks: estimate.
            tname = getattr(t, "name", "")

            if not tname:
                continue

            # Operator: assume unit lower bound
            if hasattr(t, "pos_precons") and hasattr(t, "add_effects"):
                lb = 1
            else:
                # Abstract task parsing: names are typically like "do_put_on b1 b2"
                parts = tname.strip().split()
                base = parts[0] if parts else tname
                args = parts[1:]

                lb = 0
                if base == "do_clear" and len(args) == 1:
                    x = args[0]
                    lb = 0 if clear(x) else 2 * above_count(x)

                elif base == "do_on_table" and len(args) == 1:
                    x = args[0]
                    if clear(x):
                        lb = 0  # m3 is nop when clear
                    else:
                        # Need to clear x first (at least 2 per above block)
                        lb = 2 * above_count(x)
                        # If x not ontable, at least 2 to unstack+putdown (best case)
                        if not ontable(x):
                            lb += 2

                elif base == "do_move" and len(args) == 2:
                    # Move x onto y: similar to do_put_on but without the explicit do_on_table(y)
                    x, y = args[0], args[1]
                    if on(x, y):
                        lb = 0
                    else:
                        lb = 2  # (pick/unstack) + stack
                        if not clear(x):
                            lb += 2 * above_count(x)
                        if not clear(y):
                            lb += 2 * above_count(y)
                        lb += hand_penalty

                elif base == "do_put_on" and len(args) == 2:
                    x, y = args[0], args[1]
                    if on(x, y):
                        lb = 0
                    else:
                        # Base cost to place x onto y
                        lb = 2
                        # Need x and y clear
                        if not clear(x):
                            lb += 2 * above_count(x)
                        if not clear(y):
                            lb += 2 * above_count(y)
                        # Domain method m1 explicitly does do_on_table(y) first.
                        # If y is not ontable, at least 2 actions to get it there in best case.
                        if not ontable(y):
                            lb += 2
                        lb += hand_penalty

                else:
                    # Unknown/other abstract tasks: weak but safe
                    lb = 0

            lbs.append(lb)
            if lb > max_lb:
                max_lb = lb

        if not lbs:
            return 0

        sum_lb = sum(lbs)

        # Conservative aggregation to reduce overcounting risk while still being informative.
        h_val = max(max_lb, (sum_lb + 1) // 2)
        return int(h_val)

    # ------------------------
    # Fact parsing
    # ------------------------
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
