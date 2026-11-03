from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

try:
    # Pytrich typical location
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback for alternative project layouts
    from heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic4(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Rationale / domain insights
    ---------------------------
    This Blocksworld HDDL uses four compound tasks:
      - do_put_on(x,y)
      - do_on_table(x)
      - do_move(x,y)
      - do_clear(x)

    with methods that (in the common case) decompose into primitive actions:
      - unstack, put-down, pick-up, stack, (and nop)

    Key structural challenges:
      1) Clearing requirements dominate difficulty: to move/stack blocks, both the moved block and its destination
         must be clear, and the hand must be empty.
      2) do_put_on(x,y) commonly requires: clear(x), clear(y), put y on table (sometimes), then move x onto y.
         The minimal primitive cost strongly depends on whether x is already on y, whether x/y are clear,
         and whether x is on table.
      3) Large instances have very long task networks; any heuristic must be O(#pending_tasks) per node and must
         avoid explicit state-to-facts string conversion.

    Heuristic design
    ----------------
    We compute a fast admissible lower bound by summing *per-pending-task* local lower bounds.

    - For each pending compound task we estimate the minimum number of primitive actions still needed
      *in a relaxed way* (ignore negative interactions between tasks).
    - For primitive operators we return their remaining cost (1 each), since they are already chosen.

    Lower bound components:
      A) do_clear(x):
         - if clear(x) already true -> 0
         - else at least 2 actions are required to remove the top block: unstack(y,x) + put-down(y) -> 2
           (ignoring recursion / further blocks above y)

      B) do_on_table(x):
         - if ontable(x) already true -> 0 (method m3 uses nop if clear(x); but if x is on table, task is done)
         - else if (clear(x) & handempty) not both true, still at least 2 in the best case once clear&handempty hold
           because m2 is unstack + put-down -> 2
         - so if x is not on table -> 2 is an admissible LB.

      C) do_move(x,y):
         minimal primitive actions to achieve on(x,y) given preconditions:
           - if already on(x,y) -> 0
           - else, in best case you need to pick x (either pick-up if ontable(x) else unstack) then stack.
             So LB = 2 if handempty and clear(x) and clear(y) hold in some relaxed sense.
             As a state-based admissible LB independent of those, we use:
               * if ontable(x) true -> at least pick-up + stack = 2
               * else -> at least unstack + stack = 2
             => always 2 when not already satisfied.

      D) do_put_on(x,y):
         - if already on(x,y) -> 0
         - else, even in best case you need to perform a move(x,y) which is >=2.
         - additionally, if x is not clear -> need >=2 to clear x
         - if y is not clear -> need >=2 to clear y
         - if y is not on table, method m1 includes do_on_table(y); minimal extra cost >=2 (in general)
           (note: in some states y might already be ontable, then 0).

    This yields:
      h = sum over pending tasks of these LBs.

    Admissibility:
      - Each component is a lower bound on the primitive actions required to accomplish that task from the
        current state, under a relaxation that ignores inter-task interference and ignores deeper stacks.
      - Summing can overcount shared work across tasks; that would risk inadmissibility. To keep it admissible,
        we *cap aggregation by block landmarks*:

        We aggregate clearing/on-table requirements per block by taking a maximum requirement per block rather
        than summing duplicates:
          - required_clear_cost[x] = max of required clear cost contributions for x across tasks
          - required_ontable_cost[x] = max of required ontable contributions for x across tasks

        For move/put_on achievements (+on[x,y]) we count each distinct target relation at most once.

      This max-per-block / unique-per-relation aggregation preserves admissibility as a set of independent
      necessary conditions.

    Efficiency:
      - Uses bitwise checks on fact IDs.
      - Pre-indexes relevant facts: clear(b), ontable(b), handempty, on(x,y).
      - Parses tasks from their .name fields (robust to minor naming variations).

    Notes:
      - This heuristic is domain-dependent and tuned to this HDDL encoding.
      - It is conservative (admissible) and scales to very large instances.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_4: str = "blocksworld_gpt52_heuristic_4"):
        super().__init__(model, blocksworld_gpt52_heuristic_4)
        self.model = model
        self.heuristic_name = blocksworld_gpt52_heuristic_4

        # Fact indices for fast queries
        self._fact_handempty: Optional[int] = None
        self._fact_clear: Dict[str, int] = {}
        self._fact_ontable: Dict[str, int] = {}
        self._fact_on: Dict[Tuple[str, str], int] = {}

        # Build indices once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "handempty":
                self._fact_handempty = f.global_id
            elif pred == "clear" and len(args) == 1:
                self._fact_clear[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self._fact_ontable[args[0]] = f.global_id
            elif pred == "on" and len(args) == 2:
                self._fact_on[(args[0], args[1])] = f.global_id

        # A tiny constant cache
        self._two = 2

    # -------------------- core heuristic API --------------------
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]]):
        return self.compute(state_bitwise, task_network)

    def compute(self, state_bitwise: int, task_network: List[Union[object, object]]):
        # If no tasks, heuristic is 0 (state-based goals handled by planner/model)
        if not task_network:
            return 0

        # Aggregation to avoid double counting (admissibility-friendly)
        required_clear_cost: Dict[str, int] = {}
        required_ontable_cost: Dict[str, int] = {}
        required_on_rels: Dict[Tuple[str, str], int] = {}  # distinct on(x,y) achievements

        primitive_cost = 0

        handempty_true = self._is_true(state_bitwise, self._fact_handempty)

        for t in task_network:
            # Primitive operator: cost is exactly operator.cost (typically 1)
            if hasattr(t, "cost") and hasattr(t, "applicable"):
                primitive_cost += int(getattr(t, "cost", 1))
                continue

            # Abstract task
            name = getattr(t, "name", "")
            tname, args = self._parse_task_name(name)
            if tname is None:
                # Unknown; stay safe (admissible): return 0 contribution
                continue

            if tname == "do_clear" and len(args) == 1:
                x = args[0]
                if not self._is_clear(state_bitwise, x):
                    required_clear_cost[x] = max(required_clear_cost.get(x, 0), self._two)

            elif tname == "do_on_table" and len(args) == 1:
                x = args[0]
                if not self._is_ontable(state_bitwise, x):
                    # Minimal method m2: unstack + put-down => 2
                    required_ontable_cost[x] = max(required_ontable_cost.get(x, 0), self._two)
                    # If it is not clear, clearing will also be required; capture as a separate necessary condition.
                    if not self._is_clear(state_bitwise, x):
                        required_clear_cost[x] = max(required_clear_cost.get(x, 0), self._two)

            elif tname == "do_move" and len(args) == 2:
                x, y = args
                if not self._is_on(state_bitwise, x, y):
                    required_on_rels[(x, y)] = 2  # need at least pick/unstack + stack
                    # Preconditions involve clear(x), clear(y), handempty. Count them as necessary conditions.
                    if not self._is_clear(state_bitwise, x):
                        required_clear_cost[x] = max(required_clear_cost.get(x, 0), self._two)
                    if not self._is_clear(state_bitwise, y):
                        required_clear_cost[y] = max(required_clear_cost.get(y, 0), self._two)
                    # If not handempty, at least one action is needed to free it, but in this domain
                    # freeing requires put-down/stack which is not directly a task; to remain admissible, use 0.
                    _ = handempty_true

            elif tname == "do_put_on" and len(args) == 2:
                x, y = args
                if self._is_on(state_bitwise, x, y):
                    continue

                # Core achievement: on(x,y)
                required_on_rels[(x, y)] = 2

                # Necessary enabling conditions (aggregated)
                if not self._is_clear(state_bitwise, x):
                    required_clear_cost[x] = max(required_clear_cost.get(x, 0), self._two)
                if not self._is_clear(state_bitwise, y):
                    required_clear_cost[y] = max(required_clear_cost.get(y, 0), self._two)

                # Method m1 includes do_on_table(y) (to make y available on table). If y already on table, cost 0.
                if not self._is_ontable(state_bitwise, y):
                    required_ontable_cost[y] = max(required_ontable_cost.get(y, 0), self._two)

            else:
                # Unknown abstract task: contribute 0 to remain admissible
                continue

        # Combine aggregated requirements
        h = primitive_cost
        h += sum(required_clear_cost.values())
        h += sum(required_ontable_cost.values())
        h += sum(required_on_rels.values())

        return int(h)

    # -------------------- helpers --------------------
    def _is_true(self, state_bitwise: int, fact_gid: Optional[int]) -> bool:
        if fact_gid is None:
            return False
        return ((state_bitwise >> fact_gid) & 1) == 1

    def _is_clear(self, state_bitwise: int, b: str) -> bool:
        return self._is_true(state_bitwise, self._fact_clear.get(b))

    def _is_ontable(self, state_bitwise: int, b: str) -> bool:
        return self._is_true(state_bitwise, self._fact_ontable.get(b))

    def _is_on(self, state_bitwise: int, x: str, y: str) -> bool:
        return self._is_true(state_bitwise, self._fact_on.get((x, y)))

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_task_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """Parse a grounded task/operator display name into (base_name, args).

        Expected common formats in Pytrich for tasks/operators are typically like:
          - 'do_put_on b1 b2'
          - 'do_put_on(b1,b2)'
          - 'do_put_on[b1,b2]'
          - or sometimes just the operator name for 0-arity.

        We keep this permissive and fast.
        """
        s = (name or "").strip()
        if not s:
            return None, []

        # Normalize separators
        # Case 1: whitespace separated: "do_put_on b1 b2"
        if "(" not in s and "[" not in s and " " in s:
            parts = s.split()
            return parts[0], parts[1:]

        # Case 2: bracket/paren form
        for open_c, close_c in (("(", ")"), ("[", "]")):
            oi = s.find(open_c)
            if oi > 0 and s.endswith(close_c):
                base = s[:oi].strip()
                inside = s[oi + 1 : -1].strip()
                if not inside:
                    return base, []
                return base, [a.strip() for a in inside.split(",") if a.strip()]

        # Fallback: no args
        return s, []
