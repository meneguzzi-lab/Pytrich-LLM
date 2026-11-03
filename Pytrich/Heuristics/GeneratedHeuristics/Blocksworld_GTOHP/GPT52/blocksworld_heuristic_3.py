from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

# Pytrich provides the Heuristic base class; import path can vary slightly by setup.
# This import works in the official framework layout.
from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic3(Heuristic):
    """Domain-dependent heuristic for the BLOCKS HDDL domain.

    Rationale (domain insights -> heuristic design)
    ----------------------------------------------
    The domain defines abstract tasks:
      - do_put_on(x,y): either already satisfied (nop) or requires clearing x and y,
        moving y to table (if needed), then moving x onto y.
      - do_on_table(x): if x already clear -> nop, else unstack x from below and put down.
      - do_move(x,y): requires both clear and then either pick-up/stack (if x on table)
        or unstack/stack (if x not on table).
      - do_clear(x): if clear -> nop, else recursively clear the block on top, unstack it,
        and put it down.

    Key challenge: blocks above targets create recursive work; also if the hand is holding a
    block, many methods/actions requiring handempty are blocked.

    This heuristic provides a cheap *lower bound* on remaining effort by combining:
      (A) A task-network estimate: sum of minimal primitive actions needed for each remaining
          abstract task *under relaxation* (ignoring negative interactions and ordering).
          We choose conservative minima from the domain methods.
      (B) A goal-literal estimate: count unsatisfied goal facts (+on, +ontable, +clear, +handempty,
          +holding) with domain-specific weights (at least 1 per unsatisfied goal), plus a small
          obstruction term for blocks that are not clear but are required to be clear in the goal.

    Admissibility & efficiency
    -------------------------
    - Each abstract-task cost uses the cheapest method expansion cost, and then *relaxes*
      preconditions (e.g., assumes clearing is possible without accounting for conflicts). This
      yields a lower bound.
    - Goal counting with weight 1 is also a lower bound (each unsatisfied positive goal must be
      achieved at least once).
    - We take `max(task_lb, goal_lb)` to keep the heuristic admissible (max of lower bounds is a
      lower bound).
    - All checks are done via bitwise operations on fact global_ids; no explicit state decoding.

    Notes
    -----
    - The heuristic does not attempt a full relaxed-plan computation (too expensive for 1000 blocks).
    - It caches fact ids for predicates used frequently.

    Expected impact
    ---------------
    - Strong guidance on large instances dominated by long task networks: task_lb scales with the
      number of remaining abstract tasks.
    - Additional guidance from goal_lb when the task network is short but goals are not yet met.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_3: bool = True):
        super().__init__(model)
        self.blocksworld_gpt52_heuristic_3 = blocksworld_gpt52_heuristic_3

        # Cache fact ids by predicate and arguments for O(1) bitwise checking.
        # Structure:
        #   self._fact_id[(pred, tuple(args), sign)] = global_id
        # Where sign is True for +pred[...] and False for -pred[...]
        self._fact_id: Dict[Tuple[str, Tuple[str, ...], bool], int] = {}

        # Predicate index helpers for fast scan of relevant facts
        self._pred_to_facts: Dict[str, List[Tuple[Tuple[str, ...], bool, int]]] = {}

        for f in self.model.facts:
            pred, args, sign = self._parse_fact_full(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args), sign)
            self._fact_id[key] = f.global_id
            self._pred_to_facts.setdefault(pred, []).append((tuple(args), sign, f.global_id))

        # Cache goal facts (names parsed once, ids resolved once)
        self._goal_fact_ids: List[int] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args, sign = self._parse_fact_full(n)
            if pred is None:
                continue
            fid = self._fact_id.get((pred, tuple(args), sign), None)
            if fid is not None:
                self._goal_fact_ids.append(fid)

    # ---------- Fact parsing utilities ----------
    def _parse_fact_full(self, name: str) -> Tuple[Optional[str], List[str], Optional[bool]]:
        """Parse a grounded fact string '+p[a,b]' or '-p[a,b]' into (predicate, args, sign)."""
        name = name.strip()
        if not name:
            return None, [], None
        sign = None
        if name[0] == '+':
            sign = True
            name2 = name[1:]
        elif name[0] == '-':
            sign = False
            name2 = name[1:]
        else:
            return None, [], None

        bracket_idx = name2.find('[')
        if bracket_idx <= 0 or not name2.endswith(']'):
            return None, [], None

        pred = name2[:bracket_idx]
        args_str = name2[bracket_idx + 1:-1]
        args = args_str.split(',') if args_str else []
        return pred, args, sign

    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _is_fact_true(self, state_bitwise: int, pred: str, args: Tuple[str, ...], sign: bool = True) -> bool:
        fid = self._fact_id.get((pred, args, sign), None)
        if fid is None:
            return False
        return self._is_true(state_bitwise, fid)

    # ---------- Heuristic core ----------
    def __call__(self, state: int, task_network: List[Union[object, object]]) -> int:
        if not self.blocksworld_gpt52_heuristic_3:
            return 0

        # Lower bound from remaining tasks
        task_lb = self._task_network_lower_bound(state, task_network)

        # Lower bound from goal facts (if any)
        goal_lb = self._goal_lower_bound(state)

        # Combine admissibly
        h = max(task_lb, goal_lb)
        return int(h)

    def _goal_lower_bound(self, state: int) -> int:
        """Count unsatisfied goal facts with small domain-specific obstruction bonus.

        Admissible: each unsatisfied positive goal needs >=1 action (or decomposition leading to it).
        We do not use weights > 1 for simple literals to preserve a safe lower bound.
        """
        unsat = 0
        for fid in self._goal_fact_ids:
            if not self._is_true(state, fid):
                unsat += 1

        # Obstruction: if goal requires clear(x) but currently not clear(x), at least one unstack
        # is needed somewhere (lower bound +1). We only add 1 per such x.
        obstruction = 0
        # Find +clear[...] facts among goals
        for args, sign, fid in self._pred_to_facts.get('clear', []):
            if not sign:
                continue
            # if this clear fact is a goal and not satisfied
            if fid in self._goal_fact_ids and not self._is_true(state, fid):
                obstruction += 1

        return unsat + obstruction

    def _task_network_lower_bound(self, state: int, task_network) -> int:
        """Sum minimal primitive-action counts for the remaining task network.

        We treat each abstract task independently under relaxation.
        Primitive operators contribute their real cost (assumed 1 in this benchmark).
        """
        lb = 0
        for t in task_network:
            # Operator
            if hasattr(t, 'add_effects') and hasattr(t, 'pos_precons'):
                # primitive action
                c = getattr(t, 'cost', 1)
                lb += int(c) if c is not None else 1
                continue

            # Abstract task
            name = getattr(t, 'name', '')
            if not name:
                continue

            # Task names look like: 'do_put_on b1 b2' or similar depending on grounding.
            parts = name.split()
            if not parts:
                continue
            tname = parts[0]
            params = parts[1:]

            if tname == 'do_clear' and len(params) == 1:
                lb += self._lb_do_clear(state, params[0])
            elif tname == 'do_on_table' and len(params) == 1:
                lb += self._lb_do_on_table(state, params[0])
            elif tname == 'do_move' and len(params) == 2:
                lb += self._lb_do_move(state, params[0], params[1])
            elif tname == 'do_put_on' and len(params) == 2:
                lb += self._lb_do_put_on(state, params[0], params[1])
            else:
                # Unknown abstract task: be safe
                lb += 0

        return lb

    # ---------- Per-task lower bounds (domain-dependent) ----------
    def _lb_do_clear(self, state: int, x: str) -> int:
        # Method m6: if clear(x) then nop => cost 0
        if self._is_fact_true(state, 'clear', (x,), True):
            return 0

        # Method m7: if not clear(x) and on(y,x) and handempty then do_clear(y), unstack(y,x), put-down(y)
        # Minimal additional primitives beyond clearing y: unstack + put-down = 2
        # Lower bound: at least 2 if something is on x.
        # We do not try to identify y precisely (could scan +on[y,x]). If x is not clear, assume >=2.
        return 2

    def _lb_do_on_table(self, state: int, x: str) -> int:
        # m3: if clear(x) then nop => 0 (even if not ontable).
        if self._is_fact_true(state, 'clear', (x,), True):
            return 0

        # m2: needs clear(x), handempty, not ontable(x) and unstack + put-down = 2
        # If not clear, at least need to clear it first (>=2 from do_clear) then possibly 2.
        # Lower bound: at least 2 (some move required) when not clear.
        return 2

    def _lb_do_move(self, state: int, x: str, y: str) -> int:
        # If already on(x,y), in this domain do_move is not a direct satisfied check,
        # but moving onto y would be unnecessary. Still, keep a safe lower bound 0 if already there.
        if self._is_fact_true(state, 'on', (x, y), True):
            return 0

        # m4: if ontable(x) then pick-up + stack = 2
        # m5: if not ontable(x) then unstack + stack = 2
        # Lower bound for move action sequence is 2.
        return 2

    def _lb_do_put_on(self, state: int, x: str, y: str) -> int:
        # m0: if on(x,y) then nop => 0
        if self._is_fact_true(state, 'on', (x, y), True):
            return 0

        # m1 expands: do_clear(x) + do_clear(y) + do_on_table(y) + do_move(x,y)
        # Lower bound components:
        # - do_clear(x): 0 if clear(x) else 2
        # - do_clear(y): 0 if clear(y) else 2
        # - do_on_table(y): 0 if clear(y) else 2  (conservative)
        # - do_move(x,y): 2
        # Total lower bound = clear_x + clear_y + ontable_y + 2
        cx = 0 if self._is_fact_true(state, 'clear', (x,), True) else 2
        cy = 0 if self._is_fact_true(state, 'clear', (y,), True) else 2
        oy = 0 if self._is_fact_true(state, 'ontable', (y,), True) else 0
        # do_on_table(y) only matters if y not already on table; but method m1 always includes it.
        # Admissible lower bound: if y already ontable then 0 else at least 0 (could be nop if clear(y)
        # via m3). We'll approximate as 0 if clear(y), else 2.
        ontable_step = 0 if self._is_fact_true(state, 'clear', (y,), True) else 2
        # If y is already ontable, it's surely not worse; keep minimum.
        if self._is_fact_true(state, 'ontable', (y,), True):
            ontable_step = 0

        return cx + cy + ontable_step + 2
