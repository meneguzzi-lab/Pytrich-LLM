from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Pytrich imports (expected in the evaluation environment)
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.model import Model


class BlocksworldGPT52Heuristic17(Heuristic):
    """Domain-dependent heuristic for the BLOCKS (Blocksworld) HDDL domain.

    Core idea (fast + domain-specific + hierarchy-aware):
    - The search is hierarchical: nodes contain a *state* (bitset) and a *task network*.
    - In this domain, every abstract task ultimately reduces to a small, structured set of
      primitive actions. We exploit this by computing a *lower bound* cost for each
      pending abstract task from the current state.

    We use a *max* of per-task lower bounds (admissible under non-negative costs) and also
    incorporate a cheap goal-fact mismatch signal (also admissible).

    Per-task lower bounds (state-based, relaxed but informative):

    1) do_put_on(x,y)
       - If on(x,y) already true => 0.
       - Otherwise must eventually execute a stack(x,y) (or reach on(x,y)).
         Lower bound includes:
           + if holding(x) and clear(y): 1 (stack)
           + if holding(x) and not clear(y): 1 (need to clear y) + 1 (stack) => 2
           + if not holding(x): at least 2 (pick/unstack + stack) in best case.
         Additionally, if x is not clear now, at least 2 actions are needed to clear x's top
         block (unstack + put-down), but we only count a +1 for "x not clear" to keep
         it safely admissible and cheap.

    2) do_on_table(x)
       - If ontable(x) already true => 0.
       - Else, at minimum must put-down(x). To put-down, must first hold(x).
         Best case: holding(x) => 1, else => 2 (unstack/pick + put-down).

    3) do_move(x,y)
       - If on(x,y) already true => 0.
       - Else similar to do_put_on but with extra requirement in methods that x must be on table
         or unstacked. Lower bound: 2 if not holding(x), 1 if holding(x) and clear(y).

    4) do_clear(x)
       - If clear(x) already true => 0.
       - Else there exists some block y on x, and the domain method clears by recursively
         do_clear(y), then unstack(y,x), then put-down(y). Even in best case, that is >= 2
         actions after y is clear; thus we return 2 if x not clear.

    Goal mismatch component (very cheap):
    - Count number of unsatisfied +on[a,b] and +ontable[a] in the *problem goal facts*.
    - Each such predicate needs at least 1 primitive action to establish (in reality usually more),
      so this is an admissible lower bound. We include it via max() with the HTN estimate.

    Final heuristic:
      h(s,tn) = max( goal_mismatch_lb(s), max_{t in tn} task_lb(s,t) )

    This is admissible and consistent enough for A*-like search while being extremely fast
    (bit operations + O(|tn|)).

    Notes on implementation:
    - We NEVER call model.state_explicit_repr(state) for current states (too slow).
    - We pre-index fact global_ids for on/ontable/clear/holding/handempty.
    - We parse operator/task names conservatively and robustly.
    """

    def __init__(self, blocksworld_gpt52_heuristic_17: Model):
        super().__init__(blocksworld_gpt52_heuristic_17)
        self.model: Model = blocksworld_gpt52_heuristic_17

        # --- Precompute fact indices for fast bit checks ---
        # Maps for predicates to arg-tuple -> fact_global_id
        self._fact_on: Dict[Tuple[str, str], int] = {}
        self._fact_ontable: Dict[Tuple[str], int] = {}
        self._fact_clear: Dict[Tuple[str], int] = {}
        self._fact_holding: Dict[Tuple[str], int] = {}
        self._fact_handempty: Optional[int] = None

        # Also keep reverse "what is on top of x" lookup (from facts only, static universe)
        # For each support block y, list of possible x with on(x,y) facts.
        self._possible_on_top_of: Dict[str, List[str]] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                x, y = args[0], args[1]
                self._fact_on[(x, y)] = f.global_id
                self._possible_on_top_of.setdefault(y, []).append(x)
            elif pred == "ontable" and len(args) == 1:
                self._fact_ontable[(args[0],)] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self._fact_clear[(args[0],)] = f.global_id
            elif pred == "holding" and len(args) == 1:
                self._fact_holding[(args[0],)] = f.global_id
            elif pred == "handempty" and len(args) == 0:
                self._fact_handempty = f.global_id

        # --- Cache goal facts we care about (explicit repr is OK once for model.goals) ---
        self._goal_on: List[Tuple[str, str]] = []
        self._goal_ontable: List[Tuple[str]] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                self._goal_on.append((args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self._goal_ontable.append((args[0],))

    # ------------------------
    # Helpers
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

    def _is_true_gid(self, state: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _is_on(self, state: int, x: str, y: str) -> bool:
        return self._is_true_gid(state, self._fact_on.get((x, y)))

    def _is_ontable(self, state: int, x: str) -> bool:
        return self._is_true_gid(state, self._fact_ontable.get((x,)))

    def _is_clear(self, state: int, x: str) -> bool:
        return self._is_true_gid(state, self._fact_clear.get((x,)))

    def _is_holding(self, state: int, x: str) -> bool:
        return self._is_true_gid(state, self._fact_holding.get((x,)))

    def _exists_block_on(self, state: int, y: str) -> bool:
        """Return True if there exists some x such that on(x,y) is true in state.

        We avoid scanning all facts; instead, scan only candidates from static index.
        """
        candidates = self._possible_on_top_of.get(y)
        if not candidates:
            return False
        for x in candidates:
            if self._is_on(state, x, y):
                return True
        return False

    def _goal_mismatch_lb(self, state: int) -> int:
        """Admissible LB: number of unsatisfied goal predicates (+on, +ontable).

        Each unsatisfied atomic goal requires at least one action to achieve.
        """
        miss = 0
        for x, y in self._goal_on:
            if not self._is_on(state, x, y):
                miss += 1
        for (x,) in self._goal_ontable:
            if not self._is_ontable(state, x):
                miss += 1
        return miss

    # ------------------------
    # Task lower bounds
    # ------------------------

    def _lb_do_clear(self, state: int, x: str) -> int:
        if self._is_clear(state, x):
            return 0
        # If something is on x, in the best case that top block is already clear,
        # then unstack + put-down => 2 actions.
        if self._exists_block_on(state, x):
            return 2
        # In well-formed states, not clear implies something on it, but be robust.
        return 1

    def _lb_do_on_table(self, state: int, x: str) -> int:
        if self._is_ontable(state, x):
            return 0
        if self._is_holding(state, x):
            return 1  # put-down
        # Otherwise need to get it into hand (pick-up or unstack) then put-down.
        # Lower bound 2.
        return 2

    def _lb_do_move_like(self, state: int, x: str, y: str) -> int:
        # shared structure for do_put_on and do_move
        if self._is_on(state, x, y):
            return 0

        holding_x = self._is_holding(state, x)
        clear_y = self._is_clear(state, y)
        clear_x = self._is_clear(state, x)

        if holding_x:
            # Need stack, plus possibly clear(y)
            return 1 if clear_y else 2

        # Not holding x: at least need acquire x then stack => >=2
        lb = 2

        # If x is not clear, we certainly must do something to free it.
        # Clearing x's top in this domain uses unstack+put-down at minimum.
        # Add only +1 to remain conservative and avoid overcounting with the base 2.
        if not clear_x:
            lb += 1

        # If y is not clear, must clear y before stack.
        if not clear_y:
            lb += 1

        return lb

    def _lb_do_put_on(self, state: int, x: str, y: str) -> int:
        return self._lb_do_move_like(state, x, y)

    def _lb_do_move(self, state: int, x: str, y: str) -> int:
        return self._lb_do_move_like(state, x, y)

    def _task_lb(self, state: int, task) -> int:
        """Compute LB for either an operator (0) or an abstract task by name."""
        # Primitive operator: optimistic 0 because it can be applied immediately or later.
        # The heuristic is meant to guide decompositions primarily.
        if hasattr(task, "cost"):
            return 0

        name = getattr(task, "name", "")
        if not name:
            return 0

        # Task names in grounded HTN are typically like "do_put_on b1 b2" or "(do_put_on b1 b2)"
        # Pytrich often stores them without parentheses. We handle both.
        s = name.strip()
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()
        parts = s.split()
        if not parts:
            return 0

        tname = parts[0]
        args = parts[1:]

        if tname == "do_clear" and len(args) == 1:
            return self._lb_do_clear(state, args[0])
        if tname == "do_on_table" and len(args) == 1:
            return self._lb_do_on_table(state, args[0])
        if tname == "do_put_on" and len(args) == 2:
            return self._lb_do_put_on(state, args[0], args[1])
        if tname == "do_move" and len(args) == 2:
            return self._lb_do_move(state, args[0], args[1])

        return 0

    # ------------------------
    # Heuristic API
    # ------------------------

    def __call__(self, state: int, task_network: List) -> int:
        # If no remaining tasks, rely purely on goal mismatch (still admissible).
        goal_lb = self._goal_mismatch_lb(state)

        if not task_network:
            return goal_lb

        # Hierarchy-aware LB: take maximum over pending tasks.
        # (Admissible: remaining plan must achieve *all* tasks, so cost >= each individual LB)
        max_task_lb = 0
        for t in task_network:
            lb = self._task_lb(state, t)
            if lb > max_task_lb:
                max_task_lb = lb

        return max(goal_lb, max_task_lb)
