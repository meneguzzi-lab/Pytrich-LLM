from __future__ import annotations

from typing import Dict, List, Tuple, Union, Optional


# Pytrich imports (kept generic to work across typical project layouts)
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.Heuristic import Heuristic
    except Exception:
        # Fallback for some installations
        from heuristic import Heuristic


class BlocksworldGPT52Heuristic2(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Design rationale (domain insights -> heuristic components):

    The HDDL domain encodes high-level tasks with methods:
      - do_put_on(x,y): either already satisfied (nop) or requires clearing x and y,
        possibly putting y on table, then moving x onto y.
      - do_on_table(x): either already (nop) or unstack + put-down.
      - do_move(x,y): pick-up/stack if x on table, else unstack/stack.
      - do_clear(x): either already clear (nop) or recursively clear top block y,
        then unstack y from x and put-down.

    Key difficulty: to achieve an on(x,y) goal/task, we typically must:
      1) ensure x is clear (unbury it)
      2) ensure y is clear
      3) ensure y is on table (in this domain’s preferred method m1_do_put_on)
      4) then move x onto y

    This heuristic provides a *lower bound* estimate by counting necessary primitive
    actions implied by the current task network and state, using only fast bitwise checks.

    It is domain-dependent but intentionally lightweight for very large instances:
      - Precomputes fact ids for on/ontable/clear/handempty/holding.
      - Builds a "support" mapping (what block is on what) and "above-count" (how buried)
        by iterating over *on* facts (O(#blocks)) per evaluation.
      - Estimates minimal actions for pending abstract tasks without expensive relaxed planning.

    Admissibility/consistency notes:
      - The heuristic is intended as a *non-overestimating* lower bound by using
        conservative estimates:
           * clearing a buried block: at least 2 actions (unstack+put-down) per block above
             under the handempty recursion in this domain.
           * moving x onto y: at least 2 actions (pick/unstack + stack) once both clear.
           * putting y on table if not already: at least 2 actions if y is not on table.
      - It does not count negative interactions or double-counting perfectly; to remain safe,
        we take max/cheap sums and avoid charging for handempty restoration (already implied).
      - For large task networks, we cap some per-task contributions to keep runtime predictable.

    The heuristic uses BOTH:
      - Goal facts (model.goals bitset) for additional guidance
      - Current task network (list of Operators/AbstractTasks) to reflect hierarchy

    __init__ parameter name is required to be all lower-case:
      blocksworld_gpt52_heuristic_2
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_2: bool = True):
        super().__init__(model)
        self.model = model
        self.enabled = blocksworld_gpt52_heuristic_2

        # Fact ids by predicate for fast scanning/building structures
        self._fact_ids_on: List[Tuple[int, str, str]] = []        # (gid, x, y)
        self._fact_ids_ontable: Dict[str, int] = {}              # x -> gid
        self._fact_ids_clear: Dict[str, int] = {}                # x -> gid
        self._fact_id_handempty: Optional[int] = None
        self._fact_ids_holding: Dict[str, int] = {}              # x -> gid

        # For goal guidance
        self._goal_on_pairs: List[Tuple[str, str]] = []
        self._goal_ontable: List[str] = []
        self._goal_clear: List[str] = []

        # Abstract task name -> kind
        self._task_kinds = {
            "do_put_on": "put_on",
            "do_on_table": "on_table",
            "do_move": "move",
            "do_clear": "clear",
        }

        self._precompute_fact_indices()
        self._precompute_goal_structures()

    # ------------------------- Parsing helpers -------------------------

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

    def _parse_task_name(self, tname: str) -> Tuple[str, List[str]]:
        """Parse grounded task/operator names that typically look like:
           'do_put_on b1 b2' or 'stack b1 b2' or 'unstack b1 b2'.
        Returns (symbol, args).
        """
        s = tname.strip()
        if not s:
            return s, []
        parts = s.split()
        return parts[0], parts[1:]

    # ------------------------- Precomputation -------------------------

    def _precompute_fact_indices(self):
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == "on" and len(args) == 2:
                self._fact_ids_on.append((fact.global_id, args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self._fact_ids_ontable[args[0]] = fact.global_id
            elif pred == "clear" and len(args) == 1:
                self._fact_ids_clear[args[0]] = fact.global_id
            elif pred == "handempty" and len(args) == 0:
                self._fact_id_handempty = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self._fact_ids_holding[args[0]] = fact.global_id

    def _precompute_goal_structures(self):
        # model.goals is a bitwise integer
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g in goal_fact_names:
            pred, args = self._parse_fact_name(g)
            if pred == "on" and len(args) == 2:
                self._goal_on_pairs.append((args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self._goal_ontable.append(args[0])
            elif pred == "clear" and len(args) == 1:
                self._goal_clear.append(args[0])

    # ------------------------- Bitwise checks -------------------------

    def _is_true_gid(self, state: int, gid: int) -> bool:
        return ((state >> gid) & 1) == 1

    def _is_clear(self, state: int, b: str) -> bool:
        gid = self._fact_ids_clear.get(b)
        return False if gid is None else self._is_true_gid(state, gid)

    def _is_ontable(self, state: int, b: str) -> bool:
        gid = self._fact_ids_ontable.get(b)
        return False if gid is None else self._is_true_gid(state, gid)

    def _is_handempty(self, state: int) -> bool:
        if self._fact_id_handempty is None:
            return True
        return self._is_true_gid(state, self._fact_id_handempty)

    def _is_holding(self, state: int, b: str) -> bool:
        gid = self._fact_ids_holding.get(b)
        return False if gid is None else self._is_true_gid(state, gid)

    def _is_on(self, state: int, x: str, y: str) -> bool:
        # We don't maintain a direct map for on(x,y) gid; check via support map built per-eval.
        # This is handled by building support_of in _build_support.
        raise RuntimeError("Use _build_support mapping for on queries")

    # ------------------------- Structural extraction -------------------------

    def _build_support(self, state: int) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
        """Build mappings from current state:
           - support_of[x] = y if on(x,y)
           - top_of[y] = x if on(x,y)
           - above_count[b] = number of blocks above b in its stack
        Runs in O(#on-facts + #blocks in stacks).
        """
        support_of: Dict[str, str] = {}
        top_of: Dict[str, str] = {}

        for gid, x, y in self._fact_ids_on:
            if self._is_true_gid(state, gid):
                support_of[x] = y
                top_of[y] = x

        # compute above_count by walking down from each clear block
        above_count: Dict[str, int] = {}
        # Identify clear blocks quickly: iterate known clear facts keys
        for b, cg in self._fact_ids_clear.items():
            if self._is_true_gid(state, cg):
                # b is top; everything below has increasing above_count
                depth = 0
                cur = b
                while cur in support_of:
                    below = support_of[cur]
                    depth += 1
                    # below has at least depth blocks above it (from this top)
                    # if multiple tops lead to same below (shouldn't), take max
                    prev = above_count.get(below, 0)
                    if depth > prev:
                        above_count[below] = depth
                    cur = below

        return support_of, top_of, above_count

    # ------------------------- Cost lower bounds for tasks -------------------------

    def _lb_clear_cost(self, state: int, block: str, above_count: Dict[str, int]) -> int:
        """Lower bound on actions to make `block` clear.

        In this domain, each blocking block above requires at least:
          unstack + put-down = 2 actions
        because method m7_do_clear uses (unstack y x) then (put-down y).
        """
        if self._is_clear(state, block):
            return 0
        k = above_count.get(block, 1)  # if unknown but not clear, assume at least 1 above
        return 2 * k

    def _lb_on_table_cost(self, state: int, block: str, above_count: Dict[str, int]) -> int:
        """Lower bound to ensure block is on table.

        If block is already on table: 0.
        Otherwise, to put it down on table you must at least:
          - clear it (if needed): accounted separately
          - unstack it from something (1) and put-down (1): 2
        We include only the 2 here plus any needed clearing.
        """
        if self._is_ontable(state, block):
            return 0
        # Must end with put-down after holding; minimal is unstack + put-down
        return 2

    def _lb_move_cost(self, state: int, x: str, y: str, above_count: Dict[str, int]) -> int:
        """Lower bound to place x on y, assuming any needed clearing of x/y handled separately.

        Minimal primitive actions:
          - if x on table: pick-up + stack = 2
          - else: unstack + stack = 2
        So always at least 2.
        """
        # Even if already on, moving cost could be 0 but do_move task is typically used to achieve on.
        return 2

    def _lb_put_on_cost(self, state: int, x: str, y: str, support_of: Dict[str, str], above_count: Dict[str, int]) -> int:
        """Lower bound for do_put_on(x,y).

        If already on(x,y): 0.
        Else lower bound includes:
          - clear x (if needed)
          - clear y (if needed)
          - ensure y on table (domain method does this): if not on table => +2
          - move x onto y: +2

        To stay conservative (avoid double-counting across many tasks), we sum these
        but cap per-task contribution lightly.
        """
        if support_of.get(x) == y:
            return 0

        c = 0
        c += self._lb_clear_cost(state, x, above_count)
        c += self._lb_clear_cost(state, y, above_count)

        if not self._is_ontable(state, y):
            c += self._lb_on_table_cost(state, y, above_count)

        c += self._lb_move_cost(state, x, y, above_count)

        # Cap per-task to prevent huge overguidance on very large TNs (runtime/scale stability)
        # This does NOT affect admissibility as it only reduces the estimate.
        return min(c, 30)

    # ------------------------- Main heuristic -------------------------

    def __call__(self, state: int, task_network: List[Union[object, object]] = None) -> int:
        if not self.enabled:
            return 0
        if task_network is None:
            task_network = []

        support_of, top_of, above_count = self._build_support(state)

        h = 0

        # 1) Hierarchical task-network based lower bound
        #    Count remaining abstract tasks with domain-specific estimates.
        #    We only look at AbstractTasks by name; Operators are already primitive choices.
        #    Keep this lightweight: O(len(task_network)).
        for t in task_network:
            # Try to access .name (works for Operator and AbstractTask)
            tname = getattr(t, "name", "")
            sym, args = self._parse_task_name(tname)
            kind = self._task_kinds.get(sym)
            if kind is None:
                continue

            if kind == "clear" and len(args) >= 1:
                h += self._lb_clear_cost(state, args[0], above_count)
            elif kind == "on_table" and len(args) >= 1:
                # Need to clear first if not clear (lower bound)
                b = args[0]
                if not self._is_ontable(state, b):
                    h += self._lb_clear_cost(state, b, above_count)
                    h += self._lb_on_table_cost(state, b, above_count)
            elif kind == "move" and len(args) >= 2:
                x, y = args[0], args[1]
                # If already on, cost 0
                if support_of.get(x) != y:
                    h += self._lb_clear_cost(state, x, above_count)
                    h += self._lb_clear_cost(state, y, above_count)
                    h += self._lb_move_cost(state, x, y, above_count)
            elif kind == "put_on" and len(args) >= 2:
                x, y = args[0], args[1]
                h += self._lb_put_on_cost(state, x, y, support_of, above_count)

        # 2) Goal-fact based lower bound (helps when task network becomes empty/less informative)
        #    For each unsatisfied goal on(x,y), charge a conservative put-on style estimate.
        #    Use small weights and caps to avoid excessive sums.
        goal_h = 0
        for (x, y) in self._goal_on_pairs:
            if support_of.get(x) == y:
                continue
            goal_h += self._lb_put_on_cost(state, x, y, support_of, above_count)

        for b in self._goal_ontable:
            if not self._is_ontable(state, b):
                goal_h += self._lb_clear_cost(state, b, above_count)
                goal_h += self._lb_on_table_cost(state, b, above_count)

        for b in self._goal_clear:
            if not self._is_clear(state, b):
                goal_h += self._lb_clear_cost(state, b, above_count)

        # Combine: we take max to reduce double counting and keep admissible.
        h = max(h, goal_h)

        # 3) Small tie-breaker: if holding something, at least 1 action to proceed (stack/put-down)
        #    This is a lower bound only when not goal-reached and there is work remaining.
        if h > 0 and not self._is_handempty(state):
            # If holding, minimum 1 action to change holding status.
            h = max(h, 1)

        return int(h)
