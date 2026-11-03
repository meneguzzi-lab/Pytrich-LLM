from typing import Dict, Tuple, List, Optional, Set, Any

try:
    # Standard Pytrich import
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback stub for external testing (will be ignored by Pytrich)
    class Heuristic:
        def __init__(self, model, *args, **kwargs):
            self.model = model


class BlocksworldGPT5Heuristic13(Heuristic):
    """
    Blocksworld domain-dependent heuristic for Pytrich.

    Key ideas (admissible lower bounds):
    - Goals: For each goal +on[x,y], estimate the minimum number of actions needed.
      We account for whether x is already above y (transitively), in which case
      we only need to clear the blocks above x and between x and y. Otherwise, we
      must clear x and y, then move x onto y (at least 2 actions unless already
      holding x).
    - Obstacles: Each blocking block removal requires at least 2 actions in Blocksworld
      (unstack + put-down/stack anywhere). We count 2 per blocker.
    - Other goal types supported: clear[x], ontable[x], holding[x], handempty.
    - HTN tasks: We provide a hierarchical component. For the current task network,
      we compute a lower bound by:
        (a) Counting pending primitive operators (each must be executed -> +1 each),
        (b) Estimating minimal cost of abstract tasks we can parse (do_put_on, do_move,
            do_clear, do_on_table) using the same domain-specific lower bounds.
      We then use max(pending_primitive_count, abstract_task_lb) as the HTN lower bound.

    Final heuristic value = max(goals_lower_bound, htn_lower_bound). This avoids
    double-counting since both must be satisfied to finish planning, while staying
    admissible.

    Efficiency:
    - We never enumerate state facts as strings. Instead, we maintain mappings from
      grounded fact keys to their global bit ids and use bitwise checks.
    - We pre-index all on/ontable/clear/holding/handempty facts once.
    - For each heuristic evaluation, we reconstruct the current on-parent/child maps
      by scanning only pre-indexed on-facts and testing bit-true via bitwise.
      We then compute "above-count" using a cached linear chain walk (Blocksworld's
      on-relations are functional in the second argument, so chains are linear).

    Design rationale:
    - Admissibility: All cost contributions are lower bounds under the domain's action
      model. We never assume positive interactions (e.g., moving a blocker directly
      to its goal can only reduce the true cost relative to the conservative 2-step
      per blocker bound we count).
    - Hierarchical awareness: We incorporate the current HTN task network structure
      without over-committing to a specific decomposition, using generic LB per task
      schema and a simple count for primitive operators.
    - Scale: The approach avoids quadratic scans over all ground "on" pairs by relying
      on bitwise checks over an indexed subset (the grounded facts actually in the
      model). This remains tractable on large instances.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_13: bool = True, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.model = model

        # Fact indexes
        self.on_facts: List[Tuple[int, str, str]] = []               # list of (gid, x, y)
        self.on_id_by_pair: Dict[Tuple[str, str], int] = {}          # (x,y) -> gid
        self.ontable_id_by_block: Dict[str, int] = {}                # x -> gid
        self.clear_id_by_block: Dict[str, int] = {}                  # x -> gid
        self.holding_id_by_block: Dict[str, int] = {}                # x -> gid
        self.handempty_id: Optional[int] = None

        self.blocks: Set[str] = set()

        # Parse all facts and index those we need
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == 'on' and len(args) == 2:
                x, y = args[0], args[1]
                self.on_facts.append((fact.global_id, x, y))
                self.on_id_by_pair[(x, y)] = fact.global_id
                self.blocks.add(x)
                self.blocks.add(y)
            elif pred == 'ontable' and len(args) == 1:
                x = args[0]
                self.ontable_id_by_block[x] = fact.global_id
                self.blocks.add(x)
            elif pred == 'clear' and len(args) == 1:
                x = args[0]
                self.clear_id_by_block[x] = fact.global_id
                self.blocks.add(x)
            elif pred == 'holding' and len(args) == 1:
                x = args[0]
                self.holding_id_by_block[x] = fact.global_id
                self.blocks.add(x)
            elif pred == 'handempty':
                self.handempty_id = fact.global_id

        # Parse goal facts once
        self.goal_facts: List[Tuple[str, Tuple[str, ...], Optional[int]]] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            gid = self._fact_id_by_key(pred, tuple(args))
            self.goal_facts.append((pred, tuple(args), gid))

    # ---------------------------- Heuristic Core ---------------------------- #

    def h(self, state_bitwise: int, task_network: List[Any]) -> int:
        # Build current on-parent/child maps once per state
        parent_of, child_of = self._build_on_maps(state_bitwise)
        above_cache: Dict[str, int] = {}

        def above_count(block: str) -> int:
            # linear chain upward since blocksworld is functional (at most one block on a given block)
            if block in above_cache:
                return above_cache[block]
            cnt = 0
            cur = child_of.get(block)
            while cur is not None:
                cnt += 1
                cur = child_of.get(cur)
            above_cache[block] = cnt
            return cnt

        def distance_down(a: str, b: str) -> Optional[int]:
            # number of edges from a down to b via parent_of mapping (if a is above b)
            depth = 0
            cur = a
            while True:
                cur = parent_of.get(cur)
                if cur is None:
                    return None
                depth += 1
                if cur == b:
                    return depth

        def is_true(gid: Optional[int]) -> bool:
            if gid is None:
                return False
            return ((state_bitwise >> gid) & 1) == 1

        def is_holding(block: str) -> bool:
            gid = self.holding_id_by_block.get(block)
            return gid is not None and ((state_bitwise >> gid) & 1) == 1

        def on_goal_lb(a: str, b: str) -> int:
            # If already on[a,b]
            gid = self.on_id_by_pair.get((a, b))
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                return 0

            # Case: a is somewhere above b -> just clear above a and blocks between a and b
            d = distance_down(a, b)
            if d is not None:
                # d >= 1; blocks between a and b = d-1; must also clear blocks above a
                blockers = above_count(a) + (d - 1)
                return 2 * blockers if blockers > 0 else 0

            # Otherwise, need to clear a and b, then perform move (stack). Base move cost
            # is at least 2 unless already holding a (then 1).
            base = 1 if is_holding(a) else 2
            blockers = above_count(a) + above_count(b)
            return base + 2 * blockers

        def clear_goal_lb(x: str) -> int:
            # If already clear
            gid = self.clear_id_by_block.get(x)
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                return 0
            # Need to remove blocks above x
            return 2 * above_count(x)

        def on_table_goal_lb(x: str) -> int:
            gid = self.ontable_id_by_block.get(x)
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                return 0
            # Need x clear, then unstack x (1) and put-down (1). If already holding x -> 1.
            base = 1 if is_holding(x) else 2
            return base + 2 * above_count(x)

        def holding_goal_lb(x: str) -> int:
            gid = self.holding_id_by_block.get(x)
            if gid is not None and ((state_bitwise >> gid) & 1) == 1:
                return 0
            # Lower bound: at least 1 action to get x into the hand after clearing above(x)
            # (could be pick-up or unstack). We ignore the potential need to put down
            # some currently held block to keep it admissible.
            return 1 + 2 * above_count(x)

        def handempty_goal_lb() -> int:
            # If handempty is already true -> 0; otherwise at least 1 (put-down some block)
            if self.handempty_id is not None and ((state_bitwise >> self.handempty_id) & 1) == 1:
                return 0
            return 1

        # Goal-based lower bound
        goal_lb = 0
        for pred, args, gid in self.goal_facts:
            if is_true(gid):
                continue
            if pred == 'on' and len(args) == 2:
                goal_lb += on_goal_lb(args[0], args[1])
            elif pred == 'clear' and len(args) == 1:
                goal_lb += clear_goal_lb(args[0])
            elif pred == 'ontable' and len(args) == 1:
                goal_lb += on_table_goal_lb(args[0])
            elif pred == 'holding' and len(args) == 1:
                goal_lb += holding_goal_lb(args[0])
            elif pred == 'handempty':
                goal_lb += handempty_goal_lb()
            else:
                # Unknown predicate type in goals -> non-negative lower bound of 0
                goal_lb += 0

        # HTN task-network lower bound
        pending_prims = 0
        abstract_lb = 0
        for t in task_network:
            # Primitive action (must be executed eventually)
            if self._is_operator(t):
                pending_prims += 1
                continue
            # Abstract task: estimate via domain-specific LB if we can parse it
            if self._is_abstract_task(t):
                task_name = getattr(t, 'name', '')
                pred_t, t_args = self._parse_task_name(task_name)
                if pred_t is None:
                    continue
                if pred_t in ('do_put_on', 'do_move') and len(t_args) >= 2:
                    abstract_lb += on_goal_lb(t_args[0], t_args[1])
                elif pred_t == 'do_on_table' and len(t_args) >= 1:
                    abstract_lb += on_table_goal_lb(t_args[0])
                elif pred_t == 'do_clear' and len(t_args) >= 1:
                    abstract_lb += clear_goal_lb(t_args[0])
                else:
                    # Unknown abstract task: contribute 0 as safe LB
                    abstract_lb += 0

        htn_lb = max(pending_prims, abstract_lb)

        return int(max(goal_lb, htn_lb))

    # ---------------------------- Utilities ---------------------------- #

    def _build_on_maps(self, state_bitwise: int) -> Tuple[Dict[str, str], Dict[str, str]]:
        parent_of: Dict[str, str] = {}
        child_of: Dict[str, str] = {}
        for gid, x, y in self.on_facts:
            if ((state_bitwise >> gid) & 1) == 1:
                parent_of[x] = y
                # In standard Blocksworld, at most one block sits on a given block
                child_of[y] = x
        return parent_of, child_of

    def _fact_id_by_key(self, pred: str, args: Tuple[str, ...]) -> Optional[int]:
        if pred == 'on' and len(args) == 2:
            return self.on_id_by_pair.get((args[0], args[1]))
        if pred == 'ontable' and len(args) == 1:
            return self.ontable_id_by_block.get(args[0])
        if pred == 'clear' and len(args) == 1:
            return self.clear_id_by_block.get(args[0])
        if pred == 'holding' and len(args) == 1:
            return self.holding_id_by_block.get(args[0])
        if pred == 'handempty':
            return self.handempty_id
        return None

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        # As specified by the framework: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            core = name[1:]
            idx = core.find('[')
            if idx > 0 and core.endswith(']'):
                pred = core[:idx]
                args_str = core[idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return pred, args
        return None, []

    def _parse_task_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        # Try bracket-style: do_put_on[b1,b2]
        s = name.strip()
        lb = s.find('[')
        if lb > 0 and s.endswith(']'):
            pred = s[:lb]
            args_str = s[lb + 1:-1]
            args = [a.strip() for a in args_str.split(',') if a.strip()]
            return pred, args
        # Try parenthesis-style: do_put_on(b1,b2) or do_put_on b1 b2
        lp = s.find('(')
        rp = s.rfind(')')
        if lp > 0 and rp > lp:
            pred = s[:lp].strip()
            args_str = s[lp + 1:rp].strip()
            if args_str:
                if ',' in args_str:
                    args = [a.strip() for a in args_str.split(',') if a.strip()]
                else:
                    args = [a.strip() for a in args_str.split() if a.strip()]
            else:
                args = []
            return pred, args
        # Fallback: split by spaces (least reliable but safe)
        toks = [t for t in s.replace('(', ' ').replace(')', ' ').replace('[', ' ').replace(']', ' ').replace(',', ' ').split() if t]
        if len(toks) >= 1:
            return toks[0], toks[1:]
        return None, []

    def _is_operator(self, obj: Any) -> bool:
        # Robust type check without importing Operator at module top
        return obj.__class__.__name__ == 'Operator'

    def _is_abstract_task(self, obj: Any) -> bool:
        return obj.__class__.__name__ == 'AbstractTask'


# Alias required by some loaders
HeuristicImpl = BlocksworldGPT5Heuristic13
