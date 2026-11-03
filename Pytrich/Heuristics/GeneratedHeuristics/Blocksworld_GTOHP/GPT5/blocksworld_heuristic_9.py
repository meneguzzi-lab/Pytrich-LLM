from typing import Dict, List, Set, Tuple, Optional

# Fallback-safe import for Heuristic base class
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    class Heuristic:
        def __init__(self, model=None, *args, **kwargs):
            self.model = model
        def __call__(self, state_bitwise:int, task_network:List=None) -> int:
            return 0

# Optional type hints (robust to missing imports at runtime)
try:
    from Model import Model  # type: ignore
    from Operator import Operator  # type: ignore
    from AbstractTask import AbstractTask  # type: ignore
    from Fact import Fact  # type: ignore
except Exception:
    Model = object  # type: ignore
    Operator = object  # type: ignore
    AbstractTask = object  # type: ignore
    Fact = object  # type: ignore


class BlocksworldGPT5Heuristic9(Heuristic):
    """
    Domain-dependent heuristic for BLOCKS (Blocksworld) in Pytrich.

    Design summary
    1) Admissible relaxed plan lower bound (h_max over primitive operators):
       - We compute h_max (delete-relaxation, max-aggregation) over facts using only primitive operators.
       - For any target set of facts R, h_max(R) = max_{f in R} h_max(f), where
         h_max(f) = 0 if f true in state; else min_{op adds f} (1 + max_{p in pre(op)} h_max(p)).
       - This is admissible and leverages operator structure.

    2) Hierarchical guidance via task network:
       - For abstract tasks, we map them to necessary facts:
         * do_put_on(x,y) -> +on[x,y]
         * do_move(x,y)   -> +on[x,y]
         * do_clear(x)    -> +clear[x]
         * do_on_table(x) -> +clear[x]    (since all methods require clear(x))
       - For primitive tasks present in the task network, we add a separate lower bound equal to
         the count of remaining non-nop primitive operators (each must be executed).

    3) Final heuristic value:
       - Let F_required = union of goal facts and facts required by current task network.
       - lb_facts = h_max(F_required) computed over primitive operators.
       - lb_prims = number of non-nop primitive operators pending in task network.
       - Return max(lb_facts, lb_prims). This is admissible (both are lower bounds).

    Efficiency notes
    - We precompute mappings: fact_name -> id, fact_id -> adders(operators), operator preconditions.
    - We use fast bitwise checks to test if a fact holds in the state.
    - We parse names robustly for both facts and tasks. Grounded facts use the "+pred[args]" format,
      while tasks/operators typically use "name[args]".

    This heuristic scales to large instances by keeping per-call work linear in the number of referenced facts,
    with memoization for h_max over the current state.
    """

    def __init__(self, model: Model, blocksworld_gpt5_heuristic_9: Optional[dict] = None):
        super().__init__(model)
        self.blocksworld_gpt5_heuristic_9 = blocksworld_gpt5_heuristic_9 or {}

        # Precompute fact name -> id and id -> fact name
        self.fact_name_to_id: Dict[str, int] = {}
        self.fact_id_to_name: Dict[int, str] = {}
        for f in self.model.facts:
            # f.name is like "+on[b1,b2]" or "-on[b1,b2]" (for negative pseudo-facts)
            self.fact_name_to_id[f.name] = f.global_id
            self.fact_id_to_name[f.global_id] = f.name

        # Precompute operator structures for fast relaxed planning
        self.operators = list(self.model.operators)
        self.op_precons: Dict[int, List[int]] = {}  # operator.global_id -> list of precondition fact ids
        self.fact_to_adders: Dict[int, List] = {}   # fact_id -> list of operators that add it

        for op in self.operators:
            # collect positive preconditions as fact IDs
            pre_list = list(op.get_precons()) if hasattr(op, 'get_precons') else []
            self.op_precons[op.global_id] = pre_list

            # map add effects to adder operators
            if hasattr(op, 'get_add_effects'):
                for add_fid in op.get_add_effects():
                    self.fact_to_adders.setdefault(add_fid, []).append(op)

        # Cache goal fact ids (positive literals) once
        self.goal_fact_ids: List[int] = []
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        for gname in goal_fact_names:
            # Expect +predicate[args]
            fid = self.fact_name_to_id.get(gname)
            if fid is not None:
                self.goal_fact_ids.append(fid)

        # Constants
        self._INF = 10 ** 6

    # -------------------------- Parsing utilities -------------------------- #
    @staticmethod
    def _parse_name_generic(name: str) -> Tuple[str, List[str]]:
        """Parse names of the form:
        - "+pred[arg1,arg2]" or "-pred[arg1,arg2]"  (facts)
        - "pred[arg1,arg2]"                          (operators/abstract tasks)
        - fallback: split by spaces: "pred a b"
        Returns (predicate, [args...]) lowercased predicate, raw args as seen.
        """
        if not name:
            return "", []
        s = name.strip()
        # strip leading +/- if present
        if s and (s[0] == '+' or s[0] == '-'):
            s = s[1:]
        # bracketed form
        lb = s.find('[')
        rb = s.rfind(']')
        if lb != -1 and rb != -1 and rb > lb:
            pred = s[:lb].strip()
            args_str = s[lb + 1:rb].strip()
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return pred.lower(), args
        # parenthesized form (rare)
        lb = s.find('(')
        rb = s.rfind(')')
        if lb != -1 and rb != -1 and rb > lb:
            pred = s[:lb].strip()
            args_str = s[lb + 1:rb].strip()
            args = [a.strip() for a in args_str.split()] if args_str else []
            return pred.lower(), args
        # space-separated fallback
        parts = s.split()
        if parts:
            return parts[0].lower(), parts[1:]
        return "", []

    def _pos_fact_name(self, pred: str, args: List[str]) -> str:
        return f"+{pred}[{','.join(args)}]"

    # ------------------------ Type-checking helpers ------------------------ #
    @staticmethod
    def _is_operator(obj) -> bool:
        # Duck typing: Operator has 'apply' and bitsets for pre/eff
        return hasattr(obj, 'apply') and hasattr(obj, 'pos_precons') and hasattr(obj, 'add_effects')

    @staticmethod
    def _is_abstract_task(obj) -> bool:
        return hasattr(obj, 'decompositions') and hasattr(obj, 'name') and not BlocksworldGPT5Heuristic9._is_operator(obj)

    # -------------------------- h_max over facts --------------------------- #
    def _fact_cost_hmax(self, state_bitwise: int, fact_id: int, memo: Dict[int, int], visiting: Set[int]) -> int:
        # Already computed
        if fact_id in memo:
            return memo[fact_id]

        # If already true in current state, cost 0
        if (state_bitwise >> fact_id) & 1:
            memo[fact_id] = 0
            return 0

        # Break potential cycles robustly
        if fact_id in visiting:
            return self._INF
        visiting.add(fact_id)

        # Consider all operators that can add this fact
        adders = self.fact_to_adders.get(fact_id, [])
        if not adders:
            visiting.discard(fact_id)
            memo[fact_id] = self._INF
            return memo[fact_id]

        best = self._INF
        for op in adders:
            pre_ids = self.op_precons.get(op.global_id, [])
            # h_max: max over preconditions
            max_prec = 0
            for pid in pre_ids:
                c = self._fact_cost_hmax(state_bitwise, pid, memo, visiting)
                if c >= self._INF:
                    max_prec = self._INF
                    break
                if c > max_prec:
                    max_prec = c
            if max_prec < self._INF:
                cand = 1 + max_prec
                if cand < best:
                    best = cand

        visiting.discard(fact_id)
        memo[fact_id] = best
        return best

    def _hmax_over_required_facts(self, state_bitwise: int, required_fact_ids: Set[int]) -> int:
        if not required_fact_ids:
            return 0
        memo: Dict[int, int] = {}
        visiting: Set[int] = set()
        best = 0
        for fid in required_fact_ids:
            c = self._fact_cost_hmax(state_bitwise, fid, memo, visiting)
            if c > best:
                best = c
            if best >= self._INF:
                # early stop if clearly very large
                return best
        return best

    # -------------------------- Task requirement --------------------------- #
    def _required_facts_from_task_network(self, task_network: List) -> Tuple[Set[int], int]:
        """Extract a set of required positive fact IDs implied by the current task network and
        count the number of remaining non-nop primitive operators (each must be executed).
        - Abstract tasks mapped to key necessary facts (see class docstring)
        - Primitive operators: add their preconditions to required facts; also count 1 per non-nop op.
        """
        req_fact_ids: Set[int] = set()
        non_nop_primitive_count = 0

        if not task_network:
            return req_fact_ids, non_nop_primitive_count

        for t in task_network:
            try:
                tname: str = getattr(t, 'name', '')
            except Exception:
                tname = ''

            pred, args = self._parse_name_generic(tname)

            if self._is_operator(t):
                # Count non-nop primitive operators
                if pred != 'nop':
                    non_nop_primitive_count += 1
                # Add operator preconditions to required facts
                try:
                    pre_ids = list(t.get_precons()) if hasattr(t, 'get_precons') else []
                except Exception:
                    pre_ids = []
                for pid in pre_ids:
                    # Only consider positive requirements in relaxed planning; bitset already encodes them
                    # Here, pid can be either positive or negative literal encoding.
                    # We conservatively include whatever is in pos_precons bitset; the h_max function handles truth checking.
                    req_fact_ids.add(pid)

            elif self._is_abstract_task(t):
                # Map to required facts
                if pred == 'do_put_on' or pred == 'do_move':
                    if len(args) >= 2:
                        fname = self._pos_fact_name('on', args[:2])
                        fid = self.fact_name_to_id.get(fname)
                        if fid is not None:
                            req_fact_ids.add(fid)
                elif pred == 'do_clear':
                    if len(args) >= 1:
                        fname = self._pos_fact_name('clear', args[:1])
                        fid = self.fact_name_to_id.get(fname)
                        if fid is not None:
                            req_fact_ids.add(fid)
                elif pred == 'do_on_table':
                    if len(args) >= 1:
                        # Necessary condition for all methods of do_on_table is clear(x)
                        fname = self._pos_fact_name('clear', args[:1])
                        fid = self.fact_name_to_id.get(fname)
                        if fid is not None:
                            req_fact_ids.add(fid)
                # Other abstract tasks (if any) default to no additional requirement here.

            else:
                # Unknown component type; attempt best-effort parse for known pattern names
                if pred in ('do_put_on', 'do_move') and len(args) >= 2:
                    fname = self._pos_fact_name('on', args[:2])
                    fid = self.fact_name_to_id.get(fname)
                    if fid is not None:
                        req_fact_ids.add(fid)
                elif pred in ('do_clear', 'do_on_table') and len(args) >= 1:
                    fname = self._pos_fact_name('clear', args[:1])
                    fid = self.fact_name_to_id.get(fname)
                    if fid is not None:
                        req_fact_ids.add(fid)

        return req_fact_ids, non_nop_primitive_count

    # ------------------------------ Heuristic ------------------------------ #
    def __call__(self, state_bitwise: int, task_network: Optional[List] = None) -> int:
        # Aggregate required facts: goals + task-network-implied
        required_fact_ids: Set[int] = set(self.goal_fact_ids)
        prim_pending = 0
        if task_network:
            req_from_tn, prim_cnt = self._required_facts_from_task_network(task_network)
            required_fact_ids |= req_from_tn
            prim_pending = prim_cnt

        # Admissible lower bounds
        lb_facts = self._hmax_over_required_facts(state_bitwise, required_fact_ids)
        lb_prims = prim_pending  # must execute these remaining primitive operators

        h_val = lb_facts if lb_facts >= lb_prims else lb_prims
        # Keep heuristic within reasonable integer bounds
        if h_val >= self._INF:
            # Very large -> cap to INF (still admissible, just coarse)
            return self._INF
        return int(h_val)


# Alias for external configuration systems (optional)
HeuristicClass = BlocksworldGPT5Heuristic9
