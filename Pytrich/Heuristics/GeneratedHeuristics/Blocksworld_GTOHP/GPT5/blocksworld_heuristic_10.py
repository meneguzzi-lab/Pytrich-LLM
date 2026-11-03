from typing import Dict, List, Tuple, Union, Optional, Set

# Fallback import to be resilient across Pytrich variants
try:
    from Pytrich.Heuristics.heuristic import Heuristic  # type: ignore
except Exception:
    class Heuristic:  # minimal fallback to avoid import errors in standalone tests
        def __init__(self, model, *args, **kwargs):
            self.model = model


# Type hints for Pytrich components (duck-typed)
try:
    from Model import Model  # type: ignore
except Exception:
    Model = object  # pragma: no cover


class BlocksworldGPT5Heuristic10(Heuristic):
    """
    Domain-dependent heuristic for BLOCKSWORLD (HDDL) tailored to Pytrich.

    Key ideas (admissible, efficient, hierarchy-aware):
    - Goals in this domain are On(x,y) relations. Each unsatisfied On(x,y) requires at least one Stack action.
      Hence, a tight admissible lower bound is the count of unsatisfied goal On-facts.
    - The HTN task network mainly contains abstract tasks: do_put_on(x,y), do_on_table(x), do_move(x,y), do_clear(x).
      Under provided methods:
        * do_put_on(x,y) ultimately requires one move of x onto y => at least 2 primitive actions (grasp+stack),
          but can be solved at zero cost if on(x,y) already holds (method m0). We therefore count 2 only when on(x,y)
          is not already true.
        * do_move(x,y) reduces to exactly 2 actions (pick-up/unstack + stack) regardless of table status => +2.
        * do_clear(x) and do_on_table(x) can be zero-cost due to methods with NOP (m6 for clear if already clear; m3 for
          on_table if clear), and their detailed costs depend on current stacks. For scalability and admissibility, we
          do not add extra for these here, avoiding possible double-counting across tasks.
      This yields a consistent cost model across decompositions: when do_put_on decomposes into do_move and clears,
      parent's 2 maps to child's 2 (do_move), preventing heuristic spikes across zero-cost decompositions.
    - Primitive operators currently present in the task network must be executed. Each contributes at least its cost.

    Final heuristic value is the maximum of:
       1) Number of unsatisfied goal On-facts (>= number of required Stack ops)
       2) Sum of minimal costs implied by the current task network:
          - +operator.cost for each primitive operator present in TN
          - +2 for each do_move(x,y)
          - +2 for each do_put_on(x,y) whose on(x,y) is NOT true now
          - +0 for do_clear / do_on_table (admissible lower bound; avoids double counting and heavy state analysis)

    Implementation notes for scalability:
    - We never iterate over all facts during search. We pre-index only the on-facts needed for goals and initial
      do_put_on/do_move tasks, using a single pass over all facts at initialization.
    - All state membership checks are done via O(1) bitwise operations using fact.global_id.
    - Parsing of task names is cached.
    """

    def __init__(self, model: Model, blocksworld_gpt5_heuristic_10: bool = True):
        super().__init__(model)
        self.enabled = bool(blocksworld_gpt5_heuristic_10)

        # Caches and indices
        self._task_parse_cache: Dict[str, Tuple[str, Tuple[str, ...]]] = {}
        self._fact_by_name: Dict[str, object] = {}  # name -> Fact
        self._on_fact_by_pair: Dict[Tuple[str, str], object] = {}  # (x, y) -> Fact for +on[x,y]
        self._goal_on_facts: List[object] = []  # List of Fact objects for goal +on facts

        # Prepare minimal indices we need (only once)
        self._initialize_indices()

    # ---- Public heuristic API ----
    def h(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        if not self.enabled:
            return 0

        # 1) Unsatisfied On-goal lower bound (each requires at least one Stack).
        goal_lb = self._unsatisfied_goal_on_count(state_bitwise)

        # 2) Task-network based lower bound
        tn_lb = self._task_network_lower_bound(state_bitwise, task_network)

        # Admissible bound: take the max of independent lower bounds
        return max(goal_lb, tn_lb)

    # Allow call-like usage
    def __call__(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        return self.h(state_bitwise, task_network)

    # ---- Initialization helpers ----
    def _initialize_indices(self):
        # Collect all needed on-fact names from goals and initial TN do_put_on/do_move
        needed_on_names: Set[str] = set()

        # 1) Goal on-facts
        try:
            goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_fact_names = []
        goal_on_names: List[str] = []
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'on' and len(args) == 2:
                goal_on_names.append(gname)
                needed_on_names.add(gname)

        # 2) Initial TN do_put_on/do_move pairs
        initial_pairs = self._collect_initial_tn_pairs(self.model.initial_tn)
        for (x, y) in initial_pairs:
            needed_on_names.add(self._on_name(x, y))

        # 3) Build fact name -> Fact index for required on-facts in one pass
        # Also keep a direct (x,y) -> Fact mapping for quick state checks
        if hasattr(self.model, 'facts') and self.model.facts:
            for f in self.model.facts:
                try:
                    fname = f.name
                except Exception:
                    continue
                if fname in needed_on_names:
                    self._fact_by_name[fname] = f
                    pred, args = self._parse_fact_name(fname)
                    if pred == 'on' and len(args) == 2:
                        self._on_fact_by_pair[(args[0], args[1])] = f

        # 4) Resolve goal_on_facts list
        for gname in goal_on_names:
            f = self._fact_by_name.get(gname, None)
            if f is not None:
                self._goal_on_facts.append(f)

    def _collect_initial_tn_pairs(self, tn: List[Union[int, object]]) -> Set[Tuple[str, str]]:
        pairs: Set[Tuple[str, str]] = set()
        if not isinstance(tn, list):
            return pairs
        for item in tn:
            comp = self._get_component(item)
            if comp is None:
                continue
            # AbstractTask recognition: has attribute 'task_network'
            if hasattr(comp, 'task_network') and hasattr(comp, 'name'):
                pred, args = self._parse_task_name(str(comp.name))
                if pred in ('do_put_on', 'do_move') and len(args) >= 2:
                    x, y = args[0], args[1]
                    pairs.add((x, y))
        return pairs

    # ---- Heuristic components ----
    def _unsatisfied_goal_on_count(self, state_bitwise: int) -> int:
        # Each unsatisfied goal on(x,y) entails at least one Stack
        cnt = 0
        for f in self._goal_on_facts:
            if not self._is_true(state_bitwise, f):
                cnt += 1
        return cnt

    def _task_network_lower_bound(self, state_bitwise: int, task_network: List[Union[int, object]]) -> int:
        if not isinstance(task_network, list) or not task_network:
            return 0

        cost_lb = 0
        for item in task_network:
            comp = self._get_component(item)
            if comp is None:
                continue

            # Primitive operator: must be executed eventually -> contributes at least its own cost
            if self._is_operator(comp):
                op_cost = getattr(comp, 'cost', 1)
                # Guarantee integer, default to 1 if malformed
                if not isinstance(op_cost, int):
                    op_cost = 1
                cost_lb += op_cost
                continue

            # Abstract task: guide by domain knowledge
            if hasattr(comp, 'name'):
                pred, args = self._parse_task_name(str(comp.name))
                if pred == 'do_move' and len(args) >= 2:
                    # Minimal 2 actions: pick-up/unstack + stack
                    cost_lb += 2
                elif pred == 'do_put_on' and len(args) >= 2:
                    x, y = args[0], args[1]
                    # If already on(x,y), method m0 (nop) applies => 0; else at least 2 (grasp+stack)
                    on_fact = self._on_fact_by_pair.get((x, y), None)
                    if on_fact is None:
                        # If we lack its Fact index (rare), assume not satisfied to keep guidance (still admissible)
                        cost_lb += 2
                    else:
                        if not self._is_true(state_bitwise, on_fact):
                            cost_lb += 2
                else:
                    # do_clear / do_on_table and others: 0 in our LB (admissible, avoids double counting)
                    pass

        return cost_lb

    # ---- Utilities ----
    @staticmethod
    def _on_name(x: str, y: str) -> str:
        return f"+on[{x},{y}]"

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], Tuple[str, ...]]:
        # Format expected: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if not name:
            return None, tuple()
        s = name.strip()
        if s.startswith('+') or s.startswith('-'):
            s = s[1:]
        # Bracketed form
        lb = s.find('[')
        rb = s.rfind(']')
        if lb > 0 and rb == len(s) - 1:
            pred = s[:lb]
            args_str = s[lb + 1 : rb]
            args = tuple(a.strip() for a in args_str.split(',')) if args_str else tuple()
            return pred, args
        # Fallback: no-arg predicate like "+handempty" (without brackets)
        if '[' not in s and ']' not in s:
            return s, tuple()
        return None, tuple()

    def _parse_task_name(self, name: str) -> Tuple[str, Tuple[str, ...]]:
        # Cached parsing (task names appear frequently)
        key = name
        cached = self._task_parse_cache.get(key)
        if cached is not None:
            return cached

        s = str(name).strip()
        pred: str = s
        args: Tuple[str, ...] = tuple()

        # Common compiled form: name[arg1,arg2]
        lb = s.find('[')
        rb = s.rfind(']')
        if lb > 0 and rb == len(s) - 1:
            pred = s[:lb]
            args_str = s[lb + 1 : rb]
            args = tuple(a.strip() for a in args_str.split(',')) if args_str else tuple()
            self._task_parse_cache[key] = (pred, args)
            return pred, args

        # Alternative: name(arg1,arg2)
        lp = s.find('(')
        rp = s.rfind(')')
        if lp > 0 and rp == len(s) - 1:
            pred = s[:lp]
            args_str = s[lp + 1 : rp]
            # Allow space or comma separated
            if ',' in args_str:
                args = tuple(a.strip() for a in args_str.split(','))
            else:
                args = tuple(a.strip() for a in args_str.split() if a.strip())
            self._task_parse_cache[key] = (pred, args)
            return pred, args

        # Fallback: no args
        self._task_parse_cache[key] = (s, tuple())
        return s, tuple()

    def _get_component(self, comp_or_id: Union[int, object]) -> Optional[object]:
        # Convert IDs to model components when necessary
        if isinstance(comp_or_id, int):
            try:
                return self.model.get_component(comp_or_id)
            except Exception:
                return None
        return comp_or_id

    @staticmethod
    def _is_operator(comp: object) -> bool:
        # Operators have attributes pos_precons/add_effects, but AbstractTasks have 'task_network'
        return hasattr(comp, 'apply') or (hasattr(comp, 'pos_precons') and hasattr(comp, 'add_effects') and not hasattr(comp, 'task_network'))

    @staticmethod
    def _is_true(state_bitwise: int, fact_obj: object) -> bool:
        try:
            gid = fact_obj.global_id
            return ((state_bitwise >> gid) & 1) == 1
        except Exception:
            return False


# Factory alias (if the framework expects a function or variable to register the heuristic)
def get_heuristic(model: Model, blocksworld_gpt5_heuristic_10: bool = True) -> BlocksworldGPT5Heuristic10:
    return BlocksworldGPT5Heuristic10(model, blocksworld_gpt5_heuristic_10=blocksworld_gpt5_heuristic_10)
