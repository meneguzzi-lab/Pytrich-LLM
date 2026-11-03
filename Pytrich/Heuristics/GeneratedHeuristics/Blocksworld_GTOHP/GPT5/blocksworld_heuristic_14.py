from typing import Dict, Tuple, List, Optional, Set, Union

# Pytrich is expected to provide these base classes
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Minimal fallback for static analysis; the real environment provides Heuristic
    class Heuristic:
        def __init__(self, *args, **kwargs):
            pass


class BlocksworldGPT5Heuristic14(Heuristic):
    """
    BlocksworldGPT5Heuristic14

    Domain-dependent, admissible, and efficient heuristic for the BLOCKS (Blocksworld) HDDL domain.

    Design rationale (summary):
    - Goals in the provided instances consist of positive on(x,y) facts. Achieving an on(x,y) requires
      at least one stack action if it is not already true. This yields a safe, admissible base count:
        base = |{ (x,y) in goals: on(x,y) not true in state }|.
    - Additionally, for each unsatisfied on(x,y), both x and y must be clear at the time of stacking.
      If currently x or y is not clear, at least one unstack action will be required to clear that block.
      We conservatively add 1 unstack per distinct currently-not-clear block among all x and y of
      unsatisfied goals (union-set to avoid double counting). This is still an admissible lower bound
      because an unstack clears only one particular blocked block.
    - HTN task network: We estimate a separate lower bound on the residual decomposition/primitive
      actions implied by the current task network. For each abstract task instance t:
        * do_put_on(x,y) or do_move(x,y): if on(x,y) is false -> at least one stack is needed (LB += 1).
        * do_on_table(x): if ontable(x) is false -> at least one put-down is needed (LB += 1).
        * do_clear(x): if clear(x) is false -> at least one unstack is needed (LB += 1).
      Because the same action can simultaneously progress both goal satisfaction and task completion,
      we combine the two lower bounds safely with max(goals_LB, tasks_LB) to preserve admissibility.

    Efficiency considerations:
    - All state membership tests use bitwise checks via Fact.global_id (O(1)).
    - Goals are accessed via the model's bitset (self.model.goals) and processed once at init to cache
      their arguments;
    - Massive instances (e.g., p30 with 1000 blocks) are handled by avoiding expensive string scans at
      runtime and using precomputed IDs for on/clear/ontable facts.

    Notes:
    - This heuristic is consistent under usual action cost models (unit costs) and relies only on
      necessary conditions for stacking and clearing, never overcounting shared work between goals and
      tasks (we use max, not sum, when combining the two bounds).
    - Robust parsing enables working across different internal name formats for facts/tasks.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_14: bool = True):
        super().__init__()
        self.model = model
        self.name = "BlocksworldGPT5Heuristic14"

        # Fast popcount support
        self._has_bitcount = hasattr(int, 'bit_count')

        # Build Fact name -> Fact mapping and specialized indices for on/clear/ontable
        self._fact_by_name: Dict[str, object] = {}
        self._clear_id: Dict[str, int] = {}
        self._ontable_id: Dict[str, int] = {}
        self._on_id: Dict[Tuple[str, str], int] = {}

        # Prepare caches by scanning fact base once
        for f in getattr(self.model, 'facts', []):
            self._fact_by_name[f.name] = f
            pred, args = self._parse_fact_name(f.name)
            if pred == 'clear' and len(args) == 1:
                self._clear_id[args[0]] = f.global_id
            elif pred == 'ontable' and len(args) == 1:
                self._ontable_id[args[0]] = f.global_id
            elif pred == 'on' and len(args) == 2:
                self._on_id[(args[0], args[1])] = f.global_id

        # Cache parsed goal on(x,y) entries: list of (fact_id, x, y)
        self._goal_on_entries: List[Tuple[int, str, str]] = []
        try:
            goal_names = self.model.state_explicit_repr(self.model.goals)
        except Exception:
            goal_names = []
        for g in goal_names:
            pred, args = self._parse_fact_name(g)
            if pred == 'on' and len(args) == 2:
                fid = self._on_id.get((args[0], args[1]))
                if fid is not None:
                    self._goal_on_entries.append((fid, args[0], args[1]))

    # Public heuristic API: support both evaluate and __call__
    def __call__(self, state_bitwise: int, task_network: List[Union[object, object]] = None) -> int:
        return self.evaluate(state_bitwise, task_network or [])

    def evaluate(self, state_bitwise: int, task_network: List[Union[object, object]] = None) -> int:
        if task_network is None:
            task_network = []
        goals_lb = self._goals_lower_bound(state_bitwise)
        tasks_lb = self._tasks_lower_bound(state_bitwise, task_network)
        # Combine safely to preserve admissibility
        return max(goals_lb, tasks_lb)

    # --- Core lower-bound components ---
    def _goals_lower_bound(self, state_bitwise: int) -> int:
        # Unsatisfied goal facts directly from bitsets (fast popcount)
        unsat_bits = self.model.goals & ~state_bitwise
        base_unsat = self._bitcount(unsat_bits)
        if base_unsat == 0:
            return 0

        # For each unsatisfied on(x,y), count distinct blocked x and y currently not clear
        blocked: Set[str] = set()
        for fid, x, y in self._goal_on_entries:
            if not self._is_true_by_id(state_bitwise, fid):
                # x not clear?
                cx = self._clear_id.get(x)
                if cx is not None and not self._is_true_by_id(state_bitwise, cx):
                    blocked.add(x)
                # y not clear?
                cy = self._clear_id.get(y)
                if cy is not None and not self._is_true_by_id(state_bitwise, cy):
                    blocked.add(y)
        return base_unsat + len(blocked)

    def _tasks_lower_bound(self, state_bitwise: int, task_network: List[Union[object, object]]) -> int:
        lb = 0
        for t in task_network:
            # Only abstract tasks matter here; operators either will be applied next or have exact costs
            # The AbstractTask class exists in the planner; we test by attribute presence to avoid imports.
            if not hasattr(t, 'name') or not hasattr(t, 'decompositions'):
                # Likely an Operator; skip for LB purposes.
                continue

            tname: str = getattr(t, 'name', '')
            pred, args = self._parse_task_name(tname)
            pred_l = (pred or '').lower()

            if pred_l in ('do_put_on', 'do_move') and len(args) >= 2:
                x, y = args[0], args[1]
                on_id = self._on_id.get((x, y))
                if on_id is not None and not self._is_true_by_id(state_bitwise, on_id):
                    # At least one stack needed
                    lb += 1
            elif pred_l == 'do_on_table' and len(args) >= 1:
                x = args[0]
                ontable_id = self._ontable_id.get(x)
                if ontable_id is not None and not self._is_true_by_id(state_bitwise, ontable_id):
                    # At least one put-down needed in the best case
                    lb += 1
            elif pred_l == 'do_clear' and len(args) >= 1:
                x = args[0]
                clear_id = self._clear_id.get(x)
                if clear_id is not None and not self._is_true_by_id(state_bitwise, clear_id):
                    # At least one unstack needed in the best case
                    lb += 1
            else:
                # Unknown/other abstract tasks -> we cannot safely assume positive cost; keep admissible (add 0)
                pass
        return lb

    # --- Helpers ---
    def _is_true_by_id(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _bitcount(self, x: int) -> int:
        if self._has_bitcount:
            return x.bit_count()
        return bin(x).count('1')

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse a grounded fact name according to the problem statement format:
        +predicate[arg1,arg2] or -predicate[arg1,arg2]
        Returns (predicate, [args]) or (None, []).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if not s:
            return None, []
        if s[0] in ['+', '-']:
            s = s[1:]
        bracket_idx = s.find('[')
        if bracket_idx > 0 and s.endswith(']'):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1:-1]
            args = [a.strip() for a in args_str.split(',')] if args_str else []
            return predicate, args
        return None, []

    def _parse_task_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        """
        Robust parsing for task names. Supports variants like:
        - do_put_on[b1,b2]
        - do_put_on(b1,b2)
        - do_put_on b1 b2
        Returns (predicate, [args]) or (None, []).
        """
        if not isinstance(name, str):
            return None, []
        s = name.strip()
        if not s:
            return None, []

        # Bracket format
        bidx = s.find('[')
        if bidx > 0 and s.endswith(']'):
            pred = s[:bidx]
            inside = s[bidx + 1:-1]
            args = [a.strip() for a in inside.split(',')] if inside else []
            return pred, args

        # Parenthesis format
        pidx = s.find('(')
        if pidx > 0 and s.endswith(')'):
            pred = s[:pidx]
            inside = s[pidx + 1:-1]
            # allow comma or space separated
            if ',' in inside:
                args = [a.strip() for a in inside.split(',') if a.strip()]
            else:
                args = [a.strip() for a in inside.split() if a.strip()]
            return pred, args

        # Space-separated fallback
        parts = s.split()
        if len(parts) >= 1:
            pred = parts[0]
            args = parts[1:]
            return pred, args

        return None, []


# Factory function hook (optional): some frameworks expect a callable to provide the heuristic
def blocksworld_gpt5_heuristic_14(model):
    return BlocksworldGPT5Heuristic14(model)
