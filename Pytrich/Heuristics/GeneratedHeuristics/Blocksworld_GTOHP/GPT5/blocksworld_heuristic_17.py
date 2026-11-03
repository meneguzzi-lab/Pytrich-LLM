from typing import Dict, Tuple, List, Optional

# Robust import for the Pytrich Heuristic base class
try:
    from .Heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.heuristic import Heuristic
    except Exception:
        # Fallback minimal stub (keeps module importable if framework path is different).
        class Heuristic(object):
            def __init__(self, model, *args, **kwargs):
                self.model = model
            def h(self, state_bitwise: int, task_network: List[object]) -> int:
                return 0


class BlocksworldGPT5Heuristic17(Heuristic):
    """
    Domain-dependent heuristic for BLOCKSWORLD HDDL in the Pytrich Planner.

    Core ideas
    - Cheap, admissible lower bound that scales to very large instances
      while exploiting Blocksworld-HTN structure.
    - Uses two complementary components and takes their max (to avoid double counting):
        1) h_goals: number of goal facts not yet satisfied (delete-relaxed lower bound).
           Computed via fast bitwise popcount: popcount(goals & ~state).
        2) h_tn: lower bound on the remaining task-network refinement/execution cost,
           derived from domain-specific method structure:
             * Primitive operators in TN contribute at least their cost (usually 1).
             * Abstract tasks:
                 - do_put_on(x,y):
                     If on(x,y) holds now: cheapest method m0_do_put_on -> a single 'nop' => LB = 1
                     Else: must use m1_do_put_on which (even in best case) decomposes into:
                           do_clear(x)=1, do_clear(y)=1, do_on_table(y)=1, do_move(x,y)=2
                           => LB = 5
                 - do_move(x,y): methods m4/m5 each require at least 2 primitives (pick-up+stack or unstack+stack)
                   => LB = 2 (state-independent)
                 - do_clear(x): m6 gives 'nop', m7 gives unstack/put-down, so LB >= 1
                 - do_on_table(x): m3 gives 'nop', m2 gives unstack+put-down, so LB >= 1
           These LBs are admissible because they are the minimal number of primitives required
           by any applicable decomposition from the current state (for do_put_on we check on(x,y)
           to decide whether the 1-step 'nop' method is applicable right now; otherwise we fall
           back to the 5-step lower bound derived from m1_do_put_on's structure).

    Why max(h_goals, h_tn)?
    - Summing could double-count actions (one action can both achieve a goal fact and discharge a task),
      causing inadmissibility. Taking max preserves admissibility while maintaining guidance.

    Design notes
    - Very fast: uses only bitwise checks for state membership and O(|TN|) simple parsing.
    - Scales to the largest instance due to minimal overhead.
    - Robust name parsing supports both 'name[arg1,arg2]' and 'name(arg1,arg2)' formats.
    - No reliance on slow state_explicit_repr during search (only bitwise ops at runtime).
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_17: bool = True):
        super().__init__(model)
        self.blocksworld_gpt5_heuristic_17 = blocksworld_gpt5_heuristic_17

        # Precompute fact index for fast bitwise checks
        # Mapping: predicate -> Dict[Tuple[str, ...], int(global_id)]
        self._fact_gid: Dict[str, Dict[Tuple[str, ...], int]] = {}
        self._build_fact_index()

        # Cache used during TN parsing
        self._taskname_cache: Dict[str, Tuple[str, Tuple[str, ...]]] = {}

        # Prebind predicate names used often
        self._P_ON = "on"
        self._P_ONTABLE = "ontable"
        self._P_CLEAR = "clear"

    # --------------------------
    # Utility: facts and parsing
    # --------------------------
    def _build_fact_index(self):
        self._fact_gid.clear()
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            d = self._fact_gid.setdefault(pred, {})
            d[tuple(args)] = f.global_id

    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        """
        Parse fact names of the form '+predicate[a,b]' or '-predicate[a,b]'.
        Returns (predicate, [args...]) or (None, []).
        """
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _parse_action_or_task_name(name: str) -> Tuple[str, Tuple[str, ...]]:
        """
        Robustly parse grounded operator/abstract-task names.
        Supports formats like 'do_put_on[b1,b2]' and 'do_put_on(b1,b2)'.
        Returns (symbol, (args,...)). When no args, returns empty tuple.
        """
        s = name.strip()
        # Try bracket form: name[arg1,arg2]
        lb = s.find("[")
        rb = s.rfind("]")
        if lb > 0 and rb == len(s) - 1:
            sym = s[:lb]
            args_str = s[lb + 1 : rb]
            args = tuple(a.strip() for a in args_str.split(",")) if args_str else tuple()
            return sym, args
        # Try parenthesis form: name(arg1,arg2)
        lp = s.find("(")
        rp = s.rfind(")")
        if lp > 0 and rp == len(s) - 1:
            sym = s[:lp]
            args_str = s[lp + 1 : rp]
            args = tuple(a.strip() for a in args_str.split(",")) if args_str else tuple()
            return sym, args
        return s, tuple()

    def _fact_true(self, predicate: str, args: Tuple[str, ...], state_bitwise: int) -> bool:
        gid = self._fact_gid.get(predicate, {}).get(tuple(args))
        if gid is None:
            return False
        return ((state_bitwise >> gid) & 1) == 1

    @staticmethod
    def _popcount(x: int) -> int:
        # Python 3.8+ has int.bit_count(), but keep a fallback for compatibility.
        try:
            return x.bit_count()
        except Exception:
            count = 0
            while x:
                x &= x - 1
                count += 1
            return count

    # --------------------------
    # Main heuristic
    # --------------------------
    def h(self, state_bitwise: int, task_network: List[object]) -> int:
        # Component 1: goal facts not yet satisfied (fast bitwise lower bound)
        unsat_goal_bits = self.model.goals & (~state_bitwise)
        h_goals = self._popcount(unsat_goal_bits)

        # Component 2: task-network lower bound by domain-aware counting
        h_tn = 0
        for t in task_network:
            # Primitive operator: at least its own cost (usually 1)
            if hasattr(t, 'apply') and hasattr(t, 'cost') and hasattr(t, 'add_effects'):
                try:
                    op_cost = int(getattr(t, 'cost', 1))
                except Exception:
                    op_cost = 1
                h_tn += max(0, op_cost)
                continue

            # Abstract task: infer a minimal refinement cost based on domain methods
            tname = getattr(t, 'name', '')
            if not tname:
                # Unknown object, be conservative but admissible: 1
                h_tn += 1
                continue

            sym, args = self._taskname_cache.get(tname, (None, None))
            if sym is None:
                sym, args = self._parse_action_or_task_name(tname)
                self._taskname_cache[tname] = (sym, args)

            # Map known abstract tasks
            if sym == 'do_put_on' and len(args) == 2:
                x, y = args
                # If on(x,y) holds now, we can use m0_do_put_on -> 'nop' => LB = 1
                # Else, must use m1 path: do_clear(x)=1, do_clear(y)=1, do_on_table(y)=1, do_move(x,y)=2 => LB = 5
                if self._fact_true(self._P_ON, (x, y), state_bitwise):
                    h_tn += 1
                else:
                    h_tn += 5
            elif sym == 'do_move' and len(args) == 2:
                # m4/m5 both need at least two primitives (pick-up+stack or unstack+stack)
                h_tn += 2
            elif sym == 'do_clear' and len(args) == 1:
                # m6 has 'nop', m7 has unstack/put-down => LB >= 1 in all cases
                h_tn += 1
            elif sym == 'do_on_table' and len(args) == 1:
                # m3 has 'nop', m2 has unstack+put-down => LB >= 1
                h_tn += 1
            else:
                # Any other abstract task (not expected in this domain): conservative LB = 1
                h_tn += 1

        # Use max to avoid double-counting overlapping achievements
        return max(h_goals, h_tn)


# Optional alias for framework discovery (if needed by loader conventions)
HeuristicImpl = BlocksworldGPT5Heuristic17
