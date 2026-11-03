from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic
class DepotsGPT5Heuristic2(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Depots domain.

    High-level idea (admissible lower bound):
    - Goals in Depots are almost exclusively +on[crate, surface]. The only action that adds an +on is Drop.
      Therefore, every unsatisfied +on goal requires at least one Drop.
    - We add a tighter per-goal lower bound: minimum number of primitive actions needed (in the best case) to
      achieve on[c, s] from the current state. This considers:
        * number of Lifts needed to clear the destination surface s (unstacking top-down),
        * number of Lifts needed to clear crate c (unstacking crates above c),
        * a minimal movement schema among {local Lift+Drop, unloading from truck, or transport via truck}
          counting Drives pessimistically-low by checking if a truck is already at the source.
      This is still an optimistic (delete-relaxed) estimate -> admissible.
    - For HTN, the plan must also execute all remaining primitive operators in the current task network.
      Summing these would remain a lower bound on the remaining steps, but to avoid double counting with goal
      estimates we conservatively take the maximum of: (a) count of unsatisfied on-goals (number of Drops),
      (b) the maximum per-goal cost, and (c) the number of remaining primitive operators in the task network.

    This heuristic avoids scanning the entire fact set on every node by pre-indexing facts by predicate/arguments
    and uses bitwise membership tests, as required by Pytrich.
    """

    def __init__(self, model, depots_gpt5_heuristic_2: bool = True, verbose: bool = False):
        super().__init__(model)
        self.verbose = verbose

        # Pre-index facts
        self.fact_by_key: Dict[Tuple[str, Tuple[str, ...]], object] = {}
        self.on_by_first: Dict[str, List[object]] = {}      # c -> list of +on[c, *]
        self.on_by_second: Dict[str, List[object]] = {}     # s -> list of +on[*, s]
        self.in_by_first: Dict[str, List[object]] = {}      # c -> list of +in[c, *]
        self.lifting_by_second: Dict[str, List[object]] = {}# c -> list of +lifting[*, c]
        self.at_by_first: Dict[str, List[object]] = {}      # x -> list of +at[x, p]
        self.at_by_pair: Dict[Tuple[str, str], object] = {} # (x, p) -> +at[x,p]
        self.clear_by_arg: Dict[str, object] = {}           # x -> +clear[x]

        # For fast arg access per fact id
        self._fact_pred_args_by_id: List[Tuple[str, Tuple[str, ...]]] = []

        # Collect set of truck names (appear as second arg of 'in')
        self.trucks: Set[str] = set()

        # Build all indexes once
        for f in model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                # Should not happen for well-formed models
                self._fact_pred_args_by_id.append(("", tuple()))
                continue
            self._fact_pred_args_by_id.append((pred, tuple(args)))

            self.fact_by_key[(pred, tuple(args))] = f

            if pred == 'on' and len(args) == 2:
                c, s = args
                self.on_by_first.setdefault(c, []).append(f)
                self.on_by_second.setdefault(s, []).append(f)
            elif pred == 'in' and len(args) == 2:
                c, t = args
                self.in_by_first.setdefault(c, []).append(f)
                self.trucks.add(t)
            elif pred == 'lifting' and len(args) == 2:
                h, c = args
                self.lifting_by_second.setdefault(c, []).append(f)
            elif pred == 'at' and len(args) == 2:
                x, p = args
                self.at_by_first.setdefault(x, []).append(f)
                self.at_by_pair[(x, p)] = f
            elif pred == 'clear' and len(args) == 1:
                x = args[0]
                self.clear_by_arg[x] = f

        # Pre-parse goal on-pairs once (goals are static)
        self.goal_on_pairs: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for g in goal_fact_names:
            gp, ga = self._parse_fact_name(g)
            if gp == 'on' and len(ga) == 2:
                self.goal_on_pairs.append((ga[0], ga[1]))

        if self.verbose:
            print(f"[DepotsGPT5Heuristic2] Indexed facts: on={len(self.on_by_first)} crates, trucks={len(self.trucks)}")
            print(f"[DepotsGPT5Heuristic2] Goal on-pairs: {len(self.goal_on_pairs)}")

    # ----------------------------- Core Heuristic API -----------------------------
    def h(self, state: int, task_network: List) -> int:
        # Lower bound 1: at least one Drop per unsatisfied on-goal
        unsat_count = 0
        per_goal_costs: List[int] = []

        # Caches for the current state
        cache_above_count: Dict[str, int] = {}
        cache_clear_surface: Dict[str, int] = {}

        for (c, s) in self.goal_on_pairs:
            fact = self.fact_by_key.get(('on', (c, s)))
            if fact is not None and self._is_true(state, fact):
                continue  # already satisfied

            unsat_count += 1
            cost_g = self._estimate_on_goal_cost(state, c, s, cache_above_count, cache_clear_surface)
            per_goal_costs.append(cost_g)

        h_goals_max = max(per_goal_costs) if per_goal_costs else 0
        drop_lb = unsat_count

        # Lower bound 2: primitive operators pending in the current task network
        primitive_lb = 0
        if task_network:
            for tn_elem in task_network:
                # A primitive operator in Pytrich has .cost and .apply
                if hasattr(tn_elem, 'cost') and hasattr(tn_elem, 'apply'):
                    try:
                        primitive_lb += int(getattr(tn_elem, 'cost', 1) or 1)
                    except Exception:
                        primitive_lb += 1

        # Final admissible bound: take the maximum of independent admissible bounds
        h_val = max(drop_lb, h_goals_max, primitive_lb)
        return int(h_val)

    # ----------------------------- Estimation helpers -----------------------------
    def _estimate_on_goal_cost(
        self,
        state: int,
        c: str,
        s: str,
        cache_above_count: Dict[str, int],
        cache_clear_surface: Dict[str, int]
    ) -> int:
        # 0) If goal already true (should be filtered before), return 0
        fact_goal = self.fact_by_key.get(('on', (c, s)))
        if fact_goal is not None and self._is_true(state, fact_goal):
            return 0

        # 1) Lifts required to clear the destination surface s
        lifts_clear_s = self._clear_surface_cost(state, s, cache_above_count, cache_clear_surface)

        # 2) Lifts required to clear the crate c (unstack crates above c)
        lifts_above_c = self._above_count(state, c, cache_above_count)

        # 3) Base movement cost to actually place c onto s with optimistic assumptions
        base_move = self._base_move_cost(state, c, s)

        # Total lower bound for this goal
        return lifts_clear_s + lifts_above_c + base_move

    def _base_move_cost(self, state: int, c: str, s: str) -> int:
        # Get destination place of s
        p_s = self._place_of_x(state, s)

        # If already lifted: can Drop locally if at same place as s
        lifted_fact = self._get_true_lifting_fact(state, c)
        if lifted_fact is not None:
            # lifting[h, c] true; get hoist place p_lift
            _, args = self._get_pred_args_for_fact(lifted_fact)
            h = args[0]
            p_lift = self._place_of_x(state, h)
            if p_lift is not None and p_s is not None and p_lift == p_s:
                # Just Drop: 1 action
                return 1
            else:
                # Use truck from p_lift to p_s: bring truck (if needed) + Load + (Drive if needed) + Unload + Drop
                drives_pre = 0 if (p_lift is not None and self._exists_truck_at(state, p_lift)) else 1
                drive_to_dest = 0 if (p_lift is not None and p_s is not None and p_lift == p_s) else 1
                return drives_pre + 1 + drive_to_dest + 2  # Load + (Un)Load/Drop

        # If c is in a truck already
        p_truck = self._place_of_truck_holding_c(state, c)
        if p_truck is not None:
            if p_s is not None and p_truck == p_s:
                return 2  # Unload + Drop
            else:
                return 1 + 2  # one Drive to p_s + (Unload + Drop)

        # Otherwise c is resting somewhere; get its place
        p_c = self._place_of_crate(state, c)

        # If c and s co-located, local Lift+Drop
        if p_c is not None and p_s is not None and p_c == p_s:
            return 2  # Lift + Drop

        # Else, transport via truck from p_c to p_s: (bring truck if needed) + Lift + Load + (Drive if needed) + Unload + Drop
        drives_pre = 0 if (p_c is not None and self._exists_truck_at(state, p_c)) else 1
        drive_to_dest = 0 if (p_c is not None and p_s is not None and p_c == p_s) else 1
        return drives_pre + 1 + 1 + drive_to_dest + 2  # Lift + Load + (Drive) + Unload + Drop

    # ----------------------------- Stack/clear computations -----------------------------
    def _above_count(self, state: int, c: str, cache: Dict[str, int]) -> int:
        # Number of crates stacked on top of crate c (each requires at least one Lift)
        if c in cache:
            return cache[c]
        # If c is clear => 0
        clear_c = self.clear_by_arg.get(c)
        if clear_c is not None and self._is_true(state, clear_c):
            cache[c] = 0
            return 0
        # Find the top-most crate directly on c (if any). If multiple, take one (should be unique in valid states)
        top_on_c = self._get_true_on_child(state, c)
        if top_on_c is None:
            # Not clear, but no child found -> conservatively assume 1
            cache[c] = 1
            return 1
        # 1 Lift for the child (eventually), plus whatever is above the child
        child = top_on_c
        val = 1 + self._above_count(state, child, cache)
        cache[c] = val
        return val

    def _clear_surface_cost(
        self,
        state: int,
        s: str,
        cache_above: Dict[str, int],
        cache_clear_s: Dict[str, int]
    ) -> int:
        # Number of Lifts needed to clear surface s
        if s in cache_clear_s:
            return cache_clear_s[s]
        clear_s = self.clear_by_arg.get(s)
        if clear_s is not None and self._is_true(state, clear_s):
            cache_clear_s[s] = 0
            return 0
        # Otherwise, find the crate directly on s, call it c1. To be able to lift c1, we must
        # lift everything above c1 first, then lift c1 -> 1 + above_count(c1)
        c1 = self._get_true_on_child(state, s)
        if c1 is None:
            # Inconsistent: surface not clear but no child found => at least 1 lift
            cache_clear_s[s] = 1
            return 1
        val = 1 + self._above_count(state, c1, cache_above)
        cache_clear_s[s] = val
        return val

    # ----------------------------- State queries (bitwise) -----------------------------
    def _is_true(self, state: int, fact_obj) -> bool:
        return ((state >> fact_obj.global_id) & 1) == 1

    def _get_true_on_child(self, state: int, parent: str) -> Optional[str]:
        # Return crate x such that +on[x,parent] is true in state
        for f in self.on_by_second.get(parent, []):
            if self._is_true(state, f):
                _, args = self._get_pred_args_for_fact(f)
                return args[0]  # child crate name
        return None

    def _get_true_lifting_fact(self, state: int, c: str):
        for f in self.lifting_by_second.get(c, []):
            if self._is_true(state, f):
                return f
        return None

    def _place_of_truck_holding_c(self, state: int, c: str) -> Optional[str]:
        # If c is in some truck t, return p where +at[t,p] is true
        for f in self.in_by_first.get(c, []):
            if self._is_true(state, f):
                _, args = self._get_pred_args_for_fact(f)
                t = args[1]
                # find at[t, p]
                for af in self.at_by_first.get(t, []):
                    if self._is_true(state, af):
                        _, aargs = self._get_pred_args_for_fact(af)
                        return aargs[1]
        return None

    def _place_of_crate(self, state: int, c: str) -> Optional[str]:
        # If c is on some surface s, return place of s; else if at[c,p] is true, return p; else None
        # Check on[c, s]
        for f in self.on_by_first.get(c, []):
            if self._is_true(state, f):
                _, args = self._get_pred_args_for_fact(f)
                s = args[1]
                return self._place_of_x(state, s)
        # Direct at[c, p]
        for f in self.at_by_first.get(c, []):
            if self._is_true(state, f):
                _, args = self._get_pred_args_for_fact(f)
                return args[1]
        return None

    def _place_of_x(self, state: int, x: str) -> Optional[str]:
        # Return place p where +at[x,p] is true in state
        for f in self.at_by_first.get(x, []):
            if self._is_true(state, f):
                _, args = self._get_pred_args_for_fact(f)
                return args[1]
        return None

    def _exists_truck_at(self, state: int, p: Optional[str]) -> bool:
        if p is None:
            return False
        for t in self.trucks:
            f = self.at_by_pair.get((t, p))
            if f is not None and self._is_true(state, f):
                return True
        return False

    # ----------------------------- Parsing helpers -----------------------------
    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx+1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _get_pred_args_for_fact(self, fact_obj) -> Tuple[str, Tuple[str, ...]]:
        # Fast access via global id
        return self._fact_pred_args_by_id[fact_obj.global_id]


# Alias for external configuration option name ensure consistency with requested parameter name
__all__ = ['DepotsGPT5Heuristic2']
