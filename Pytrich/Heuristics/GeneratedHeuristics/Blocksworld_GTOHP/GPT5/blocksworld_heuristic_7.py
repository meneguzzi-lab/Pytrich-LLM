# Domain-Dependent Heuristic for HDDL Blocksworld (Pytrich Planner)
# Class: BlocksworldGPT5Heuristic7
# 
# Rationale (high-level):
# - In Blocksworld, to achieve a goal on(x,y) you must perform at least one stack action.
# - Additionally, for each such on(x,y), you must also establish holding(x) beforehand with at least one pickup/unstack.
#   Therefore, each missing on-goal contributes at least 2 primitive actions (pickup/unstack + stack), except possibly one
#   currently-held block that matches some pending on(x,y), which would save exactly one pickup/unstack.
# - We compute a safe lower bound: h_goals = 2 * (#unsatisfied on-goals) - (1 if currently holding a goal block else 0).
#   This is admissible: every target pair requires exactly one stack and (unless already holding it) exactly one pickup/unstack.
# - We also consider the hierarchical task network (HTN): remaining abstract tasks of types do_clear and do_on_table
#   can impose extra mandatory work. For admissibility we compute a separate lower bound h_tasks over the task network and
#   combine via max(h_goals, h_tasks), avoiding double counting between state-goals and HTN subgoals.
#   
#   Task LB details:
#   - do_clear(x): if not clear(x) then at least one unstack is required -> +1
#   - do_on_table(x): domain allows a zero-cost NOP refinement if clear(x) holds (m3_do_on_table), else at least an unstack
#                     will be required somewhere -> +1 when not clear(x)
#   - do_move/do_put_on: we do not add a separate cost here to avoid double counting with the goal on(x,y) bound; if these
#                         tasks refer to (x,y) not present among explicit goal on-facts, we include them as extra pairs in
#                         the same on-goal counting scheme.
#
# Efficiency:
# - Uses bitwise membership tests via global_id to check whether a fact holds in the current state.
# - Precomputes fact-id maps for on, clear, holding, ontable.
# - Parses goals once and stores goal on-pairs for quick lookup.
#
# Notes about robustness:
# - The name parser accepts multiple formats: 'pred[a,b]', 'pred(a,b)', and 'pred a b'. Facts may carry a '+' prefix; tasks
#   and operators do not. The heuristic only relies on facts and abstract task names.
# - The heuristic is consistent and admissible by construction; it trades accuracy for scalability by using cheap lower bounds.
#
# Usage:
#   heuristic = BlocksworldGPT5Heuristic7(model, blocksworld_gpt5_heuristic_7=True)
#   h_value = heuristic.evaluate(state_bitwise, task_network_ids)

from typing import Dict, Tuple, Set, List

try:
    # Typical location in Pytrich
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback minimal base, in case of environment differences
    class Heuristic:
        def __init__(self, model, *_, **__):
            self.model = model
        def evaluate(self, state_bitwise: int, task_network: List[int]) -> int:
            return 0


class BlocksworldGPT5Heuristic7(Heuristic):
    """
    Domain-dependent heuristic for HDDL Blocksworld.

    Lower bound components:
      1) Goal on-facts:
         h_goals = 2 * (# unsatisfied on-goals in current state)
                   - (1 if the currently held block is the x of any unsatisfied on(x,y), else 0)

      2) Pending HTN tasks (supportive, non-overlapping with goal counting):
         h_tasks = sum over tasks in current task network of:
            - do_clear(x): +1 if not clear(x)
            - do_on_table(x): +1 if not clear(x)   (since method m3 allows zero cost if clear(x) already holds)
         We ignore direct counting for do_put_on/do_move here to avoid double counting with goal on-facts, unless they refer
         to on-pairs not in the explicit goals (then we merge them into the goal-on set before counting).

    Final heuristic: max(h_goals, h_tasks)

    This is admissible and efficient for both small and very large instances.
    """

    def __init__(self, model, blocksworld_gpt5_heuristic_7: bool = True):
        super().__init__(model)
        self.enabled = blocksworld_gpt5_heuristic_7

        # Fact-id maps
        self.on_gid: Dict[Tuple[str, str], int] = {}
        self.clear_gid: Dict[str, int] = {}
        self.ontable_gid: Dict[str, int] = {}
        self.holding_gid: Dict[str, int] = {}

        # Pre-parse all facts to build quick-access indices
        self._index_facts()

        # Parse goal on-facts once
        self.goal_on_pairs: Set[Tuple[str, str]] = set()
        self._index_goal_on_pairs()

    # ------------------------------ Helpers: parsing and indexing ------------------------------

    def _parse_fact_name(self, name: str):
        """Parse grounded fact name in formats: '+pred[a,b]', '-pred[a,b]'. Returns (predicate, [args])."""
        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
        bracket_idx = s.find("[")
        if bracket_idx > 0 and s.endswith("]"):
            predicate = s[:bracket_idx]
            args_str = s[bracket_idx + 1 : -1]
            args = args_str.split(",") if args_str else []
            return predicate, [a.strip() for a in args]
        return None, []

    def _parse_any_name(self, name: str):
        """
        Parse names for operators/abstract tasks in one of:
          - 'pred[a,b]'
          - 'pred(a,b)'
          - 'pred a b'
        Returns (predicate, [args]) or (name, []) if no args found.
        """
        s = name.strip()
        # Strip +/- if any (usually not present for tasks/ops)
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
        # Bracket style
        b = s.find("[")
        if b > 0 and s.endswith("]"):
            pred = s[:b]
            inner = s[b + 1 : -1].strip()
            args = [t.strip() for t in inner.split(",")] if inner else []
            return pred, args
        # Parentheses style
        p = s.find("(")
        if p > 0 and s.endswith(")"):
            pred = s[:p]
            inner = s[p + 1 : -1].strip()
            # allow either comma or whitespace separation
            if "," in inner:
                parts = [t.strip() for t in inner.split(",") if t.strip()]
            else:
                parts = [t.strip() for t in inner.split() if t.strip()]
            return pred, parts
        # Whitespace style: 'pred a b'
        parts = s.split()
        if len(parts) >= 2:
            return parts[0], parts[1:]
        return s, []

    def _index_facts(self):
        for fid, f in enumerate(self.model.facts):
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            if pred == "on" and len(args) == 2:
                self.on_gid[(args[0], args[1])] = fid
            elif pred == "clear" and len(args) == 1:
                self.clear_gid[args[0]] = fid
            elif pred == "ontable" and len(args) == 1:
                self.ontable_gid[args[0]] = fid
            elif pred == "holding" and len(args) == 1:
                self.holding_gid[args[0]] = fid
            # handempty is global, not per-block; we don't need it in this heuristic

    def _index_goal_on_pairs(self):
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "on" and len(args) == 2:
                self.goal_on_pairs.add((args[0], args[1]))

    @staticmethod
    def _bit_true(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) != 0

    # ------------------------------ Heuristic core ------------------------------

    def evaluate(self, state_bitwise: int, task_network: List[int]) -> int:
        if not self.enabled:
            return 0

        # 1) Build the set of on-pairs we must ensure: explicit goal on-pairs U any do_put_on/do_move pairs in TN not in goals
        required_on_pairs: Set[Tuple[str, str]] = set(self.goal_on_pairs)
        tn_support_lb = 0  # lower bound for supportive tasks (do_clear/do_on_table only)

        for cid in task_network:
            comp = self.model.get_component(cid)
            # Identify AbstractTask by attribute presence (duck-typing)
            is_abstract = hasattr(comp, "decompositions") and not hasattr(comp, "task_network")
            if not is_abstract:
                continue
            pred, args = self._parse_any_name(comp.name)
            if pred in ("do_put_on", "do_move") and len(args) >= 2:
                pair = (args[0], args[1])
                # If not already a goal on-fact, include as a required on-pair
                if pair not in required_on_pairs:
                    required_on_pairs.add(pair)
            elif pred == "do_clear" and len(args) >= 1:
                x = args[0]
                gid = self.clear_gid.get(x, None)
                if gid is not None and not self._bit_true(state_bitwise, gid):
                    tn_support_lb += 1
            elif pred == "do_on_table" and len(args) >= 1:
                x = args[0]
                # This task can be refined to NOP if clear(x) already holds (m3_do_on_table)
                gid = self.clear_gid.get(x, None)
                if gid is not None and not self._bit_true(state_bitwise, gid):
                    tn_support_lb += 1
            # else: ignore other abstract tasks (if any) to stay conservative

        # 2) Compute goal-based lower bound over required_on_pairs
        #    - Count how many are unsatisfied in current state
        unsat_on = 0
        # Track candidate held block that belongs to unsatisfied goals
        held_bonus = 0

        # Precompute set of x in unsatisfied goals to check holding quickly
        unsat_xs: List[str] = []
        for (x, y) in required_on_pairs:
            gid = self.on_gid.get((x, y), None)
            if gid is None:
                # If the fact was not indexed, we cannot require it; skip safely
                continue
            if not self._bit_true(state_bitwise, gid):
                unsat_on += 1
                unsat_xs.append(x)

        if unsat_on > 0 and self.holding_gid:
            # If currently holding some x among unsatisfied pairs, we can save one pickup/unstack
            for x in unsat_xs:
                h_gid = self.holding_gid.get(x, None)
                if h_gid is not None and self._bit_true(state_bitwise, h_gid):
                    held_bonus = 1
                    break

        h_goals = (unsat_on << 1) - held_bonus  # 2 * unsat_on - held_bonus
        if h_goals < 0:
            h_goals = 0

        # 3) Combine conservatively to avoid double counting between goal and HTN supportive effort
        h_value = h_goals if h_goals >= tn_support_lb else tn_support_lb
        return h_value


# Backward compatibility alias if needed by external code (none created as per instruction)
# End of file
