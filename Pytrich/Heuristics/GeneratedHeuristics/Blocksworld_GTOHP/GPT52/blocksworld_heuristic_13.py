from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# The framework provides Heuristic base class at runtime.
# We only rely on its interface: self.model and __call__(node) -> number.
try:
    from Pytrich.Heuristics.heuristic import Heuristic  # typical location
except Exception:  # pragma: no cover
    try:
        from Heuristics.heuristic import Heuristic
    except Exception:  # pragma: no cover
        class Heuristic:  # fallback stub for type checkers
            def __init__(self, model, *args, **kwargs):
                self.model = model


class BlocksworldGPT52Heuristic13(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Key domain insights used
    -----------------------
    The domain defines abstract tasks that ultimately reduce to the classic
    Blocksworld primitive actions:
      - pick-up, put-down, stack, unstack, nop

    The HTN methods reveal typical effort patterns:
      - do_move(x,y): either pick-up+stack (if x on table) or unstack+stack (if x not on table)
        => at least 2 primitive actions whenever on(x,y) is not already true.
      - do_on_table(x): if x is not already ontable and clear, requires unstack+put-down
        => at least 2 primitive actions.
      - do_clear(x): if not clear, recursively clear the top block then unstack+put-down
        => at least 2 primitive actions per blocking block.
      - do_put_on(x,y): if already on(x,y) => 0; else requires clearing x and y, forcing y to table,
        and moving x onto y.

    Heuristic design
    ----------------
    We compute a fast admissible lower bound as a sum of independent lower bounds:

    (A) Task-network lower bound (HTN-aware):
        For each remaining abstract task in the current task network,
        add a cheap optimistic estimate of the minimal number of primitive
        actions still needed, based on whether its achievement literal already holds.
        This exploits that many tasks in instances are "do_put_on".

    (B) Goal-literal lower bound (state-based):
        For each goal fact not yet true, add 1 as a minimal correction count.

    (C) Clearing obstruction lower bound (very lightweight):
        For each required goal on(a,b) not yet true, at least the top of a and b
        must be cleared at execution time. If currently not clear(a) or not clear(b),
        add 1 each (optimistic: assumes one action can fix it).

    Finally we take max(A, B+C) to keep admissibility (max of lower bounds is a lower bound)
    while avoiding over-counting across components.

    Performance considerations
    --------------------------
    - No explicit state enumeration via state_explicit_repr(state).
    - All checks are bitwise using precomputed fact global ids.
    - Task parsing uses string inspection only (task names are few) and avoids heavy regex.

    Notes
    -----
    This heuristic is *admissible* as constructed (only optimistic costs, combined by max).
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_13: Optional[dict] = None):
        super().__init__(model)
        self.blocksworld_gpt52_heuristic_13 = blocksworld_gpt52_heuristic_13 or {}

        # Fact-id caches for fast bit checks
        self._fid_on: Dict[Tuple[str, str], int] = {}
        self._fid_ontable: Dict[str, int] = {}
        self._fid_clear: Dict[str, int] = {}
        self._fid_handempty: Optional[int] = None
        self._fid_holding: Dict[str, int] = {}

        # Parse and cache all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2:
                self._fid_on[(args[0], args[1])] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self._fid_ontable[args[0]] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self._fid_clear[args[0]] = f.global_id
            elif pred == "handempty" and len(args) == 0:
                self._fid_handempty = f.global_id
            elif pred == "holding" and len(args) == 1:
                self._fid_holding[args[0]] = f.global_id

        # Cache goal facts (as ids and parsed structure) once
        self._goal_fact_ids: List[int] = []
        self._goal_on_pairs: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            # Goals are positive in provided instances
            if pred and name.startswith("+"):
                fid = self._fact_id_from_name(pred, args)
                if fid is not None:
                    self._goal_fact_ids.append(fid)
                if pred == "on" and len(args) == 2:
                    self._goal_on_pairs.append((args[0], args[1]))

    # ---------- core heuristic ----------
    def __call__(self, node) -> int:
        state = node.state
        tn = getattr(node, "task_network", None)
        if tn is None:
            tn = getattr(node, "tn", [])

        # (A) HTN task-network optimistic cost
        h_tn = self._tn_lower_bound(state, tn)

        # (B) Unsatisfied goal facts count
        h_goal = 0
        for gid in self._goal_fact_ids:
            if ((state >> gid) & 1) == 0:
                h_goal += 1

        # (C) Cheap "must be clear" corrections for goal on(a,b) not yet satisfied
        h_clear = 0
        for a, b in self._goal_on_pairs:
            gid_on = self._fid_on.get((a, b), None)
            if gid_on is not None and ((state >> gid_on) & 1) == 1:
                continue
            # If on(a,b) not true, eventually a and b must be clear to stack.
            gid_ca = self._fid_clear.get(a)
            if gid_ca is not None and ((state >> gid_ca) & 1) == 0:
                h_clear += 1
            gid_cb = self._fid_clear.get(b)
            if gid_cb is not None and ((state >> gid_cb) & 1) == 0:
                h_clear += 1

        # Combine lower bounds without double counting by taking max
        return max(h_tn, h_goal + h_clear)

    # ---------- task-network lower bound ----------
    def _tn_lower_bound(self, state: int, task_network: List[Union[object, int]]) -> int:
        """Optimistic cost of completing remaining task network.

        We interpret primitive operators as cost 1, and abstract tasks by
        a minimal action count derived from their intended effect.
        """
        if not task_network:
            return 0

        lb = 0
        for t in task_network:
            # Primitive operator: has 'pos_precons' attribute in this framework
            if hasattr(t, "pos_precons") and hasattr(t, "add_effects"):
                lb += getattr(t, "cost", 1)
                continue

            # Abstract task: use name and parameters embedded in t.name
            name = getattr(t, "name", "")
            # Names often look like: "(do_put_on b1 b2)" or "do_put_on" depending on printer.
            tname, args = self._parse_task_signature(name)

            if tname == "do_put_on" and len(args) == 2:
                x, y = args
                gid_on = self._fid_on.get((x, y))
                if gid_on is not None and ((state >> gid_on) & 1) == 1:
                    # m0_do_put_on can do nop
                    continue
                # Otherwise at least a move is needed => >=2 actions (pick+stack or unstack+stack)
                lb += 2
                continue

            if tname == "do_move" and len(args) == 2:
                x, y = args
                gid_on = self._fid_on.get((x, y))
                if gid_on is not None and ((state >> gid_on) & 1) == 1:
                    continue
                lb += 2
                continue

            if tname == "do_on_table" and len(args) == 1:
                x = args[0]
                gid_ot = self._fid_ontable.get(x)
                if gid_ot is not None and ((state >> gid_ot) & 1) == 1:
                    continue
                # If not on table, minimal is unstack+put-down => 2
                lb += 2
                continue

            if tname == "do_clear" and len(args) == 1:
                x = args[0]
                gid_c = self._fid_clear.get(x)
                if gid_c is not None and ((state >> gid_c) & 1) == 1:
                    continue
                # At least one unstack+put-down somewhere => >=2, but keep it conservative
                lb += 1
                continue

            # Unknown/other abstract task: give 0 to stay admissible
            # (This domain only uses the tasks above.)
            lb += 0

        return lb

    # ---------- parsing helpers ----------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _fact_id_from_name(self, pred: str, args: List[str]) -> Optional[int]:
        if pred == "on" and len(args) == 2:
            return self._fid_on.get((args[0], args[1]))
        if pred == "ontable" and len(args) == 1:
            return self._fid_ontable.get(args[0])
        if pred == "clear" and len(args) == 1:
            return self._fid_clear.get(args[0])
        if pred == "handempty" and len(args) == 0:
            return self._fid_handempty
        if pred == "holding" and len(args) == 1:
            return self._fid_holding.get(args[0])
        return None

    def _parse_task_signature(self, task_name: str) -> Tuple[str, List[str]]:
        """Parse an AbstractTask.name into (task_symbol, [args]).

        Supports common renderings:
          - "(do_put_on b1 b2)"
          - "do_put_on" (no args)
          - "do_put_on(b1,b2)" (rare)
        """
        s = (task_name or "").strip()
        if not s:
            return "", []

        # Strip surrounding parentheses if present
        if s[0] == "(" and s[-1] == ")":
            s = s[1:-1].strip()

        # Space-separated form: do_put_on b1 b2
        parts = s.split()
        if len(parts) >= 1:
            sym = parts[0]
            if sym.startswith("task") and len(parts) == 1:
                # sometimes names may be like 'task1' etc.; can't parse
                return s, []
            if len(parts) > 1 and sym.startswith("do_"):
                return sym, parts[1:]

        # Function-like form: do_put_on(b1,b2)
        par_i = s.find("(")
        if par_i > 0 and s.endswith(")"):
            sym = s[:par_i].strip()
            inside = s[par_i + 1 : -1].strip()
            args = [a.strip() for a in inside.split(",") if a.strip()] if inside else []
            return sym, args

        return s, []
