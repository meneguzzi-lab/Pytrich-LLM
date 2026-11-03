from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic12(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Rationale (domain insights -> heuristic components)
    -----------------------------------------------
    The domain has 4 compound tasks that decompose into a small set of methods:

      - do_put_on(x,y)
          * m0: if on(x,y) already holds -> (nop)
          * m1: otherwise -> clear x, clear y, ensure y on table, then move x onto y

      - do_on_table(x)
          * m2: if clear(x), handempty, and not ontable(x) -> unstack(x,?) then put-down(x)
          * m3: if clear(x) -> (nop)   (note: may still be not ontable; method-preconditions allow nop)

      - do_move(x,y)
          * m4: if x on table -> pick-up(x) then stack(x,y)
          * m5: else -> unstack(x,?) then stack(x,y)

      - do_clear(x)
          * m6: if clear(x) -> (nop)
          * m7: if something y on x and handempty -> recursively clear y then unstack(y,x) then put-down(y)

    Key planning difficulty is *clearing* blocks (removing a whole tower above them) and
    achieving required on(x,y) relations, which may require clearing both x and y and
    moving x (possibly unstacking first).

    This heuristic is a fast, admissible lower bound combining:

      1) State goal distance: count of unsatisfied goal facts (only +on and +ontable here).
         Each unsatisfied goal relation needs at least one primitive action, so it's a LB.

      2) Task-network lower bounds: for each abstract task in the current task network, compute
         a cheap optimistic cost based on the *cheapest possible method* given current state.
         Additionally, for tasks that require clearing, we add an admissible LB on clearing cost
         by counting the number of blocks above a target block in the current state (each requires
         at least one unstack to remove; put-down is ignored to keep a lower bound).

    Combination method:
      We take max(goal_LB, TN_LB) to keep admissibility (sum could double count).

    Efficiency considerations:
      - All state checks use bitwise operations on fact global_ids.
      - Fact parsing is done once at init to create direct access tables:
          on(x,y), ontable(x), clear(x), handempty, holding(x)
      - The "blocks above" count is computed via a pre-built above adjacency (who is on whom)
        in O(height) for a block; memoized per state evaluation.

    This heuristic is designed to scale to very large instances by keeping per-node work
    linear in |task_network| plus the heights of only referenced blocks.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_12: Optional[dict] = None):
        super().__init__(model)
        self.config = blocksworld_gpt52_heuristic_12 or {}

        # --- Fact id caches (global_id based) ---
        self._handempty_id: Optional[int] = None

        # unary predicate -> arg -> fact_id
        self._clear_id: Dict[str, int] = {}
        self._ontable_id: Dict[str, int] = {}
        self._holding_id: Dict[str, int] = {}

        # binary predicate -> x -> y -> fact_id
        self._on_id: Dict[str, Dict[str, int]] = {}

        # Parse and index facts once.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            if pred == "handempty" and len(args) == 0:
                self._handempty_id = f.global_id
            elif pred == "clear" and len(args) == 1:
                self._clear_id[args[0]] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self._ontable_id[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 1:
                self._holding_id[args[0]] = f.global_id
            elif pred == "on" and len(args) == 2:
                x, y = args
                self._on_id.setdefault(x, {})[y] = f.global_id

        # Precompute goal facts (as Fact global_ids) for fast checking.
        self._goal_fact_ids: List[int] = []
        for gf_name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gf_name)
            if not pred:
                continue
            fid = self._fact_id_from_pred_args(pred, args)
            if fid is not None:
                self._goal_fact_ids.append(fid)

    # ------------------------ Core Heuristic API ------------------------
    def __call__(self, state: int, task_network) -> int:
        # 1) Goal lower bound: count unsatisfied goal facts.
        goal_lb = 0
        for fid in self._goal_fact_ids:
            if ((state >> fid) & 1) == 0:
                goal_lb += 1

        # 2) Task-network lower bound.
        tn_lb = 0

        # For this node, build quick "on" structure and memoize above-count computations.
        # We extract only relations from current state (via bit tests over known ids).
        on_top_of: Dict[str, str] = {}  # top -> below
        # We can build by scanning our on_id dictionary (O(#on-facts)). In blocksworld that is O(n).
        # This is acceptable even for 1000 blocks, and avoids any slow explicit repr.
        for top, below_map in self._on_id.items():
            for below, fid in below_map.items():
                if ((state >> fid) & 1) == 1:
                    on_top_of[top] = below
                    break

        above_cache: Dict[str, int] = {}

        def blocks_above(x: str) -> int:
            """Count how many blocks are (transitively) on top of x in the current state.

            Admissible LB on cost to clear(x): each such block must be unstacked at least once.
            """
            if x in above_cache:
                return above_cache[x]
            # Build inverse pointer: below -> top (only one possible in blocksworld)
            # Instead of building full inverse map, we can search linearly by walking up.
            # But linear search would be O(n^2) in worst case, so build inverse once.
            return 0  # placeholder

        # Build inverse map once.
        top_on: Dict[str, str] = {}  # below -> top
        for top, below in on_top_of.items():
            top_on[below] = top

        def blocks_above(x: str) -> int:
            if x in above_cache:
                return above_cache[x]
            c = 0
            cur = x
            while cur in top_on:
                c += 1
                cur = top_on[cur]
            above_cache[x] = c
            return c

        # Evaluate each task in current task network.
        # task_network elements are Operators or AbstractTasks.
        for t in task_network:
            name = getattr(t, "name", "")
            if name.startswith("do_clear"):
                x = self._extract_args(name, 1)[0]
                # If already clear -> 0, else at least blocks_above(x) unstack actions.
                if not self._is_true_clear(state, x):
                    tn_lb += max(1, blocks_above(x))  # ensure some guidance if structure missing
            elif name.startswith("do_on_table"):
                x = self._extract_args(name, 1)[0]
                # Cheapest: if clear(x) then method m3 can do nop.
                # Otherwise, need at least one unstack to make it clear.
                if not self._is_true_clear(state, x):
                    tn_lb += max(1, blocks_above(x))
                else:
                    # If it's clear but not ontable, minimal primitive work could be 0 in this HTN
                    # (due to m3). Keep LB as 0.
                    tn_lb += 0
            elif name.startswith("do_move"):
                x, y = self._extract_args(name, 2)
                # To do_move(x,y), must make x and y clear and handempty.
                # Lower bound: if x not clear -> need blocks_above(x) unstack.
                #            if y not clear -> need blocks_above(y) unstack.
                #            plus at least 1 action to place x on y (stack).
                lb = 0
                if not self._is_true_clear(state, x):
                    lb += max(1, blocks_above(x))
                if not self._is_true_clear(state, y):
                    lb += max(1, blocks_above(y))
                # If on(x,y) already true, this task might not appear, but keep conservative.
                if not self._is_true_on(state, x, y):
                    lb += 1
                tn_lb += lb
            elif name.startswith("do_put_on"):
                x, y = self._extract_args(name, 2)
                # If already on(x,y) -> 0.
                # Else, need at least: clear x (if not), clear y (if not), and at least 1 stack/unstack.
                if self._is_true_on(state, x, y):
                    tn_lb += 0
                else:
                    lb = 1  # at least one action to establish relation (stack)
                    if not self._is_true_clear(state, x):
                        lb += max(1, blocks_above(x))
                    if not self._is_true_clear(state, y):
                        lb += max(1, blocks_above(y))
                    tn_lb += lb
            else:
                # Primitive operators: admissible LB is 0 since they might be immediately applicable.
                tn_lb += 0

        return max(goal_lb, tn_lb)

    # ------------------------ Helpers ------------------------
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

    def _fact_id_from_pred_args(self, pred: str, args: List[str]) -> Optional[int]:
        if pred == "handempty" and len(args) == 0:
            return self._handempty_id
        if pred == "clear" and len(args) == 1:
            return self._clear_id.get(args[0])
        if pred == "ontable" and len(args) == 1:
            return self._ontable_id.get(args[0])
        if pred == "holding" and len(args) == 1:
            return self._holding_id.get(args[0])
        if pred == "on" and len(args) == 2:
            return self._on_id.get(args[0], {}).get(args[1])
        return None

    def _is_true_clear(self, state: int, x: str) -> bool:
        fid = self._clear_id.get(x)
        return fid is not None and ((state >> fid) & 1) == 1

    def _is_true_on(self, state: int, x: str, y: str) -> bool:
        fid = self._on_id.get(x, {}).get(y)
        return fid is not None and ((state >> fid) & 1) == 1

    def _extract_args(self, task_name: str, arity: int) -> List[str]:
        """Extract arguments from grounded task/operator name.

        Pytrich uses names like 'do_put_on b1 b2' or sometimes 'do_put_on(b1,b2)'
        depending on the pipeline. We support both, conservatively.
        """
        s = task_name.strip()
        # Try parenthesis/comma format
        if "(" in s and s.endswith(")"):
            inside = s[s.find("(") + 1 : -1]
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            if len(parts) >= arity:
                return parts[:arity]

        # Fall back to space-separated: 'name arg1 arg2'
        parts = s.replace("\t", " ").split()
        # first token is the name
        if len(parts) >= 1 + arity:
            return parts[1 : 1 + arity]

        # If cannot parse, return placeholders to avoid crashing.
        return ["?"] * arity
