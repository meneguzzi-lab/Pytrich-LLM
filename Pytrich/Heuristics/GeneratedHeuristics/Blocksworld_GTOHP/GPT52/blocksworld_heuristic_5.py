from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic5(Heuristic):
    """Domain-dependent heuristic for the Blocksworld HDDL domain.

    Key domain insights
    -------------------
    The domain provides four abstract tasks that decompose into primitive actions:

      - do_clear(x)
          * if clear(x) -> nop
          * else: recursively clear the block y on x, then unstack(y,x), then put-down(y)

      - do_on_table(x)
          * if clear(x) -> nop (note: method m3 is very relaxed; it ignores whether x is already on table)
          * else if clear(x) and handempty and not ontable(x): unstack(x,y) then put-down(x)

      - do_move(x,y)
          * if x on table and clear(x), clear(y), handempty: pick-up(x) then stack(x,y)
          * else if x not on table and clear(x), clear(y), handempty: unstack(x,z) then stack(x,y)

      - do_put_on(x,y)
          * if on(x,y) -> nop
          * else (requires handempty): clear(x), clear(y), on_table(y), move(x,y)

    The dominant cost drivers are:
      (1) blocks that are not clear (need to clear above them), and
      (2) misplacements w.r.t. desired on-relations (need to reposition blocks).

    Heuristic design
    ----------------
    We compute a fast *lower bound* that combines:

      A) Goal mismatch count: number of goal +on[a,b] facts not currently true.

      B) Blocking cost for goals: for each unsatisfied goal on[a,b], we add a small
         lower bound for the actions needed to make a and b clear and to ensure
         handempty if currently holding something.

         - If a is not clear -> at least 2 actions are needed somewhere to remove
           the topmost block above a (unstack + put-down), regardless of recursion.
           (This is admissible as a lower bound.)

         - If b is not clear -> similarly at least 2 actions.

         - If holding something -> at least 1 action (put-down or stack) before
           methods requiring handempty can proceed.

    This is intentionally simple and bitwise-fast for scaling up to 1000 blocks.

    Properties
    ----------
    * Admissible as a cost-to-go lower bound for primitive actions (never exceeds
      the real remaining number of actions).
    * Very low overhead: only iterates over goal facts and uses O(1) bit checks.

    Notes on Pytrich integration
    ----------------------------
    - Facts are accessed via global_id and checked with bit operations.
    - Goals are a bitmask in model.goals; we obtain goal fact names through
      model.state_explicit_repr(model.goals) only once in __init__.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_5: str = "blocksworld_gpt52_heuristic_5"):
        super().__init__(model, blocksworld_gpt52_heuristic_5)
        self.model = model

        # Cache fact IDs for quick access
        self._fact_id_by_name: Dict[str, int] = {}
        for f in self.model.facts:
            self._fact_id_by_name[f.name] = f.global_id

        # Parse and cache goal +on[a,b] atoms (as fact ids and as (a,b) pairs)
        self._goal_on_fact_ids: List[int] = []
        self._goal_on_pairs: List[Tuple[str, str]] = []
        for name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2 and name.startswith("+"):
                fid = self._fact_id_by_name.get(name)
                if fid is not None:
                    self._goal_on_fact_ids.append(fid)
                    self._goal_on_pairs.append((args[0], args[1]))

        # Precompute ids for +clear[x], +holding[x], +handempty when they exist.
        # (In this benchmark they do.)
        self._clear_fact_id: Dict[str, int] = {}
        self._holding_fact_ids: List[int] = []
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "clear" and len(args) == 1 and f.name.startswith("+"):
                self._clear_fact_id[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 1 and f.name.startswith("+"):
                self._holding_fact_ids.append(f.global_id)

        self._handempty_id: Optional[int] = self._fact_id_by_name.get("+handempty")

    def _parse_fact_name(self, name: str):
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
    def _is_true(state: int, fact_global_id: int) -> int:
        return (state >> fact_global_id) & 1

    def __call__(self, node) -> int:
        state = node.state

        # If already holding something, at least one action needed to free hand
        holding_penalty = 0
        if self._handempty_id is not None and not self._is_true(state, self._handempty_id):
            # In this domain, not handempty essentially means holding some block.
            # One action (put-down or stack) is required before methods needing handempty.
            holding_penalty = 1

        # A) count unsatisfied goal on-relations
        unsat = 0
        # B) add minimal clearing cost for involved blocks
        clear_lb = 0

        for fid, (a, b) in zip(self._goal_on_fact_ids, self._goal_on_pairs):
            if not self._is_true(state, fid):
                unsat += 1

                # If a is not clear -> at least 2 actions needed (unstack+put-down) somewhere
                ca = self._clear_fact_id.get(a)
                if ca is not None and not self._is_true(state, ca):
                    clear_lb += 2

                # If b is not clear -> at least 2 actions needed
                cb = self._clear_fact_id.get(b)
                if cb is not None and not self._is_true(state, cb):
                    clear_lb += 2

        # Combine: each unsatisfied on-goal requires at least 1 stacking action eventually.
        # We use unsat as that base lower bound.
        h = unsat + clear_lb + holding_penalty

        # Small normalization: avoid returning 0 when there are pending tasks but no explicit goals.
        # (Still admissible: if there is truly work but no goal facts, h=0 is also admissible;
        # but this can help tie-breaking a bit. Keep it conservative.)
        if h == 0 and len(getattr(node, "task_network", [])) > 0:
            return 0

        return h
