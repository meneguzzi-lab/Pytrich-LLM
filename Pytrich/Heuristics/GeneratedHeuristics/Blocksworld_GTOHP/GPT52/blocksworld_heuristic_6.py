from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic6(Heuristic):
    """Domain-dependent heuristic for the Pytrich HDDL Blocksworld domain.

    Key domain insight
    ------------------
    The domain’s abstract tasks deterministically refine into the classic 4-action
    Blocksworld operators (pick-up, put-down, unstack, stack), but wrapped in HTN
    tasks:

      * do_clear(x): may recursively clear blocks above x
      * do_on_table(x): if x not on table, must unstack x and put-down
      * do_move(x,y): either pick-up+stack (if x ontable) or unstack+stack
      * do_put_on(x,y): if not already on(x,y), must clear x and y, ensure y on table,
                       then move x onto y

    In large instances, most search difficulty comes from choosing decompositions
    whose preconditions are *almost* satisfied but blocked by “clear” constraints.
    Therefore the heuristic focuses on a *cheap lower bound* derived from:

      (A) unsatisfied goal facts (on/ontable)
      (B) whether required blocks are clear (if not, at least one unstack is needed)
      (C) whether the hand is holding something (extra put-down needed)

    The heuristic remains admissible by only counting actions that are
    unavoidably required in any valid plan (lower bound).

    Efficiency
    ----------
    * Uses bitwise checks only (no state_explicit_repr on current states).
    * Precomputes fact ids for on/ontable/clear/holding/handempty.
    * Evaluates goal facts from model.goals once in __init__.

    Admissibility rationale (sketch)
    -------------------------------
    For each unsatisfied goal +on[x,y], at minimum some action must add on[x,y].
    Only 'stack(x,y)' adds on[x,y], so count >= 1.

    Additionally, if y is not clear in the current state, at least one unstack
    must occur at some point before stack(x,y) can be applied (because clear(y)
    is a positive precondition of stack). So we can add +1 if not clear(y).
    Similarly, if x is not clear, +1.

    If the goal includes +ontable[x] and it is currently false, at least one
    action must add it; only 'put-down(x)' adds ontable(x), so count >= 1.
    If x is not clear currently, at least one unstack is needed before x can be
    held and put down, so +1.

    Finally, if the agent is holding some block, at least one put-down/stack
    must occur before reaching a complete plan (handempty is needed by most
    methods/actions); adding +1 is a safe lower bound.

    Note: This heuristic ignores negative interactions and double-counting; that
    is fine because we only need a lower bound.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_6: str = "blocksworld_gpt52_heuristic_6"):
        super().__init__(model, blocksworld_gpt52_heuristic_6)
        self.model = model

        # --- predicate -> fact-id caches ---
        # on_ids[x][y] = global_id
        self.on_ids: Dict[str, Dict[str, int]] = {}
        self.ontable_ids: Dict[str, int] = {}
        self.clear_ids: Dict[str, int] = {}
        self.holding_ids: Dict[str, int] = {}
        self.handempty_id: Optional[int] = None

        # Parse all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == "on" and len(args) == 2:
                x, y = args
                self.on_ids.setdefault(x, {})[y] = f.global_id
            elif pred == "ontable" and len(args) == 1:
                self.ontable_ids[args[0]] = f.global_id
            elif pred == "clear" and len(args) == 1:
                self.clear_ids[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 1:
                self.holding_ids[args[0]] = f.global_id
            elif pred == "handempty" and len(args) == 0:
                self.handempty_id = f.global_id

        # --- cache goal facts as parsed atoms ---
        self.goal_on: List[Tuple[str, str]] = []
        self.goal_ontable: List[str] = []
        self.goal_clear: List[str] = []
        self.goal_handempty: bool = False
        self.goal_holding: List[str] = []

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            if pred == "on" and len(args) == 2:
                self.goal_on.append((args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self.goal_ontable.append(args[0])
            elif pred == "clear" and len(args) == 1:
                self.goal_clear.append(args[0])
            elif pred == "handempty" and len(args) == 0:
                self.goal_handempty = True
            elif pred == "holding" and len(args) == 1:
                self.goal_holding.append(args[0])

    # ------------------------
    # Helpers
    # ------------------------
    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]  # Remove +/- prefix
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _any_holding(self, state_bitwise: int) -> bool:
        # Iterate holding fact ids (number of blocks); still cheap.
        for fid in self.holding_ids.values():
            if ((state_bitwise >> fid) & 1) == 1:
                return True
        return False

    # ------------------------
    # Heuristic interface
    # ------------------------
    def __call__(self, parent_node, node) -> int:
        """Compute an admissible, domain-informed lower bound."""
        state = node.state

        h = 0

        # If holding something, at least one action is needed to make progress
        # toward most HTN methods (handempty required) and toward a finished plan.
        if self._any_holding(state):
            h += 1

        # --- goal: on(x,y) ---
        # Lower bound:
        #   +1 if on(x,y) not currently true (only stack adds it)
        #   +1 if x not clear (need at least one unstack somewhere)
        #   +1 if y not clear (need at least one unstack somewhere)
        for x, y in self.goal_on:
            on_xy_id = self.on_ids.get(x, {}).get(y)
            if not self._is_true(state, on_xy_id):
                h += 1

                clear_x_id = self.clear_ids.get(x)
                clear_y_id = self.clear_ids.get(y)
                if not self._is_true(state, clear_x_id):
                    h += 1
                if not self._is_true(state, clear_y_id):
                    h += 1

        # --- goal: ontable(x) ---
        # Lower bound:
        #   +1 if ontable(x) false (only put-down adds it)
        #   +1 if x not clear (need at least one unstack to access it)
        for x in self.goal_ontable:
            ontable_x_id = self.ontable_ids.get(x)
            if not self._is_true(state, ontable_x_id):
                h += 1
                clear_x_id = self.clear_ids.get(x)
                if not self._is_true(state, clear_x_id):
                    h += 1

        # --- goal: clear(x) ---
        # If clear(x) is a goal and currently false, at least one unstack must occur.
        for x in self.goal_clear:
            clear_x_id = self.clear_ids.get(x)
            if not self._is_true(state, clear_x_id):
                h += 1

        # --- goal: handempty ---
        # If required and false, at least one put-down/stack is required.
        if self.goal_handempty and self.handempty_id is not None:
            if not self._is_true(state, self.handempty_id):
                h += 1

        # --- goal: holding(x) ---
        # If holding(x) required and false, at least one pick-up/unstack must occur.
        # (We avoid further conditions to keep it admissible and cheap.)
        for x in self.goal_holding:
            hid = self.holding_ids.get(x)
            if not self._is_true(state, hid):
                h += 1

        return h
