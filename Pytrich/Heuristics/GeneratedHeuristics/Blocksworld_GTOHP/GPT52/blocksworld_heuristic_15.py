from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Pytrich framework base class
from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic15(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Design rationale (domain insights)
    -------------------------------
    The domain contains four abstract tasks that compile down to standard Blocksworld actions:

      - do_clear(x): ensures clear(x). If x is not clear and handempty, recursively clears
        the block on top and unstack+put-down it.
      - do_on_table(x): if x is clear but not ontable, it must be unstacked from its support
        then put down.
      - do_move(x,y): move x onto y, via either pick-up+stack (if x ontable) or unstack+stack.
      - do_put_on(x,y): either nop if already on(x,y), else clear x and y, ensure y is ontable,
        then do_move(x,y).

    Key difficulty: for a target relation on(x,y), we often must:
      (1) clear x (remove a whole pile above x),
      (2) clear y (remove a whole pile above y),
      (3) ensure y is ontable (possibly unstack y then put-down),
      (4) move x onto y (unstack/pick-up then stack).

    This heuristic computes a fast *lower bound* on remaining primitive actions by counting
    unavoidable work in a relaxed but still domain-shaped way:

      - For each goal fact +on(x,y):
          * If already satisfied: 0.
          * Otherwise, estimate:
                cost_to_clear(x) + cost_to_clear(y) + cost_to_make_ontable(y) + cost_to_place_on(x,y)

    The crucial domain-dependent part is how we estimate clearing/ontable costs:
      - In any legal Blocks state, each block has at most one block directly on top.
        So to clear a block b, you must remove the chain above it; each removal minimally
        requires at least 2 actions (unstack, put-down) in this domain.
      - To make y on the table when it is not ontable, you must at least unstack y from its
        support (1 action) and put it down (1 action) => 2 actions.
      - To place x onto y when goal not satisfied:
          * if holding(x): at least stack => 1
          * else if ontable(x) and clear(x) and handempty: pick-up+stack => 2
          * else: you must at least unstack x from its support (1) then stack (1) => 2
        (We take a conservative lower bound among these cases.)

    Aggregation / admissibility
    ---------------------------
    We sum these per-goal estimates. This is *not guaranteed admissible* in the strict sense
    because different goals can share the same clearing actions (double counting). However,
    it is very informative and typically reduces node expansions substantially.

    To remain robust and fast on very large instances, we:
      - use only bitwise checks on facts (no explicit state expansion),
      - pre-index all relevant facts (on/ontable/clear/handempty/holding) once in __init__,
      - compute "block directly on top of b" and "support of b" by scanning only the true on-facts.

    If you need strict admissibility, replace the sum with a max over goal estimates.
    (That will be admissible but usually weaker.)
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_15: Optional[dict] = None):
        super().__init__(model)
        self.blocksworld_gpt52_heuristic_15 = blocksworld_gpt52_heuristic_15 or {}

        # --- Parse and index facts by predicate ---
        self._on_fact_gid: Dict[Tuple[str, str], int] = {}
        self._ontable_fact_gid: Dict[str, int] = {}
        self._clear_fact_gid: Dict[str, int] = {}
        self._holding_fact_gid: Dict[str, int] = {}
        self._handempty_gid: Optional[int] = None

        # Also store reverse maps for scanning true on-facts cheaply
        self._all_on_pairs: List[Tuple[str, str, int]] = []  # (x,y,gid)

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "on" and len(args) == 2 and f.name.startswith("+"):
                x, y = args[0], args[1]
                self._on_fact_gid[(x, y)] = f.global_id
                self._all_on_pairs.append((x, y, f.global_id))
            elif pred == "ontable" and len(args) == 1 and f.name.startswith("+"):
                self._ontable_fact_gid[args[0]] = f.global_id
            elif pred == "clear" and len(args) == 1 and f.name.startswith("+"):
                self._clear_fact_gid[args[0]] = f.global_id
            elif pred == "holding" and len(args) == 1 and f.name.startswith("+"):
                self._holding_fact_gid[args[0]] = f.global_id
            elif pred == "handempty" and len(args) == 0 and f.name.startswith("+"):
                self._handempty_gid = f.global_id

        # Cache goal on-facts (only +on/2 are relevant in provided instances, but we keep generic)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_on_pairs: List[Tuple[str, str]] = []
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2 and name.startswith("+"):
                self._goal_on_pairs.append((args[0], args[1]))

    # --------- Utility: parsing and bit checks ---------

    def _parse_fact_name(self, name: str):
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

    def _is_true_gid(self, state: int, gid: Optional[int]) -> bool:
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    # --------- Extract local structural info from state ---------

    def _compute_tops_and_supports(self, state: int) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return (top_of, support_of) maps from true on(x,y) facts.

        top_of[y] = x if x is directly on y.
        support_of[x] = y if x is directly on y.

        Computed by scanning all possible on-facts and checking bitwise truth.
        Complexity: O(#on-facts) which is O(n) in Blocksworld (roughly n-#tables).
        """
        top_of: Dict[str, str] = {}
        support_of: Dict[str, str] = {}
        for x, y, gid in self._all_on_pairs:
            if ((state >> gid) & 1) == 1:
                top_of[y] = x
                support_of[x] = y
        return top_of, support_of

    # --------- Domain-dependent cost components ---------

    def _height_above(self, b: str, top_of: Dict[str, str]) -> int:
        """Number of blocks stacked above b (0 if clear)."""
        h = 0
        cur = b
        while cur in top_of:
            h += 1
            cur = top_of[cur]
            # defensive: avoid pathological cycles
            if h > 100000:
                break
        return h

    def _cost_to_clear(self, b: str, state: int, top_of: Dict[str, str]) -> int:
        """Lower-bound cost to make clear(b).

        Each block above b must be removed. In this domain that minimally requires
        (unstack, put-down) = 2 primitive actions per block removed.
        """
        gid_clear = self._clear_fact_gid.get(b)
        if self._is_true_gid(state, gid_clear):
            return 0
        # If not clear, there is some block directly on top; count chain length.
        return 2 * self._height_above(b, top_of)

    def _cost_to_make_ontable(self, y: str, state: int, support_of: Dict[str, str]) -> int:
        """Lower-bound cost to ensure ontable(y).

        If y is not on the table, at least unstack(y, support) + put-down(y) are needed => 2.
        """
        gid_ont = self._ontable_fact_gid.get(y)
        if self._is_true_gid(state, gid_ont):
            return 0
        # If not ontable and it has a support, it must be moved to table.
        if y in support_of:
            return 2
        # If representation is partial/inconsistent, return 0 to stay safe.
        return 0

    def _cost_to_place_on(self, x: str, y: str, state: int, support_of: Dict[str, str]) -> int:
        """Lower-bound cost to achieve on(x,y) once x and y are clear and y is on table.

        Conservative estimate based on current situation:
          - if already holding(x): only stack => 1
          - else if x ontable: need pick-up + stack => 2
          - else if x has support: need unstack + stack => 2
        """
        gid_hold = self._holding_fact_gid.get(x)
        if self._is_true_gid(state, gid_hold):
            return 1
        gid_ont = self._ontable_fact_gid.get(x)
        if self._is_true_gid(state, gid_ont):
            return 2
        if x in support_of:
            return 2
        # Unknown/partial: at least stack is needed once holding, but we don't know.
        return 1

    # --------- Heuristic main entry ---------

    def __call__(self, node) -> int:
        state = node.state

        # Fast path: if no goal on-facts were detected, return 0
        if not self._goal_on_pairs:
            return 0

        top_of, support_of = self._compute_tops_and_supports(state)

        h = 0
        # Sum of shaped per-goal costs
        for x, y in self._goal_on_pairs:
            gid_goal = self._on_fact_gid.get((x, y))
            if gid_goal is not None and ((state >> gid_goal) & 1) == 1:
                continue

            h += self._cost_to_clear(x, state, top_of)
            h += self._cost_to_clear(y, state, top_of)
            h += self._cost_to_make_ontable(y, state, support_of)
            h += self._cost_to_place_on(x, y, state, support_of)

        # Small tie-breaker: encourage handempty when not holding (pure guidance; keep tiny)
        # (If handempty is false and not holding something, state is likely inconsistent; ignore.)
        if self._handempty_gid is not None:
            if ((state >> self._handempty_gid) & 1) == 0:
                # if holding some block, being non-handempty is normal; don't penalize much
                holding_any = False
                for gid in self._holding_fact_gid.values():
                    if ((state >> gid) & 1) == 1:
                        holding_any = True
                        break
                if not holding_any:
                    h += 1

        return int(h)
