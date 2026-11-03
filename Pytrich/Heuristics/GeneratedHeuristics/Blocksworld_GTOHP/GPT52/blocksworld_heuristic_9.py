from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

# Pytrich provides Heuristic base class in its framework.
# Import path can differ by installation; keep it minimal/robust.
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from pytrich.heu.heuristic import Heuristic  # type: ignore
    except Exception:
        from Heuristics.heuristic import Heuristic  # type: ignore


class BlocksworldGPT52Heuristic9(Heuristic):
    """Domain-dependent heuristic for the BLOCKS (Blocksworld) HDDL domain.

    Design rationale (domain insights):
    - The domain is essentially classical Blocksworld with HTN wrappers:
      * do_put_on(x,y) decomposes (when not already on) into:
          do_clear(x), do_clear(y), do_on_table(y), do_move(x,y)
      * do_clear(x) recursively removes blocks above x (unstack+put-down).
      * do_on_table(x) either nop if already clear, else unstack+put-down.
      * do_move(x,y) is either pick-up+stack if x on table, or unstack+stack.

    Key planning challenge:
    - To achieve an on(x,y) relation, both x and y must be clear, and x must be held
      at some point. Clearing requires recursively removing blocks above.

    Heuristic idea:
    - Provide a fast admissible lower bound by counting *necessary* primitive actions
      implied by unsatisfied goal facts, using only positive facts and ignoring
      negative interactions (a relaxation).

    What we count (lower bounds):
    1) For each goal +on(x,y) not currently true:
       - At least 1 STACK action is required to create on(x,y).
       - Additionally, to be able to stack x onto y, x and y must become clear.
         If a block is not clear now, at least one UNSTACK is needed to remove
         the top block above it (in reality more may be needed, but 1 is a LB).
       - To move x at all, if x is not currently holding, at least 1 action is
         needed to pick it up or unstack it (PICK-UP or UNSTACK). We count 1.

    2) For each goal +ontable(x) not currently true (rare in provided training but supported):
       - At least 1 PUT-DOWN is required.
       - If x is not clear, at least 1 UNSTACK is needed.
       - If x is not already holding, at least 1 PICK-UP/UNSTACK is needed.

    3) For each goal +clear(x) not currently true:
       - At least 1 UNSTACK is needed to remove something from x.

    Notes:
    - We avoid expensive string-based state inspection. All checks are bitwise using
      fact.global_id.
    - We pre-index facts for O(1) lookup.
    - We deduplicate shared requirements: if multiple goals require clear(x), we
      count the "make clear" lower bound once.

    Admissibility:
    - Each counted unit corresponds to a primitive action that must occur in any
      valid plan (in the standard Blocksworld dynamics). We take care to only count
      *lower bounds* (e.g., 1 unstack even if multiple blocks above).
    - The heuristic may be weak but scales well to 1000 blocks.

    This heuristic is domain-dependent because it interprets Blocksworld-specific
    predicates (on/ontable/clear/holding/handempty) and uses known necessary steps
    (stack, unstack, pick-up, put-down).
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_9: str = "blocksworld_gpt52_heuristic_9"):
        super().__init__(model, blocksworld_gpt52_heuristic_9)
        self.model = model

        # Fact name -> Fact object for quick global_id access
        self._fact_by_name: Dict[str, object] = {}
        for f in self.model.facts:
            self._fact_by_name[f.name] = f

        # Cache goal facts (explicit names), then parse into structures.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_on: List[Tuple[str, str]] = []
        self._goal_ontable: List[str] = []
        self._goal_clear: List[str] = []
        self._goal_holding: List[str] = []
        self._goal_handempty: bool = False

        for nm in goal_fact_names:
            pred, args = self._parse_fact_name(nm)
            if pred is None:
                continue
            if pred == "on" and len(args) == 2:
                self._goal_on.append((args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self._goal_ontable.append(args[0])
            elif pred == "clear" and len(args) == 1:
                self._goal_clear.append(args[0])
            elif pred == "holding" and len(args) == 1:
                self._goal_holding.append(args[0])
            elif pred == "handempty":
                self._goal_handempty = True

        # Prebuild fact ids for common queries; missing facts are tolerated.
        self._clear_fact_id: Dict[str, int] = {}
        self._ontable_fact_id: Dict[str, int] = {}
        self._holding_fact_id: Dict[str, int] = {}
        self._on_fact_id: Dict[Tuple[str, str], int] = {}
        self._handempty_fact_id: Optional[int] = None

        for name, fact in self._fact_by_name.items():
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            gid = fact.global_id
            if pred == "clear" and len(args) == 1:
                self._clear_fact_id[args[0]] = gid
            elif pred == "ontable" and len(args) == 1:
                self._ontable_fact_id[args[0]] = gid
            elif pred == "holding" and len(args) == 1:
                self._holding_fact_id[args[0]] = gid
            elif pred == "on" and len(args) == 2:
                self._on_fact_id[(args[0], args[1])] = gid
            elif pred == "handempty" and len(args) == 0:
                self._handempty_fact_id = gid

        # For "not clear" detection we need to know if something is on top of x.
        # Build mapping y -> list of x such that on(x,y) is a possible fact.
        # This is domain-level adjacency derived from grounded facts, not from state.
        self._possible_above: Dict[str, List[str]] = {}
        for (x, y) in self._on_fact_id.keys():
            self._possible_above.setdefault(y, []).append(x)

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

    def _is_true_gid(self, state: int, gid: Optional[int]) -> bool:
        if gid is None or gid < 0:
            return False
        return ((state >> gid) & 1) == 1

    def _is_clear(self, state: int, b: str) -> bool:
        gid = self._clear_fact_id.get(b)
        return self._is_true_gid(state, gid)

    def _is_holding(self, state: int, b: str) -> bool:
        gid = self._holding_fact_id.get(b)
        return self._is_true_gid(state, gid)

    def _is_ontable(self, state: int, b: str) -> bool:
        gid = self._ontable_fact_id.get(b)
        return self._is_true_gid(state, gid)

    def _is_on(self, state: int, x: str, y: str) -> bool:
        gid = self._on_fact_id.get((x, y))
        return self._is_true_gid(state, gid)

    def __call__(self, state: int, task_network=None) -> int:
        # If goals already reached (classical part), return 0.
        # Note: HTN goal test also checks empty TN; heuristic should be safe regardless.
        if (self.model.goals & state) == self.model.goals:
            return 0

        h = 0

        # Deduplicate "need clear" and "need move" requirements across goals.
        need_clear: Set[str] = set()
        need_move: Set[str] = set()  # blocks that must be manipulated (picked/unstacked) at least once
        need_putdown: Set[str] = set()

        # ------------- Handle on-goals -------------
        for (x, y) in self._goal_on:
            if self._is_on(state, x, y):
                continue

            # To achieve on(x,y), at least one stack is necessary.
            h += 1

            # x and y must be clear at time of stacking.
            need_clear.add(x)
            need_clear.add(y)

            # x must be moved/held at least once.
            need_move.add(x)

        # ------------- Handle ontable goals -------------
        for x in self._goal_ontable:
            if self._is_ontable(state, x):
                continue
            # To get ontable(x), at least one put-down is necessary.
            need_putdown.add(x)
            need_clear.add(x)
            need_move.add(x)

        # ------------- Handle clear goals -------------
        for x in self._goal_clear:
            if self._is_clear(state, x):
                continue
            need_clear.add(x)

        # ------------- Handle holding goals (rare) -------------
        for x in self._goal_holding:
            if self._is_holding(state, x):
                continue
            # To hold x, must do a pick-up or unstack: at least 1 action.
            h += 1
            need_clear.add(x)

        # ------------- Lower bound for making blocks clear -------------
        # If clear(x) is not true now, at least one unstack must happen (remove the top block).
        # (Even if multiple blocks above, we still count only 1 => admissible.)
        for b in need_clear:
            if not self._is_clear(state, b):
                h += 1

        # ------------- Lower bound for moving blocks at least once -------------
        # If we need to manipulate x and we are not already holding it, at least one action
        # is needed to obtain holding(x): either pick-up or unstack.
        for b in need_move:
            if not self._is_holding(state, b):
                h += 1

        # ------------- Lower bound for put-down on table -------------
        # If ontable(x) is required and x is not already ontable, we already reserved 1 stack?
        # No, for ontable we reserved a put-down. If already holding, put-down is still needed.
        for b in need_putdown:
            if not self._is_ontable(state, b):
                h += 1

        # ------------- Optional: handempty goal -------------
        # If handempty is a goal and currently holding something, at least 1 action needed
        # (put-down or stack). We count 1.
        if self._goal_handempty and self._handempty_fact_id is not None:
            if not self._is_true_gid(state, self._handempty_fact_id):
                h += 1

        # Ensure non-negative int.
        return int(max(0, h))
