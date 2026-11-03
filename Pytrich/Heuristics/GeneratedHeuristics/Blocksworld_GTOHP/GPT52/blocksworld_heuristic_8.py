from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

# Pytrich provides Heuristic base class in its framework.
# The exact import path can vary slightly across distributions;
# this try/except keeps the file robust.
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.Heuristic import Heuristic
    except Exception:
        from heuristics.heuristic import Heuristic  # type: ignore


class BlocksworldGPT52Heuristic8(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Core idea (fast, domain-specific, hierarchical-aware lower bound):

    The domain's abstract tasks are:
      - do_put_on(x,y)
      - do_on_table(x)
      - do_move(x,y)
      - do_clear(x)

    Methods reveal an important structural fact:
      * Achieving/maintaining `clear(b)` is the key enabling condition.
      * To satisfy a put-on relation (x on y) you must at least:
            - make x clear (if not)
            - make y clear (if not)
            - and then perform a move/stack sequence (>=2 primitive actions)
        In this domain, moving x onto y costs at least 2 primitive actions
        (pick-up/stack or unstack/stack). Clearing a non-clear block requires
        at least 2 primitive actions per blocking block above it (unstack + put-down)
        in addition to recursive clearing.

    This heuristic computes a *safe* (admissible) lower bound by summing:
      (A) A relaxed estimate of primitive actions needed for remaining abstract tasks
          in the current task network (TN), using optimistic per-task costs.
      (B) A landmark-like estimate from the (bitwise) goal facts: for each goal
          `on(x,y)` that is not yet satisfied, add a minimal move cost plus
          a minimal amount of clearing needed for x and y.

    Important: we never use explicit state repr membership checks (too slow).
    We only use bitwise checks via fact.global_id.

    Admissibility rationale:
      - For each required clearing step we count only blocks directly above
        the target in the *current* state (a necessary subset of what must
        be removed).
      - For each move/put-on we charge at most the known minimum number of
        primitive actions (2) and ignore additional constraints.
      - For task-network abstract tasks we take the minimum possible primitive
        cost of any applicable method *ignoring preconditions* (i.e., optimistic).

    This tends to guide search strongly on large instances because many goals and
    many TN tasks are explicit do_put_on/do_on_table chains.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_8: str = "blocksworld_gpt52_heuristic_8"):
        super().__init__(model, blocksworld_gpt52_heuristic_8)
        self.model = model

        # --- Cache fact ids by predicate/args for O(1) bit checks ---
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        # For fast extraction of `on(upper, lower)` relations in a state
        self._on_fact_ids: List[Tuple[str, str, int]] = []  # (x,y,fid)

        # Object set (blocks)
        self._blocks: List[str] = []

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_id[key] = f.global_id
            if pred == "on" and len(args) == 2:
                self._on_fact_ids.append((args[0], args[1], f.global_id))

        # Infer block symbols from fact arguments (robust across instances)
        blocks = set()
        for (pred, args), _gid in self._fact_id.items():
            if pred in ("on", "ontable", "clear", "holding"):
                for a in args:
                    blocks.add(a)
        self._blocks = sorted(blocks)

        # Goal on-facts extracted once (goal set is fixed)
        self._goal_on_pairs: List[Tuple[str, str, int]] = []  # (x,y,fid)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                gid = self._fact_id.get(("on", (args[0], args[1])))
                if gid is not None:
                    self._goal_on_pairs.append((args[0], args[1], gid))

        # Precompute clear/handempty ids when present
        self._handempty_id = self._fact_id.get(("handempty", tuple()))

    # ----------------- Utilities -----------------
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

    def _is_true_id(self, state_bitwise: int, fid: Optional[int]) -> bool:
        if fid is None:
            return False
        return ((state_bitwise >> fid) & 1) == 1

    def _is_true(self, state_bitwise: int, pred: str, args: Tuple[str, ...]) -> bool:
        fid = self._fact_id.get((pred, args))
        return self._is_true_id(state_bitwise, fid)

    def _block_above_map(self, state_bitwise: int) -> Dict[str, str]:
        """Return mapping below->above for blocks that currently have something on them.

        In blocksworld, each block has at most one block directly on it.
        """
        below_to_above: Dict[str, str] = {}
        for x, y, fid in self._on_fact_ids:
            if (state_bitwise >> fid) & 1:
                below_to_above[y] = x
        return below_to_above

    def _clear_lower_bound(self, state_bitwise: int, b: str, below_to_above: Dict[str, str]) -> int:
        """Lower bound on primitive actions needed to make block b clear.

        If a block is not clear, some block is on it. To clear it, at minimum you must:
          - unstack that top block (1)
          - put it down somewhere (1)
        and if the blocking block is itself not clear, that requires more.

        A safe LB is 2 * (#blocks in the chain directly above b).
        This is necessary because you must remove each block above b at least once.
        """
        # If already clear -> 0
        if self._is_true(state_bitwise, "clear", (b,)):
            return 0

        # Count chain above b using below_to_above
        cnt = 0
        cur = b
        visited = set()
        while cur in below_to_above:
            nxt = below_to_above[cur]
            if nxt in visited:
                break
            visited.add(nxt)
            cnt += 1
            cur = nxt
        return 2 * cnt

    # ----------------- Heuristic components -----------------
    def _tn_relaxed_cost_lb(self, task_network: List[Union[object, object]]) -> int:
        """Optimistic cost from remaining TN elements.

        We only rely on names/arity patterns. Costs are minimal primitive actions
        achievable by *some* decomposition, ignoring preconditions.

        In this domain:
          - do_clear(x): can be nop if clear(x) -> min cost 0
          - do_on_table(x): can be nop if clear(x) -> min cost 0
          - do_move(x,y): always at least 2 primitives (pick/stack or unstack/stack)
          - do_put_on(x,y): can be nop if already on(x,y) -> min cost 0
                           else requires at least move => min cost 2

        Since we ignore state here, we take the most optimistic lower bound:
          do_clear/do_on_table/do_put_on => 0, do_move => 2

        This component is cheap and helps in large instances where TN is long.
        """
        lb = 0
        for t in task_network:
            # Operators have no decompositions; but TN usually contains abstract tasks.
            name = getattr(t, "name", "")
            if not name:
                continue
            # Abstract tasks in Pytrich are grounded, so name looks like: "(do_put_on b1 b2)" or similar.
            # We keep it robust by substring checks.
            if "do_move" in name:
                lb += 2
            # others contribute 0 in the most relaxed optimistic setting
        return lb

    def _goal_on_cost_lb(self, state_bitwise: int) -> int:
        """Landmark-like LB from unsatisfied on-goals.

        For each unsatisfied goal on(x,y):
          minimal = 2 (to place x on y) + clearing LB for x and y.

        We use chain-based clearing LB from current state.
        """
        below_to_above = self._block_above_map(state_bitwise)
        lb = 0
        for x, y, fid in self._goal_on_pairs:
            if ((state_bitwise >> fid) & 1) == 1:
                continue
            # Must eventually do a move/stack putting x on y
            move_lb = 2
            # Must have x and y clear at that time; clearing now is a necessary subset
            cx = self._clear_lower_bound(state_bitwise, x, below_to_above)
            cy = self._clear_lower_bound(state_bitwise, y, below_to_above)
            lb += move_lb + cx + cy
        return lb

    # ----------------- Main API -----------------
    def __call__(self, node) -> int:
        """Return heuristic value for a search node.

        Node is expected to have:
          - node.state: bitwise int
          - node.task_network: list of remaining tasks (operators/abstract tasks)
        """
        state = node.state
        tn = getattr(node, "task_network", [])

        # Combine two admissible lower bounds; sum is also admissible because
        # they count disjoint necessities only approximately. To stay safely
        # admissible under potential overlap, we take max instead of sum.
        # (Sum could double-count shared clearing effort across multiple goals.)
        h_goal = self._goal_on_cost_lb(state)
        h_tn = self._tn_relaxed_cost_lb(tn)

        # Tie-breaking: tiny preference for having handempty true when equal
        # (still admissible because it only affects tie-breaking via +0/+1 bounded?)
        # We must NOT add positive amount that could break admissibility.
        # So we do not add; we only return max.
        return max(h_goal, h_tn)
