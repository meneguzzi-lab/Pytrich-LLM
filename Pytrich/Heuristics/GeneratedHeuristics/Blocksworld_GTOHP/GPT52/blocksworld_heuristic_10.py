from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union

# Pytrich provides Heuristic base class in its framework.
# The import path can vary by version; the try/except keeps the file robust.
try:
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    try:
        from Pytrich.Heuristics.Heuristic import Heuristic
    except Exception:
        # Fallback for some installations
        from Pytrich.heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic10(Heuristic):
    """Domain-dependent heuristic for HDDL Blocksworld in Pytrich.

    Core idea (cheap lower bound, domain-informed):

    In this domain, the HTN tasks ultimately enforce a set of (on x y) relations.
    Achieving a single missing (on x y) requires at least:
      - 1 stack action (to create +on[x,y])
      - plus possibly an unstack and/or pick-up to get x into hand
      - plus possibly clearing x and y (by unstacking blocks above them)

    We combine two admissible components:

    (A) Unsatisfied goal facts count:
        h_goal = number of goal facts not yet true.
        This is admissible because each missing goal (on x y) needs at least one
        primitive action that adds it (a stack).

    (B) Mandatory clearing lower bound for involved blocks:
        For each block that appears as x or y in an unsatisfied goal (on x y),
        if it is not clear in the current state, at least one unstack is required
        somewhere to clear it (because only unstack can make the below block clear).
        We count distinct involved blocks that are not clear:
          h_clear = |{b involved in unsatisfied on-goals : clear(b) is false}|
        This is admissible as a set-union lower bound: each unstack can clear
        at most one specific blocked block at a time.

    Final heuristic is:
        h = max(h_goal, h_clear)

    Why this helps in HTN Blocksworld:
      - Large instances contain many do_put_on tasks; counting unsatisfied on-goals
        gives strong guidance with O(#goals) time.
      - The dominant difficulty is clearing targets; h_clear differentiates states
        where required blocks are buried under others, while staying cheap.

    Implementation notes (Pytrich-specific, performance-critical):
      - Never build explicit state representations for the current state.
      - Use bitwise checks via fact.global_id.
      - Goal facts are obtained once from model.goals (bitwise int) via
        model.state_explicit_repr(model.goals).

    The heuristic is admissible and consistent enough for A*/greedy usage in practice.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_10: bool = True):
        super().__init__(model)
        self.blocksworld_gpt52_heuristic_10 = blocksworld_gpt52_heuristic_10

        # Build fast lookup: fact name -> global_id
        self._fact_gid: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Parse goal facts once (explicit repr is OK here; it's a one-time cost).
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)

        # Store only positive on-goals and (optionally) other goals if present.
        self._goal_on_pairs: List[Tuple[str, str]] = []
        self._goal_fact_gids: List[int] = []

        for gf in goal_fact_names:
            gid = self._fact_gid.get(gf, None)
            if gid is not None:
                self._goal_fact_gids.append(gid)

            pred, args = self._parse_fact_name(gf)
            if pred == "on" and len(args) == 2 and gf.startswith("+"):
                self._goal_on_pairs.append((args[0], args[1]))

        # Precompute clear fact ids for all blocks encountered in goals.
        self._clear_gid: Dict[str, int] = {}
        blocks_in_on_goals = set()
        for x, y in self._goal_on_pairs:
            blocks_in_on_goals.add(x)
            blocks_in_on_goals.add(y)
        for b in blocks_in_on_goals:
            name = f"+clear[{b}]"
            gid = self._fact_gid.get(name)
            if gid is not None:
                self._clear_gid[b] = gid

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]  # Remove +/- prefix
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, fact_gid: int) -> int:
        return (state_bitwise >> fact_gid) & 1

    def __call__(self, node) -> int:
        """Compute heuristic value for a search node.

        Expected node interface in Pytrich: node.state is a bitwise int.
        """
        state = node.state

        # Component (A): count unsatisfied goal facts (cheap, general).
        # If goals include other predicates, they also contribute.
        unsat_goal = 0
        for gid in self._goal_fact_gids:
            if ((state >> gid) & 1) == 0:
                unsat_goal += 1

        if unsat_goal == 0:
            # Goals satisfied in state-space sense; HTN may still have pending tasks,
            # but as a lower bound on remaining primitive effort, 0 is safe.
            return 0

        # Component (B): lower bound on required clearing among blocks involved in
        # *unsatisfied* on-goals.
        blocked_involved = set()
        for x, y in self._goal_on_pairs:
            on_gid = self._fact_gid.get(f"+on[{x},{y}]")
            if on_gid is None:
                continue
            if ((state >> on_gid) & 1) == 1:
                continue  # already satisfied

            # If x is not clear, it must be cleared by some unstack.
            gx = self._clear_gid.get(x)
            if gx is not None and ((state >> gx) & 1) == 0:
                blocked_involved.add(x)

            # If y is not clear, it must be cleared by some unstack.
            gy = self._clear_gid.get(y)
            if gy is not None and ((state >> gy) & 1) == 0:
                blocked_involved.add(y)

        h_clear = len(blocked_involved)

        # Admissible combination: max of admissible lower bounds.
        h = unsat_goal if unsat_goal >= h_clear else h_clear

        # Optional tiny tie-breaker (still admissible if it never increases h): none.
        return h
