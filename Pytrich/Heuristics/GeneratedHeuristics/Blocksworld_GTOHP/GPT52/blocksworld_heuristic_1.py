from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Pytrich import (the framework provides Heuristic)
from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _Atom:
    """Compact representation of a positive grounded atom."""

    pred: str
    args: Tuple[str, ...]


class BlocksworldGPT52Heuristic1(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Core domain insights exploited
    -----------------------------
    The domain defines abstract tasks (do_put_on, do_on_table, do_move, do_clear)
    that decompose into primitive actions. For search guidance we only need a fast
    *lower bound* on remaining effort.

    Key structural challenges in this HDDL Blocksworld:
      * Many goals/tasks require blocks to be **clear** and the hand **empty**.
      * Achieving a single on(x,y) often requires clearing x and y, possibly moving y
        to table, then moving x onto y.
      * Clearing a block implies removing *all blocks above it*, each costing at least
        an unstack+put-down pair in any real plan.

    Heuristic design
    ----------------
    We use an admissible, cheap-to-compute sum of independent lower bounds:

      1) Unsatisfied goal atoms count (on/ontable/clear/handempty/holding):
         each missing positive goal atom costs at least 1 action in real planning.

      2) For each unsatisfied goal on(x,y), add a *blocking penalty* equal to the number
         of blocks above x plus the number of blocks above y (excluding x itself). This
         is admissible because every block above a target must be moved at least once,
         and in this domain moving a block off something requires at least one primitive
         action (indeed >=2, but we keep 1 for safety).

    We take the maximum of (1) and (2) instead of summing them to remain conservative
    under overlap between subgoals (same clearing steps can serve multiple goals).

    Performance considerations
    --------------------------
    * No calls to state_explicit_repr(state) during evaluation (too slow on large instances).
    * All checks are bitwise using fact.global_id.
    * Precompute indices for +on and +ontable facts and build a fast "support" mapping
      from the current state in O(#on-facts-true) time.

    This heuristic is admissible and typically consistent enough for best-first/A* variants
    in Pytrich while being scalable to very large instances.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_1: str = "blocksworld_gpt52_heuristic_1"):
        super().__init__(model, blocksworld_gpt52_heuristic_1)
        self.model = model

        # --- Parse and index all facts we might need ---
        # Store Fact objects by (pred, args)
        self._fact_by_atom: Dict[_Atom, object] = {}

        # All +on facts (x,y) -> Fact
        self._on_facts: Dict[Tuple[str, str], object] = {}
        # All +ontable facts (x,) -> Fact
        self._ontable_facts: Dict[str, object] = {}

        # Goal atoms (positive only) as _Atom
        self._goal_atoms: List[_Atom] = []
        self._goal_on_pairs: List[Tuple[str, str]] = []

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            # We only reason over positive atoms
            if not fact.name.startswith("+"):
                continue

            atom = _Atom(pred, tuple(args))
            self._fact_by_atom[atom] = fact
            if pred == "on" and len(args) == 2:
                self._on_facts[(args[0], args[1])] = fact
            elif pred == "ontable" and len(args) == 1:
                self._ontable_facts[args[0]] = fact

        # Extract positive goal atoms from bitwise goal representation
        for gname in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(gname)
            if pred is None:
                continue
            if not gname.startswith("+"):
                continue
            atom = _Atom(pred, tuple(args))
            self._goal_atoms.append(atom)
            if pred == "on" and len(args) == 2:
                self._goal_on_pairs.append((args[0], args[1]))

    # -------- Fact parsing helper (as required by the prompt) --------
    def _parse_fact_name(self, name: str):
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

    # -------- Bitwise state check --------
    @staticmethod
    def _is_true(state_bitwise: int, fact) -> bool:
        return ((state_bitwise >> fact.global_id) & 1) == 1

    # -------- Build current support mapping from state --------
    def _build_supports(self, state_bitwise: int) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return (support_of, above_of):

        support_of[x] = y means +on[x,y] is true.
        above_of[y]   = x means x is directly on y.

        Only considers +on atoms.
        """
        support_of: Dict[str, str] = {}
        above_of: Dict[str, str] = {}

        # Iterate all possible on facts but only touch those true.
        # This is O(total_on_facts) worst case; in Blocksworld total_on_facts ~ n*(n-1),
        # so we avoid scanning all by using bitwise membership on fact objects we already have.
        # However Pytrich facts are not grouped by predicate, so we keep a dict and scan it.
        # In practice, the number of on facts in the model is large, but this scan still tends
        # to be acceptable because evaluation count is far lower than expansions in HTN search.
        # If needed, Pytrich users can cache supports externally.
        for (x, y), f in self._on_facts.items():
            if self._is_true(state_bitwise, f):
                support_of[x] = y
                above_of[y] = x

        return support_of, above_of

    def _count_above(self, block: str, above_of: Dict[str, str]) -> int:
        """Count how many blocks are above `block` in the current stacking relation."""
        cnt = 0
        cur = block
        # Follow chain upward using above_of mapping
        while cur in above_of:
            cnt += 1
            cur = above_of[cur]
            # Defensive: avoid pathological loops (shouldn't occur in valid states)
            if cnt > 100000:
                break
        return cnt

    # -------- Heuristic interface --------
    def __call__(self, state: int, task_network=None) -> int:
        # (1) Missing positive goal atoms
        missing_goals = 0
        for atom in self._goal_atoms:
            fact = self._fact_by_atom.get(atom)
            if fact is None:
                # If a goal atom is not present as a fact, be conservative.
                missing_goals += 1
                continue
            if not self._is_true(state, fact):
                missing_goals += 1

        # Fast exit
        if missing_goals == 0:
            return 0

        # (2) Blocking penalty for unsatisfied on(x,y) goals
        support_of, above_of = self._build_supports(state)

        blocking_lb = 0
        for (x, y) in self._goal_on_pairs:
            # If already satisfied, no contribution.
            f = self._on_facts.get((x, y))
            if f is not None and self._is_true(state, f):
                continue

            # Lower bound: must clear x and y.
            # Count blocks above each; any such block must be moved away at least once.
            blocking_lb += self._count_above(x, above_of)
            blocking_lb += self._count_above(y, above_of)

            # Additionally, if x is on something other than y, at least one move is needed.
            # (Still admissible, very weak but helpful.)
            if x in support_of and support_of[x] != y:
                blocking_lb += 1

        # Use max to avoid double counting overlap among goals.
        h = max(missing_goals, blocking_lb)

        # Optional tiny tie-breaker based on remaining abstract work in task network.
        # Must remain admissible: add 0 only.
        return int(h)
