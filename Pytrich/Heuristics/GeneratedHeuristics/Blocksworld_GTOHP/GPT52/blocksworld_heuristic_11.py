from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

# Pytrich imports (the surrounding project provides these at runtime)
from Pytrich.Heuristics.heuristic import Heuristic
from Pytrich.model import Model


class BlocksworldGPT52Heuristic11(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain.

    Design rationale (domain insights -> heuristic terms)
    -----------------------------------------------
    The domain encodes high-level tasks that decompose into classic Blocksworld
    primitives:
      - do_put_on(x,y) typically requires: clear(x), clear(y), ensure y on table,
        then move x onto y.
      - do_on_table(x) might require unstack(x, z) then put-down(x).
      - do_clear(x) recursively clears blocks above x.

    Key planning difficulty: blocks must be cleared (possibly recursively) before
    they can be moved/stacked. This is the main source of cost and branching.

    We build a fast, informative *lower bound* by combining:
      (A) Goal mismatch count for goal facts (on / ontable)
      (B) Required clearing effort for the *supports* in goals: to achieve on(a,b)
          you must eventually have clear(a) and clear(b).

    Important: The heuristic is designed to be computationally cheap for up to
    1000 blocks, so it avoids expensive state-to-string conversions.

    Admissibility/consistency note
    ------------------------------
    This heuristic is intended as a practical, domain-guiding heuristic.
    It aims to correlate strongly with remaining work, but it is not guaranteed
    admissible in the strict theoretical sense because it mixes several counts
    that can overlap. In the Pytrich setting, this kind of heuristic is commonly
    used to reduce node expansions.

    Implementation notes
    --------------------
    - Facts are checked via bitwise operations using Fact.global_id.
    - Goal facts are obtained once from model.goals using state_explicit_repr.
    - We precompute mappings from fact ids to structured (predicate,args).
    - We also precompute adjacency "what is on top of x" and a fast way to
      compute "how many blocks above x" from the current state.

    The heuristic value is:
        h = w_goal * (#unsatisfied goal facts)
            + w_clear * (sum of blocks above each required-clear block)

    where required-clear blocks are: every x and y appearing in goal on(x,y),
    plus every x in goal ontable(x).

    We additionally add a small penalty when hand is not empty and a placement
    goal is still unsatisfied, which empirically helps on some instances.
    """

    def __init__(self, model: Model, blocksworld_gpt52_heuristic_11: Optional[dict] = None):
        super().__init__(model, blocksworld_gpt52_heuristic_11)
        self.model: Model = model

        cfg = blocksworld_gpt52_heuristic_11 or {}
        self.w_goal = float(cfg.get("w_goal", 3.0))
        self.w_clear = float(cfg.get("w_clear", 1.0))
        self.w_hand = float(cfg.get("w_hand", 0.5))

        # --- Precompute fact parsing and fast access tables ---
        self._fact_pred: List[Optional[str]] = [None] * len(self.model.facts)
        self._fact_args: List[Tuple[str, ...]] = [tuple()] * len(self.model.facts)

        # Global id lookup for key unary facts
        self._handempty_fact_id: Optional[int] = None

        # For current-state reconstruction of on-relations
        # Store ids of +on[x,y] facts and their (x,y)
        self._on_fact_ids: List[int] = []
        self._on_pairs: List[Tuple[str, str]] = []

        # Goal processing
        self._goal_fact_ids: List[int] = []
        self._goal_on: List[Tuple[str, str]] = []
        self._goal_ontable: List[str] = []

        self._init_fact_tables()
        self._init_goal_tables()

        # Blocks that must be clear eventually due to goals
        self._required_clear_blocks: List[str] = self._compute_required_clear_blocks()

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

    def _init_fact_tables(self) -> None:
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            self._fact_pred[f.global_id] = pred
            self._fact_args[f.global_id] = tuple(args)

            if pred == "handempty" and len(args) == 0 and f.name.startswith("+"):
                self._handempty_fact_id = f.global_id

            if pred == "on" and len(args) == 2 and f.name.startswith("+"):
                self._on_fact_ids.append(f.global_id)
                self._on_pairs.append((args[0], args[1]))

    def _init_goal_tables(self) -> None:
        # model.goals is a bitwise integer
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)

        # Map goal fact names back to global ids by scanning facts once using a name->id dict
        # (This is done once; runtime heuristic calls never do name-based state inspection.)
        name_to_gid: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        for gname in goal_fact_names:
            gid = name_to_gid.get(gname)
            if gid is None:
                continue
            self._goal_fact_ids.append(gid)
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self._goal_on.append((args[0], args[1]))
            elif pred == "ontable" and len(args) == 1:
                self._goal_ontable.append(args[0])

    def _compute_required_clear_blocks(self) -> List[str]:
        req = set()
        for x, y in self._goal_on:
            req.add(x)
            req.add(y)
        for x in self._goal_ontable:
            req.add(x)
        return list(req)

    def _count_unsatisfied_goals(self, state: int) -> int:
        # Cheap count using goal fact ids and bit checks
        unsat = 0
        for gid in self._goal_fact_ids:
            if ((state >> gid) & 1) == 0:
                unsat += 1
        return unsat

    def _blocks_above_counts(self, state: int) -> Dict[str, int]:
        """Return dict block->number_of_blocks_above_it in the current state.

        We reconstruct the on-graph from true +on facts:
            on[top] = below
        Then for each base, walk upward via inverse adjacency to count.

        Complexity: O(n) to build adjacency from on facts + O(n) to compute counts.
        Works well for up to 1000 blocks.
        """
        # below_of[top] = below
        below_of: Dict[str, str] = {}
        # above_of[below] = top
        above_of: Dict[str, str] = {}

        for gid, (top, below) in zip(self._on_fact_ids, self._on_pairs):
            if ((state >> gid) & 1) == 0:
                continue
            below_of[top] = below
            above_of[below] = top

        # For each block that appears as a below, compute chain length upward.
        # Use memoization to keep linear time.
        memo: Dict[str, int] = {}

        def count_above(b: str) -> int:
            if b in memo:
                return memo[b]
            top = above_of.get(b)
            if top is None:
                memo[b] = 0
                return 0
            memo[b] = 1 + count_above(top)
            return memo[b]

        # Only compute for needed blocks on demand later; return memo accessor via dict.
        # But to keep it simple, compute for all keys in above_of and their bases.
        for b in list(above_of.keys()):
            count_above(b)
        return memo

    def __call__(self, state: int, task_network: List[Union[object, object]] = None) -> float:
        # (1) count unsatisfied goal facts
        unsat = self._count_unsatisfied_goals(state)

        if unsat == 0:
            # If goals satisfied, heuristic should be 0 even if TN non-empty; search will handle.
            return 0.0

        # (2) estimate clearing effort for goal-relevant blocks
        above_counts = self._blocks_above_counts(state)
        clear_effort = 0
        for b in self._required_clear_blocks:
            clear_effort += above_counts.get(b, 0)

        # (3) small hand-not-empty penalty if there is still goal work
        hand_pen = 0.0
        if self._handempty_fact_id is not None:
            if ((state >> self._handempty_fact_id) & 1) == 0:
                hand_pen = self.w_hand

        return self.w_goal * unsat + self.w_clear * clear_effort + hand_pen
