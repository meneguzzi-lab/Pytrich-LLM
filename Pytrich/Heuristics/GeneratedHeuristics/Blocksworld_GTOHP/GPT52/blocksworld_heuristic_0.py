from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic0(Heuristic):
    """Domain-dependent heuristic for the HDDL Blocksworld domain (Pytrich).

    Rationale (domain insights -> heuristic terms)
    --------------------------------------------
    This domain has 4 abstract tasks used in training instances:
      - do_put_on(x,y): if already on(x,y) => nop, else must clear x and y, ensure y on table, then move x onto y.
      - do_on_table(x): if clear(x) => nop, else unstack top block and put it down.
      - do_move(x,y): if x on table => pick-up+stack (2 actions), else unstack+stack (2 actions).
      - do_clear(x): if clear(x) => nop, else recursively clear the block on top, then unstack+put-down.

    Key planning difficulty is *clearing*: to achieve on(x,y) you must free x and y from anything above them.
    Also, in this domain the only way to place x onto y is via stack, requiring holding(x) and clear(y).

    Heuristic design goals:
      - Fast on 1000 blocks: avoid state_explicit_repr(state) and any O(#facts) per call scans.
      - Use bitwise checks via fact.global_id.
      - Provide a useful lower bound on remaining primitive actions.

    Heuristic definition (admissible lower bound)
    --------------------------------------------
    We estimate remaining cost as a sum of three *lower-bounding* components:

      (A) Unsatisfied goal on-relations count.
          Each missing on(x,y) must be achieved by at least one stack action => >= 1 primitive.

      (B) Minimal "clearance" actions needed for all involved blocks.
          If a block b is not clear, at least one unstack is required to remove the top block from b.
          In this domain, whenever unstack occurs during clearing, it is immediately followed by put-down
          by the do_clear method (m7) => >= 2 primitives per blocked block.
          We count 2 per distinct non-clear block among all goal-relevant blocks.

      (C) Holding / handempty readiness.
          If hand is not empty (i.e., holding some block), at least one put-down is needed before methods
          requiring handempty (most methods) can proceed. This adds >= 1.

    We deliberately avoid double-counting towers precisely (expensive) and instead use conservative
    lower bounds, keeping admissibility.

    Notes:
      - This heuristic is *state-based* and ignores the current HTN task network content because Pytrich
        calls heuristics frequently and the largest instances have huge task networks; however, goal-driven
        clearance still strongly correlates with remaining HTN effort in this benchmark.
      - All checks are bitwise using cached fact ids.

    Interface
    ---------
    Pytrich expects a Heuristic with __call__(state, task_network) -> number.

    """

    def __init__(self, model, blocksworld_gpt52_heuristic_0: bool = True):
        super().__init__(model)
        self.blocksworld_gpt52_heuristic_0 = blocksworld_gpt52_heuristic_0

        # Cache: predicate -> tuple(arg_count, list of (fact_global_id, args_tuple))
        self._pred_facts: Dict[str, List[Tuple[int, Tuple[str, ...]]]] = {}

        # Useful fact ids for fast membership checks
        self._fact_ids_clear: Dict[str, int] = {}
        self._fact_ids_handempty: Optional[int] = None

        # Goal on(x,y) pairs extracted from model.goals
        self._goal_on_pairs: List[Tuple[str, str, int]] = []  # (x,y,fact_id)
        self._goal_relevant_blocks: Set[str] = set()

        self._build_fact_indices()
        self._extract_goals()

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

    def _build_fact_indices(self) -> None:
        # Build indices once; O(#facts) one-time cost.
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._pred_facts.setdefault(pred, []).append((f.global_id, tuple(args)))
            if pred == "clear" and len(args) == 1:
                self._fact_ids_clear[args[0]] = f.global_id
            elif pred == "handempty" and len(args) == 0:
                self._fact_ids_handempty = f.global_id

    def _extract_goals(self) -> None:
        # model.goals is bitwise int.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gf in goal_fact_names:
            pred, args = self._parse_fact_name(gf)
            if pred == "on" and len(args) == 2:
                x, y = args[0], args[1]
                # Find the corresponding fact global_id by scanning all 'on' facts list.
                # This is done once.
                fid = None
                for cand_id, cand_args in self._pred_facts.get("on", []):
                    if cand_args == (x, y):
                        fid = cand_id
                        break
                if fid is not None:
                    self._goal_on_pairs.append((x, y, fid))
                    self._goal_relevant_blocks.add(x)
                    self._goal_relevant_blocks.add(y)

    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def __call__(self, state: int, task_network=None) -> int:
        # If already a goal state (including empty TN), return 0 quickly.
        # (model.goal_reached checks both goal facts and empty TN; but we keep it cheap here.)
        if (self.model.goals & state) == self.model.goals:
            # Might still have remaining TN; heuristic is about refinement cost, so keep 0 only when TN empty.
            if not task_network:
                return 0

        h = 0

        # (A) Each missing goal on(x,y) needs at least one stack.
        missing_on = 0
        for x, y, fid in self._goal_on_pairs:
            if not self._is_true(state, fid):
                missing_on += 1
        h += missing_on

        # (B) Clearance lower bound: for each goal-relevant block that is not clear, at least
        # one unstack+put-down pair will be needed somewhere to clear it => >= 2.
        blocked = 0
        for b in self._goal_relevant_blocks:
            fid_clear = self._fact_ids_clear.get(b)
            if fid_clear is None:
                continue
            if not self._is_true(state, fid_clear):
                blocked += 1
        h += 2 * blocked

        # (C) If not handempty, at least one put-down is required before many methods apply.
        if self._fact_ids_handempty is not None:
            if not self._is_true(state, self._fact_ids_handempty):
                h += 1

        return h
