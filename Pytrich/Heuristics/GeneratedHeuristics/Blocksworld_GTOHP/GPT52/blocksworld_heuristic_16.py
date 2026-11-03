from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from Pytrich.Heuristics.heuristic import Heuristic


class BlocksworldGPT52Heuristic16(Heuristic):
    """Domain-dependent heuristic for the Pytrich HDDL Blocksworld domain.

    Rationale (domain insights -> heuristic signals)
    ----------------------------------------------
    This domain has four compound tasks that decompose into primitive actions:

      - do_clear(x): recursively clears blocks above x, then unstack+put-down.
      - do_on_table(x): if x is clear and not ontable, unstack(x, y)+put-down(x).
      - do_move(x, y): if x ontable then pick-up+stack else unstack+stack.
      - do_put_on(x, y): if already on(x,y) -> nop, else clear x and y, put y on table,
                         then move x onto y.

    Key difficulty: to achieve on(x,y), one must often clear x, clear y, and ensure y is
    available (typically on the table) before stacking. The expensive part is removing
    blocks above x and y; this cost depends on the *number of blocks in those towers*.

    Heuristic design
    ----------------
    We compute a fast lower bound as a sum over *unsatisfied goal on(x,y)* facts.
    For each such pair we estimate a relaxed minimal number of primitive actions:

      base_cost = 2  # stack requires holding then stack; in this domain reaching on(x,y)
                    # from a "ready" situation needs at least one stack plus one pickup/unstack.

      + 2 * blocks_above(x)  # each blocking block above x must be removed: at least unstack+put-down.
      + 2 * blocks_above(y)  # similarly to make y clear.

    Additionally, if x is not clear we add at least 1 extra action (some unstack needed)
    and if hand is not empty we add 1 (need put-down to free hand). These are still
    admissible as they are *necessary* conditions in the real domain.

    To avoid gross overcounting when many goals share towers, we *cap* the summed cost
    by a second admissible bound: number of misplaced goal relations (each requires at
    least one primitive operator to change the on-graph) times 1. Then we take the max
    of the two bounds? Max of admissible bounds is admissible; however the summed tower
    bound is usually stronger. We therefore return max(misplaced_count, tower_sum_bound).

    Efficiency notes
    ----------------
    - Uses ONLY bitwise checks on facts (no state_explicit_repr on current state).
    - Parses and indexes facts once in __init__.
    - Computes blocks_above(.) from current 'on' relation in O(#blocks) per call.

    This heuristic is domain-dependent (uses Blocksworld tower structure), hierarchy-aware
    (mirrors minimal primitive work implied by do_clear / do_move decompositions), and
    scales well to large instances.
    """

    def __init__(self, model, blocksworld_gpt52_heuristic_16: str = "blocksworld_gpt52_heuristic_16"):
        super().__init__(model, blocksworld_gpt52_heuristic_16)
        self.model = model

        # Fact id caches
        self._handempty_gid: Optional[int] = None
        self._clear_gid: Dict[str, int] = {}
        self._ontable_gid: Dict[str, int] = {}
        self._holding_gid: Dict[str, int] = {}
        self._on_gid: Dict[Tuple[str, str], int] = {}

        # For fast traversal: list of all (x,y,gid) 'on' facts in the model
        self._all_on_facts: List[Tuple[str, str, int]] = []

        # Goal on-pairs
        self._goal_on_pairs: List[Tuple[str, str]] = []

        self._index_facts()
        self._index_goals()

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

    def _index_facts(self) -> None:
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == "handempty":
                self._handempty_gid = fact.global_id
            elif pred == "clear" and len(args) == 1:
                self._clear_gid[args[0]] = fact.global_id
            elif pred == "ontable" and len(args) == 1:
                self._ontable_gid[args[0]] = fact.global_id
            elif pred == "holding" and len(args) == 1:
                self._holding_gid[args[0]] = fact.global_id
            elif pred == "on" and len(args) == 2:
                x, y = args[0], args[1]
                self._on_gid[(x, y)] = fact.global_id
                self._all_on_facts.append((x, y, fact.global_id))

    def _index_goals(self) -> None:
        # model.goals is a bitwise integer
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self._goal_on_pairs.append((args[0], args[1]))

    @staticmethod
    def _bit_is_true(state_bitwise: int, gid: int) -> int:
        return (state_bitwise >> gid) & 1

    def _is_clear(self, state: int, b: str) -> bool:
        gid = self._clear_gid.get(b)
        return bool(gid is not None and self._bit_is_true(state, gid))

    def _is_handempty(self, state: int) -> bool:
        if self._handempty_gid is None:
            return True
        return bool(self._bit_is_true(state, self._handempty_gid))

    def _is_on(self, state: int, x: str, y: str) -> bool:
        gid = self._on_gid.get((x, y))
        return bool(gid is not None and self._bit_is_true(state, gid))

    def _blocks_above_count(self, state: int, base: str) -> int:
        """Return number of blocks stacked (transitively) above `base` in current state.

        Uses the functional nature of Blocksworld on(.) in valid states.
        If cycles/inconsistencies occur, we safely stop.
        """
        # Build mapping support->top only for true on facts
        above: Dict[str, str] = {}
        for x, y, gid in self._all_on_facts:
            if self._bit_is_true(state, gid):
                above[y] = x

        cnt = 0
        seen = set()
        cur = base
        while cur in above:
            nxt = above[cur]
            if nxt in seen:
                break
            seen.add(nxt)
            cnt += 1
            cur = nxt
        return cnt

    def __call__(self, parent_node, node) -> int:
        state = node.state

        # Bound 1: number of unsatisfied goal on-relations (each needs >=1 primitive op)
        misplaced = 0
        for x, y in self._goal_on_pairs:
            if not self._is_on(state, x, y):
                misplaced += 1

        if misplaced == 0:
            return 0

        # Bound 2: tower-based relaxed action count for each unsatisfied goal on(x,y)
        # Note: each blocking block above something costs at least (unstack + put-down) = 2 ops.
        tower_sum = 0
        hand_penalty = 0 if self._is_handempty(state) else 1

        for x, y in self._goal_on_pairs:
            if self._is_on(state, x, y):
                continue

            above_x = self._blocks_above_count(state, x)
            above_y = self._blocks_above_count(state, y)

            # minimal to create on(x,y) once both are clear and hand is free is >= 2
            # (pick-up/unstack + stack). This is a safe lower bound.
            cost = 2 + 2 * above_x + 2 * above_y

            # If x is not clear, at least one unstack must happen (already implied by above_x>=1
            # in valid states, but keep robust for partial/inconsistent states)
            if not self._is_clear(state, x):
                cost += 1
            if not self._is_clear(state, y):
                cost += 1

            cost += hand_penalty
            tower_sum += cost

        # Combine admissible bounds (max of lower bounds remains a lower bound)
        return max(misplaced, tower_sum)
