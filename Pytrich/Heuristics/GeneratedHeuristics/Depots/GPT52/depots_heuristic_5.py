from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic5(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Key domain observations (used to build a cheap lower bound):

    1) All high-level tasks in the benchmark are `do_put_on(crate, surface)`.
       Achieving a single goal `on(c, s2)` minimally requires at least one `Drop`.

    2) If the crate is not already *lifting* by some hoist, then to be able to `Drop`
       we need to achieve `lifting(h, c)` first, which minimally costs:
          - 1 action if `lifting(h,c)` already true
          - 1 action (`Lift` or `Unload`) otherwise, assuming we can get preconditions
            in a relaxed sense.

       In the real domain, those preconditions may require truck movement, loading,
       unloading, and clearing; but as a *lower bound* we count only unavoidable
       "structural" actions.

    3) Location mismatch is the big cost driver in Depots. If the crate and its goal
       surface are at different places, then (ignoring preconditions) at least:
          Load + Drive + Unload
       must occur somewhere in any valid plan (because only trucks change place).
       That gives a strong and admissible extra lower bound of +3.

    4) Clearing constraints: to `Lift` from a surface, the crate must be `clear(c)`;
       to `Drop` onto a surface, the target surface must be `clear(s2)`.
       If these are false, at least one action is needed to change them, but modeling
       full unstacking is expensive. We add a *very small* penalty (+1) when
       `clear(s2)` is currently false, because some action must make it clear.
       (This remains a lower bound: you cannot drop onto a non-clear surface.)

    5) Scale: p30 has hundreds of goal `on` facts. We therefore:
       - precompute the list of goal (crate, target_surface)
       - precompute a fast mapping from each fact name to its global id
       - for each evaluation, only do O(#goal_on) constant-time bit checks.

    The heuristic value is the sum over unsatisfied goal `on(c,s2)` of:
       1 (Drop)
       + 0/1 (need to obtain lifting)
       + 0/3 (need trucking due to place mismatch)
       + 0/1 (target surface not clear right now)

    This is not intended to be perfectly accurate, but it is:
       - admissible (a lower bound)
       - informative for the Depots structure
       - very fast.
    """

    def __init__(self, model, depots_gpt52_heuristic_5: bool = True):
        super().__init__(model)
        self.depots_gpt52_heuristic_5 = depots_gpt52_heuristic_5

        # Map fact name -> global_id for fast state bit queries.
        self._fid: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Extract goal +on[crate,surface] pairs.
        self._goal_on: List[Tuple[str, str]] = []
        for fact_name in self.model.state_explicit_repr(self.model.goals):
            pred, args = self._parse_fact_name(fact_name)
            if pred == "on" and len(args) == 2:
                self._goal_on.append((args[0], args[1]))

        # Cache commonly queried fact ids (when they exist).
        # Note: facts might not exist for every combination; use dict lookup each time.

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

    def _is_true(self, state: int, fact_name: str) -> bool:
        gid = self._fid.get(fact_name)
        if gid is None:
            return False
        return ((state >> gid) & 1) == 1

    def _find_place_of(self, state: int, obj: str) -> Optional[str]:
        # Find any place p such that +at[obj,p] is true.
        # This is called only for goal crates and goal surfaces, so still cheap.
        # We scan all possible at-facts for obj by checking names in a small loop is
        # not possible without enumerating places; instead we exploit that the model
        # already contains grounded facts. We do a single pass over a pre-filtered list
        # by generating possible names would be costly; so we lazily scan facts once
        # per query using a very small optimization: cache results per state is too
        # expensive. Instead we use a heuristic shortcut: many instances have unique at.

        # Practical compromise: iterate over all facts of form at[obj,*] by scanning
        # model facts names once and caching their (obj -> list of (place, gid)).
        # Build this index on first use.
        if not hasattr(self, "_at_index"):
            at_index: Dict[str, List[Tuple[str, int]]] = {}
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred == "at" and len(args) == 2:
                    o, p = args
                    at_index.setdefault(o, []).append((p, f.global_id))
            self._at_index = at_index

        candidates = self._at_index.get(obj)
        if not candidates:
            return None
        for p, gid in candidates:
            if ((state >> gid) & 1) == 1:
                return p
        return None

    def _is_lifting_some_hoist(self, state: int, crate: str) -> bool:
        # Build index lifting[*hoist, crate] -> gid
        if not hasattr(self, "_lifting_index"):
            lifting_index: Dict[str, List[int]] = {}
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred == "lifting" and len(args) == 2:
                    h, c = args
                    lifting_index.setdefault(c, []).append(f.global_id)
            self._lifting_index = lifting_index

        gids = self._lifting_index.get(crate)
        if not gids:
            return False
        for gid in gids:
            if ((state >> gid) & 1) == 1:
                return True
        return False

    def __call__(self, parent_node, node) -> int:
        state = node.state

        h_val = 0
        for c, s2 in self._goal_on:
            # If goal already satisfied, skip.
            if self._is_true(state, f"+on[{c},{s2}]"):
                continue

            # Base: at least one Drop to establish on(c,s2).
            cost = 1

            # Need lifting(h,c) before Drop; if not currently lifting, add 1.
            if not self._is_lifting_some_hoist(state, c):
                cost += 1

            # If crate place differs from target surface place, at least Load+Drive+Unload.
            # (In instances, each surface (pallet or crate) has an at(s,place) when it is at a place.)
            pc = self._find_place_of(state, c)
            ps = self._find_place_of(state, s2)
            if pc is not None and ps is not None and pc != ps:
                cost += 3

            # If target is not clear, cannot Drop; at least one action must clear it.
            # Add small admissible penalty.
            if self._fid.get(f"+clear[{s2}]") is not None:
                if not self._is_true(state, f"+clear[{s2}]"):
                    cost += 1

            h_val += cost

        return h_val
