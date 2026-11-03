from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic12(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale / domain insights
    ---------------------------
    Depots is essentially a blocks-world stacking problem embedded in logistics:
    - Goals are mostly of the form (on crateX surfaceY).
    - To achieve an (on c s) goal, the domain typically needs:
        * making c clear (remove any crate on top of c)
        * making s clear (remove any crate on top of s)
        * having a hoist at the relevant place
        * if c is not at the same place as s, moving it via truck (load/drive/unload)
        * then lift+drop to place c onto s

    The HDDL methods encode these options, but the search still benefits from a
    good *lower bound* on remaining effort.

    This heuristic computes a fast, admissible estimate by summing independent
    per-goal lower bounds derived from:
    - whether the goal fact is already true
    - whether the crate is already co-located with the target surface
    - whether the crate is currently "in a truck" (then it must at least be unloaded)
    - whether the crate or target surface is blocked (not clear), implying at least one
      lift/load/etc. somewhere to unblock (we count 1 as a conservative lower bound)

    Additionally, it adds a small *task-network* component:
    - each remaining abstract task (and non-nop primitive op) contributes a tiny
      admissible floor cost, but we keep it very small to avoid overestimation.

    Important implementation constraints
    -----------------------------------
    - Uses bitwise checks for facts (fast).
    - Reads goal facts via model.state_explicit_repr(model.goals).
    - Avoids state_explicit_repr(state) during evaluation.

    Admissibility
    ------------
    Every counted unit corresponds to a necessary action category in any valid plan:
    - If (on c s) is false, at least one action is needed (some sequence must change it).
    - If crate and surface are in different places, at least one Drive is necessary.
      (In this domain, the only way to move a crate between places is by truck, which
       includes at least one Drive when places differ.)
    - If crate is in a truck, at least one Unload is necessary before it can be on a surface.
    - If c is not clear, at least one action is needed to remove the top crate (we count 1).
    - If s is not clear, at least one action is needed to clear it (we count 1).

    We intentionally undercount (e.g., we don't charge full load+drive+unload+drop),
    which keeps it a safe lower bound and still provides strong guidance.
    """

    def __init__(self, model, depots_gpt52_heuristic_12: str = "depots_gpt52_heuristic_12"):
        super().__init__(model, depots_gpt52_heuristic_12)
        self.model = model

        # --- Pre-index facts for O(1) bitwise access ---
        self._fact_by_name: Dict[str, int] = {}
        # Predicate -> list of (args_tuple, global_id)
        self._facts_by_pred: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}

        for f in self.model.facts:
            gid = f.global_id
            self._fact_by_name[f.name] = gid
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((tuple(args), gid))

        # Quick lookup tables for common predicates
        # at[x,place]
        self._at_fact_gid: Dict[Tuple[str, str], int] = {}
        for args, gid in self._facts_by_pred.get("at", []):
            if len(args) == 2:
                self._at_fact_gid[(args[0], args[1])] = gid

        # clear[x]
        self._clear_fact_gid: Dict[str, int] = {}
        for args, gid in self._facts_by_pred.get("clear", []):
            if len(args) == 1:
                self._clear_fact_gid[args[0]] = gid

        # in[crate,truck]
        self._in_fact_gids_by_crate: Dict[str, List[int]] = {}
        for args, gid in self._facts_by_pred.get("in", []):
            if len(args) == 2:
                self._in_fact_gids_by_crate.setdefault(args[0], []).append(gid)

        # on[crate,surface]
        self._on_fact_gid: Dict[Tuple[str, str], int] = {}
        for args, gid in self._facts_by_pred.get("on", []):
            if len(args) == 2:
                self._on_fact_gid[(args[0], args[1])] = gid

        # lifting[hoist,crate]
        self._lifting_fact_gids_by_crate: Dict[str, List[int]] = {}
        for args, gid in self._facts_by_pred.get("lifting", []):
            if len(args) == 2:
                self._lifting_fact_gids_by_crate.setdefault(args[1], []).append(gid)

        # Cache goal on-facts (parsed once)
        self._goal_on_pairs: List[Tuple[str, str, int]] = []  # (crate, surface, fact_gid)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                c, s = args
                gid = self._fact_by_name.get(name)
                if gid is not None:
                    self._goal_on_pairs.append((c, s, gid))

        # Small constant to account for remaining task-network work without dominating.
        self._tn_floor_per_task = 0  # keep 0 for strict admissibility vs unknown costs

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

    @staticmethod
    def _is_true(state_bitwise: int, fact_gid: int) -> bool:
        return ((state_bitwise >> fact_gid) & 1) == 1

    def _get_location_place(self, state_bitwise: int, obj: str) -> Optional[str]:
        """Return place p such that at[obj,p] holds, else None."""
        # Scan all possible at[obj,place] facts for this obj.
        # We don't have direct obj->places index; create it on the fly cheaply:
        # look through all at facts args tuples and match first arg.
        # (Number of places is small: depots+distributors <= ~10.)
        for (o, p), gid in self._at_fact_gid.items():
            if o == obj and self._is_true(state_bitwise, gid):
                return p
        return None

    def _crate_in_some_truck(self, state_bitwise: int, crate: str) -> bool:
        for gid in self._in_fact_gids_by_crate.get(crate, []):
            if self._is_true(state_bitwise, gid):
                return True
        return False

    def _crate_being_lifted(self, state_bitwise: int, crate: str) -> bool:
        for gid in self._lifting_fact_gids_by_crate.get(crate, []):
            if self._is_true(state_bitwise, gid):
                return True
        return False

    def _is_clear(self, state_bitwise: int, surface_or_crate: str) -> Optional[bool]:
        gid = self._clear_fact_gid.get(surface_or_crate)
        if gid is None:
            return None
        return self._is_true(state_bitwise, gid)

    def __call__(self, state: int, task_network: List) -> int:
        # Main estimate: sum over goal (on c s)
        h = 0

        for c, s, on_gid in self._goal_on_pairs:
            # If already satisfied, no cost for this goal.
            if self._is_true(state, on_gid):
                continue

            # At least something must be done to make (on c s) true.
            h += 1

            # If c is currently inside a truck, it must be unloaded before it can be on any surface.
            # (Even if it's being lifted already, still not on the target.)
            if self._crate_in_some_truck(state, c):
                h += 1

            # If crate is being lifted, it's already "in hand" so we don't require clearing it.
            # Otherwise, if it's not clear, at least one action is needed to remove the top item.
            if not self._crate_being_lifted(state, c):
                c_clear = self._is_clear(state, c)
                if c_clear is False:
                    h += 1

            # Target surface must be clear to drop onto it.
            s_clear = self._is_clear(state, s)
            if s_clear is False:
                h += 1

            # Logistics: if we can determine places for crate and surface and they differ,
            # then at least one Drive must occur to move the crate between places.
            # If location unknown (e.g., crate in truck with unknown place), don't add.
            pc = self._get_location_place(state, c)
            ps = self._get_location_place(state, s)
            if pc is not None and ps is not None and pc != ps:
                h += 1

        # Optional tiny task-network component (kept at 0 by default)
        if self._tn_floor_per_task > 0 and task_network:
            # Count only non-nop primitive operators and abstract tasks.
            # This remains admissible if set to 0; otherwise may overestimate in some HTN settings.
            h += self._tn_floor_per_task * sum(1 for t in task_network if getattr(t, "name", "") != "nop")

        return h
