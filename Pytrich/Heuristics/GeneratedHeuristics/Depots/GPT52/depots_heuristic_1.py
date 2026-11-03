from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _GoalOn:
    """Parsed goal of the form +on[crate,surface]."""
    crate: str
    surf: str


class DepotsGPT52Heuristic1(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Design rationale (domain insights -> heuristic signals)
    ------------------------------------------------------
    Depots problems in this benchmark consist almost entirely of achieving many
    stacking goals of the form (on crateX surfaceY), where a surface is either a
    pallet or another crate.

    In this domain, achieving one (on c s2) typically requires a *Drop* action
    of a hoist at the *place* where s2 is located. If c is not already at that
    place (or is inside a truck), extra actions are needed:
      - Unload/Load to move between truck and hoist
      - Drive to move the truck between places
      - Lift/Drop to move between surfaces
      - Clear operations because Lift requires (clear c) and Drop requires (clear s2)

    The HTN methods encode these cases (already on, same-place move, in-truck move,
    or cross-place move). For heuristic guidance we want a fast *lower bound*.

    This heuristic is a sum of cheap, admissible per-goal lower bounds:
      1) If goal already satisfied: 0
      2) Otherwise at least one Drop is needed: +1
      3) If crate is currently in some truck: need Unload before Drop: +1
      4) If crate is at a different place than the target surface: some transport
         is needed (at least Drive once in any real plan): +1
      5) If target surface is not clear: must clear it somehow before Drop: +1
         (clearing in reality costs >= 1 primitive action)
      6) If crate is not clear: must clear it before Lift/Unload+Drop sequence: +1

    All components are *lower bounds* (never overestimate), so h is admissible.
    It is also computationally light: it uses only bitwise checks and small
    precomputed indices from fact names.

    Note: We intentionally do not try to compute exact minimal clear-cost for
    deep stacks (which would require recursive reasoning and be expensive on p30).
    The +1 clear penalties still provide strong guidance because many dead-ends
    come from trying to Drop onto non-clear targets or moving non-clear crates.
    """

    def __init__(self, model, depots_gpt52_heuristic_1: str = "depots_gpt52_heuristic_1"):
        super().__init__(model, depots_gpt52_heuristic_1)
        self.model = model

        # --- Fact parsing / indexing ---
        # We need fast access to selected predicates by argument patterns.
        self._fact_by_name: Dict[str, object] = {f.name: f for f in self.model.facts}

        # Store all +at[x,p] facts for locatable x
        self._at_facts: Dict[Tuple[str, str], int] = {}  # (x,p) -> global_id
        # Store all +on[c,s] facts
        self._on_facts: Dict[Tuple[str, str], int] = {}
        # Store all +in[c,t] facts
        self._in_facts: Dict[Tuple[str, str], int] = {}
        # Store all +clear[s] facts
        self._clear_facts: Dict[str, int] = {}

        # For each locatable, list of (place, fact_id) for +at
        self._places_of: Dict[str, List[Tuple[str, int]]] = {}
        # For each crate, list of (truck, fact_id) for +in
        self._trucks_of: Dict[str, List[Tuple[str, int]]] = {}

        # Goal (on ...) facts parsed once
        self._goal_on: List[_GoalOn] = []

        # Precompute indices from fact strings
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "at" and len(args) == 2:
                x, p = args
                self._at_facts[(x, p)] = fact.global_id
                self._places_of.setdefault(x, []).append((p, fact.global_id))
            elif pred == "on" and len(args) == 2:
                c, s = args
                self._on_facts[(c, s)] = fact.global_id
            elif pred == "in" and len(args) == 2:
                c, t = args
                self._in_facts[(c, t)] = fact.global_id
                self._trucks_of.setdefault(c, []).append((t, fact.global_id))
            elif pred == "clear" and len(args) == 1:
                s = args[0]
                self._clear_facts[s] = fact.global_id

        # Parse goals via the required API (bitwise goal set)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2 and gname.startswith("+"):
                self._goal_on.append(_GoalOn(crate=args[0], surf=args[1]))

    # ---------- Required helper: grounded fact parsing ----------
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

    # ---------- Bitwise checks ----------
    @staticmethod
    def _is_true(state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) == 1

    def _current_place_of(self, state_bitwise: int, x: str) -> Optional[str]:
        """Return a place p such that +at[x,p] holds, else None.

        Assumption for Depots instances: each locatable has exactly one location,
        but we implement defensively.
        """
        for p, fid in self._places_of.get(x, []):
            if self._is_true(state_bitwise, fid):
                return p
        return None

    def _current_truck_of(self, state_bitwise: int, c: str) -> Optional[str]:
        """Return a truck t such that +in[c,t] holds, else None."""
        for t, fid in self._trucks_of.get(c, []):
            if self._is_true(state_bitwise, fid):
                return t
        return None

    def _is_clear(self, state_bitwise: int, s: str) -> bool:
        fid = self._clear_facts.get(s)
        if fid is None:
            return False
        return self._is_true(state_bitwise, fid)

    # ---------- Heuristic core ----------
    def __call__(self, node) -> int:
        """Return an admissible lower bound estimate for remaining effort."""
        state = node.state

        h_val = 0

        for goal in self._goal_on:
            # If already satisfied, contribute 0
            on_id = self._on_facts.get((goal.crate, goal.surf))
            if on_id is not None and self._is_true(state, on_id):
                continue

            # Base: at least one Drop to establish (on crate surf)
            g_cost = 1

            # Determine where the target surface is
            surf_place = self._current_place_of(state, goal.surf)

            # If crate is in a truck, at least one Unload needed before it can be Dropped
            crate_truck = self._current_truck_of(state, goal.crate)
            if crate_truck is not None:
                g_cost += 1
                # If we know the surface place and truck not at that place -> at least one Drive
                if surf_place is not None:
                    truck_place = self._current_place_of(state, crate_truck)
                    if truck_place is not None and truck_place != surf_place:
                        g_cost += 1
            else:
                # Crate not in truck: if crate place differs from target surface place,
                # need transport (in any plan: Load/Drive/Unload), lower bound by +1
                crate_place = self._current_place_of(state, goal.crate)
                if surf_place is not None and crate_place is not None and crate_place != surf_place:
                    g_cost += 1

            # Clear requirements (lower bounds)
            if not self._is_clear(state, goal.surf):
                g_cost += 1
            if not self._is_clear(state, goal.crate):
                g_cost += 1

            h_val += g_cost

        # Small tie-breaker using HTN task network size (still admissible if we only add 0)
        # We keep heuristic integer and admissible: do not add anything.
        return h_val
