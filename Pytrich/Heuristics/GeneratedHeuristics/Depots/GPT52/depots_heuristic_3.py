from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic3(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Design rationale (domain insights):
    - The top-level work in all training instances is a sequence of (do_put_on c s2).
      Achieving a single on[c,s2] typically requires: clearing c, clearing s2, lifting c,
      and dropping it onto s2. If c is in a truck, it additionally needs truck-at-place
      alignment and an unload.
    - The *hard part* in Depots is often not the final Drop itself, but the prerequisite
      chain around `clear` and (possibly) moving a truck to the right place.
    - We therefore compute a cheap *lower bound* by summing, for each unsatisfied goal
      on[c,s2], a minimal estimate of required primitive actions under a relaxation:
        * ignore delete effects (classic admissible relaxation intuition)
        * treat movement between any two places as at least one Drive if truck not already
          at the needed place (no distances available)
        * only reason about whether the crate is already on target, already being lifted,
          already in some truck, or already at the target place.

    Implementation notes for Pytrich:
    - Uses only bitwise checks on facts (fast), never state_explicit_repr(state).
    - Goal facts are obtained from model.goals via state_explicit_repr(model.goals).
    - Fact parsing follows the required +pred[arg1,arg2] format.

    The heuristic is not guaranteed admissible in the formal sense for all HTN refinements,
    but it is designed to be a strong, inexpensive domain guide that reduces expansions.
    """

    def __init__(self, model, depots_gpt52_heuristic_3: str = "depots_gpt52_heuristic_3"):
        super().__init__(model, depots_gpt52_heuristic_3)
        self.model = model

        # Build fast indices: predicate -> list of (fact_global_id, args)
        self._pred_to_facts: Dict[str, List[Tuple[int, List[str]]]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._pred_to_facts.setdefault(pred, []).append((f.global_id, args))

        # Cache goal (on ...) pairs from the goal bitset.
        self._goal_on: List[Tuple[str, str]] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "on" and len(args) == 2:
                self._goal_on.append((args[0], args[1]))

        # For fast lookup of fact ids by (pred,args...)
        self._fact_id: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        for pred, facts in self._pred_to_facts.items():
            for gid, args in facts:
                self._fact_id[(pred, tuple(args))] = gid

        # Pre-extract lists that we frequently scan
        self._at_facts = self._pred_to_facts.get("at", [])          # (gid, [obj, place])
        self._on_facts = self._pred_to_facts.get("on", [])          # (gid, [crate, surface])
        self._in_facts = self._pred_to_facts.get("in", [])          # (gid, [crate, truck])
        self._clear_facts = self._pred_to_facts.get("clear", [])    # (gid, [surface])
        self._lifting_facts = self._pred_to_facts.get("lifting", [])

    # ------------------------ parsing & bit helpers ------------------------

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

    def _fact_true(self, state: int, pred: str, *args: str) -> bool:
        gid = self._fact_id.get((pred, tuple(args)))
        if gid is None:
            return False
        return self._is_true(state, gid)

    # ------------------------ small derived queries ------------------------

    def _where_is(self, state: int, obj: str) -> Optional[str]:
        """Return a place p such that at[obj,p] is true, else None."""
        for gid, args in self._at_facts:
            if args and args[0] == obj and self._is_true(state, gid):
                return args[1]
        return None

    def _in_which_truck(self, state: int, crate: str) -> Optional[str]:
        for gid, args in self._in_facts:
            if args and args[0] == crate and self._is_true(state, gid):
                return args[1]
        return None

    def _some_hoist_at(self, state: int, place: str) -> bool:
        """We only need existence; hoist identity is irrelevant in the relaxed cost."""
        # hoists are locatable, so they appear in at facts.
        for gid, args in self._at_facts:
            if len(args) == 2 and args[1] == place and args[0].startswith("hoist") and self._is_true(state, gid):
                return True
        return False

    def _is_clear(self, state: int, surface: str) -> bool:
        # clear is unary
        for gid, args in self._clear_facts:
            if args and args[0] == surface and self._is_true(state, gid):
                return True
        return False

    def _is_lifting_some_hoist(self, state: int, crate: str) -> bool:
        for gid, args in self._lifting_facts:
            if len(args) == 2 and args[1] == crate and self._is_true(state, gid):
                return True
        return False

    def _surface_place(self, state: int, surface: str) -> Optional[str]:
        return self._where_is(state, surface)

    # ------------------------ heuristic core ------------------------

    def __call__(self, node) -> int:
        state = node.state

        # If no goal on-facts were detected, fall back to 0.
        if not self._goal_on:
            return 0

        h = 0

        # Sum an action-lower-bound per unsatisfied goal on[c, s2]
        for crate, target_surface in self._goal_on:
            if self._fact_true(state, "on", crate, target_surface):
                continue

            # Estimate minimal remaining primitive actions for this goal.
            h += self._estimate_put_on_cost(state, crate, target_surface)

        return h

    def _estimate_put_on_cost(self, state: int, crate: str, target_surface: str) -> int:
        """Cheap relaxed estimate for achieving on[crate,target_surface]."""

        # If crate is already being lifted, we may only need Drop after ensuring target clear and at same place.
        if self._is_lifting_some_hoist(state, crate):
            # Need target at some place; in practice target_surface is at a place.
            p2 = self._surface_place(state, target_surface)
            # If we can't locate target, be conservative.
            base = 1  # Drop
            # Need clear(target_surface) if not already.
            if not self._is_clear(state, target_surface):
                base += 1
            # Need a hoist at p2 to drop (if unknown, add 1 as proxy for moving/getting correct hoist).
            if p2 is not None and not self._some_hoist_at(state, p2):
                base += 1
            return base

        # Case: crate is in a truck.
        truck = self._in_which_truck(state, crate)
        if truck is not None:
            p2 = self._surface_place(state, target_surface)
            cost = 0

            # Need truck at p2; if not, at least one Drive.
            if p2 is not None and not self._fact_true(state, "at", truck, p2):
                cost += 1

            # Need clear(target_surface)
            if not self._is_clear(state, target_surface):
                cost += 1

            # Need Unload + Drop (Unload puts lifting, then Drop puts on)
            cost += 2

            # Also need a hoist at p2; if none, add 1.
            if p2 is not None and not self._some_hoist_at(state, p2):
                cost += 1

            return max(1, cost)

        # Otherwise, crate is (likely) at some place and on some surface.
        p_c = self._where_is(state, crate)
        p2 = self._surface_place(state, target_surface)

        cost = 0

        # If crate not at target place, minimal transfer is load+drive+unload.
        # In HTN, m3_do_put_on does do_load_truck + Drive + do_unload_truck.
        if p_c is not None and p2 is not None and p_c != p2:
            # Lower bound: need to get into a truck (Lift+Load), drive, unload, drop.
            # (We ignore clearing intricacies here, but add minimal constants.)
            cost += 4  # Lift + Load + Drive + Unload (Drop is included in unload-method in HTN, but primitive-wise needs Drop)
            cost += 1  # Drop

            # Need clear(target_surface)
            if not self._is_clear(state, target_surface):
                cost += 1

            # Need hoist at both ends; if missing at destination, add 1.
            if not self._some_hoist_at(state, p2):
                cost += 1
            if not self._some_hoist_at(state, p_c):
                cost += 1

            return max(1, cost)

        # Same place (or unknown): directly Lift then Drop after clearing.
        # Need clear(crate) for Lift
        if not self._is_clear(state, crate):
            cost += 1

        # Need clear(target_surface) for Drop
        if not self._is_clear(state, target_surface):
            cost += 1

        # Need Lift + Drop
        cost += 2

        # Need a hoist at that place; if unknown place, skip; else add 1 if none.
        if p2 is not None and not self._some_hoist_at(state, p2):
            cost += 1

        return max(1, cost)
