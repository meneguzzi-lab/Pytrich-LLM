from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Any

# Pytrich imports (the framework provides these at runtime)
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic19(Heuristic):
    """Domain-dependent heuristic for the HDDL *Barman* domain (Pytrich).

    Design rationale (domain insights -> heuristic):
    - In this benchmark, top-level goals are usually of the form +contains[shot,cocktail].
      Achieving one cocktail in one shot (from scratch) has a very stereotyped minimal
      primitive sequence:
        * make cocktail in shaker: fill two shots (2x fill-shot), pour both into shaker
          (2x pour-shot-to-*-shaker), shake (1x shake)
        * pour to target shot (1x pour-shaker-to-shot)
      => lower bound ~ 6 primitive actions per missing goal, if we relax cleaning/holding.

    - Cleaning and grasp/drop are frequent sources of extra actions, but they are not
      strictly necessary in the relaxed estimate. However, they matter for *guidance*.
      We therefore add *small* admissible-looking bonuses via lower bounds that still
      cannot exceed the true optimal remaining cost:
        * If the target shot is not empty, you must at least empty-shot once before it
          can contain the cocktail (since actions only add one beverage and require empty).
        * If no hand is empty, you must drop something at least once before any grasp/fill.

    - We also incorporate current progress: if the cocktail is already in the shaker and
      shaked, then the remaining work for that goal is closer to just the final pour.

    Implementation notes:
    - We must use fast bitwise checks: (state >> fact_id) & 1
    - Goals are stored as a bitwise integer in model.goals; we extract them once.
    - The heuristic returns a non-negative integer. For states where no goal facts exist,
      it returns 0.

    This heuristic is intentionally lightweight: it focuses on the dominant structure
    and scales well to the largest instance (50 cocktails) by evaluating each goal fact
    independently and summing per-goal lower bounds.
    """

    def __init__(self, model, barman_gpt52_heuristic_19: Any = None):
        super().__init__(model, barman_gpt52_heuristic_19)
        self.model = model

        # --- Parse all facts once and build indices for O(1) lookup ---
        self._fact_by_name: Dict[str, int] = {}  # name -> global_id
        # predicate -> list of (global_id, args)
        self._facts_by_pred: Dict[str, List[Tuple[int, List[str]]]] = {}

        for f in self.model.facts:
            self._fact_by_name[f.name] = f.global_id
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._facts_by_pred.setdefault(pred, []).append((f.global_id, args))

        # --- Extract goal facts (explicit) once ---
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        self._goal_contains_shot: List[Tuple[str, str, int]] = []  # (shot, cocktail, fact_id)

        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "contains" and len(args) == 2:
                ctn, bev = args[0], args[1]
                # in this domain, goals are always shot contains cocktail
                # but we don't assume type; we just keep contains facts.
                gid = self._fact_by_name.get(gname)
                if gid is not None:
                    self._goal_contains_shot.append((ctn, bev, gid))

        # --- Helper sets for faster membership ---
        self._all_handempty_ids: List[int] = [fid for fid, _ in self._facts_by_pred.get("handEmpty", [])]

        # Pre-index: for each shot, list of all +contains[shot,*] fact ids (any beverage)
        self._contains_by_container: Dict[str, List[Tuple[int, str]]] = {}
        for fid, args in self._facts_by_pred.get("contains", []):
            if len(args) != 2:
                continue
            cont, bev = args
            self._contains_by_container.setdefault(cont, []).append((fid, bev))

        # Pre-index empty[container]
        self._empty_fact_id: Dict[str, int] = {}
        for fid, args in self._facts_by_pred.get("empty", []):
            if len(args) == 1:
                self._empty_fact_id[args[0]] = fid

        # Pre-index: +shaked[shaker], +unshaked[shaker]
        self._shaked_fact_id: Dict[str, int] = {}
        self._unshaked_fact_id: Dict[str, int] = {}
        for fid, args in self._facts_by_pred.get("shaked", []):
            if len(args) == 1:
                self._shaked_fact_id[args[0]] = fid
        for fid, args in self._facts_by_pred.get("unshaked", []):
            if len(args) == 1:
                self._unshaked_fact_id[args[0]] = fid

        # Identify shakers (from facts mentioning shakerLevel/shakerEmptyLevel/ontable etc.)
        self._shakers: Set[str] = set()
        for pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked"):
            for _, args in self._facts_by_pred.get(pred, []):
                if args:
                    self._shakers.add(args[0])

        # Pre-index: contains[shaker,cocktail] fact ids for quick progress checks
        self._contains_shaker_cocktail: Dict[Tuple[str, str], int] = {}
        for fid, args in self._facts_by_pred.get("contains", []):
            if len(args) != 2:
                continue
            cont, bev = args
            if cont in self._shakers:  # likely shaker
                self._contains_shaker_cocktail[(cont, bev)] = fid

    # ------------------ core API ------------------
    def __call__(self, state: int, task_network=None) -> int:
        """Return an integer estimate of remaining effort."""
        if not self._goal_contains_shot:
            return 0

        h = 0

        # Small global lower bound: if no hand is empty, at least one drop is needed.
        # (Most actions require a handEmpty somewhere; dropping creates handEmpty.)
        no_empty_hand = True
        for fid in self._all_handempty_ids:
            if (state >> fid) & 1:
                no_empty_hand = False
                break
        if no_empty_hand and self._all_handempty_ids:
            h += 1

        # Sum independent per-goal estimates (admissible-ish under strong relaxations).
        for shot, cocktail, goal_fid in self._goal_contains_shot:
            if (state >> goal_fid) & 1:
                continue  # already achieved

            # Base lower bound for producing a cocktail in a shot from scratch.
            # Under delete-relaxation and ignoring hand/cleaning constraints, ~6.
            per_goal = 6

            # If the cocktail is already prepared in some shaker and shaked, then
            # the remaining work could be as low as just the final pour (1).
            # We take the best-case across all shakers.
            best_progress = per_goal
            for shaker in self._shakers:
                c_fid = self._contains_shaker_cocktail.get((shaker, cocktail))
                if c_fid is None:
                    continue
                if (state >> c_fid) & 1:
                    shaked_id = self._shaked_fact_id.get(shaker)
                    if shaked_id is not None and ((state >> shaked_id) & 1):
                        best_progress = min(best_progress, 1)
                    else:
                        # cocktail in shaker but not shaked is rare (shake adds cocktail),
                        # but if it happens, at least shake+pour => 2
                        best_progress = min(best_progress, 2)

            per_goal = best_progress

            # If the shot is not empty, at least one empty-shot is needed before it can
            # receive the cocktail (pour requires empty shot). This is a true lower bound.
            empty_id = self._empty_fact_id.get(shot)
            if empty_id is not None and (((state >> empty_id) & 1) == 0):
                per_goal += 1

            h += per_goal

        return h

    # ------------------ utilities ------------------
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
