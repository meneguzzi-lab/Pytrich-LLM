from __future__ import annotations

from typing import Dict, List, Tuple, Optional

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic15(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Rationale / domain insights
    ---------------------------
    The training instances have a very regular structure: the HTN goal network
    is a sequence of tasks of the form:
        (AchieveContainsShotCocktail shotX cocktailY)

    In this domain, producing one cocktail in a shot *necessarily* requires (in
    the non-null case) the following primitive actions (lower bound in a relaxed
    sense):
      - make the cocktail in the shaker:
          fill-shot (ingredient part1)
          pour-shot-to-*-shaker (part1)
          fill-shot (ingredient part2)
          pour-shot-to-*-shaker (part2)
          shake
      - pour shaker to target shot:
          pour-shaker-to-shot

    Additionally, each 'fill-shot' requires the shot being held by one hand and
    the other hand empty, and the shot must be clean+empty. In the benchmark
    initial states, shots and shaker start clean+empty and both hands empty, so
    the dominant cost is the fixed production pipeline above.

    Key heuristic idea
    ------------------
    Provide an admissible lower bound by:
      1) Counting remaining "deliveries" (unsatisfied +contains[shot,cocktail]
         goal facts): each missing delivery needs at least one primitive action
         (indeed, at least pour-shaker-to-shot), so +1 each.
      2) For each missing delivery, if the cocktail is not already present in
         the shaker (contains[shaker,cocktail] AND shaked[shaker]), we add a
         conservative lower bound for "making" that cocktail: at least 1 shake.
         (We do NOT count ingredient filling/pouring because those can be
         partially achieved already and counting them without careful reasoning
         could overestimate. Counting only the mandatory 'shake' remains
         admissible.)

    This yields a fast, state-based heuristic with good guidance on large task
    networks: it strongly prefers nodes where more cocktails have already been
    produced in the shaker and/or more shots already contain their target.

    Implementation notes (Pytrich constraints)
    -----------------------------------------
    - We must not call model.state_explicit_repr(state) on large instances.
      We only use bit tests via fact.global_id.
    - Goals are stored as a bitmask in model.goals.
    - Facts are in the grounded +/-predicate[arg,...] form.

    The heuristic is admissible (never counts more than the true remaining
    primitive actions) and very cheap: it only iterates over goal facts and does
    O(1) bit tests per relevant fact.
    """

    def __init__(self, model, barman_gpt52_heuristic_15: str = "barman_gpt52_heuristic_15"):
        super().__init__(model, barman_gpt52_heuristic_15)
        self.model = model

        # Cache: map from (predicate, tuple(args), polarity) -> Fact
        # polarity: True for '+', False for '-'
        self._fact_index: Dict[Tuple[str, Tuple[str, ...], bool], object] = {}
        for f in self.model.facts:
            pred, args, pol = self._parse_fact_name_full(f.name)
            if pred is None:
                continue
            self._fact_index[(pred, tuple(args), pol)] = f

        # Determine (a) shaker object(s), (b) all goal deliveries (+contains shot cocktail)
        self._shakers: List[str] = []
        self._goal_deliveries: List[Tuple[str, str]] = []  # (shot, cocktail)

        # Identify shakers from facts: any argument that appears in shakerLevel(?s, ?lvl)
        # (this is robust and avoids needing typing information).
        for f in self.model.facts:
            pred, args, pol = self._parse_fact_name_full(f.name)
            if pol is True and pred == "shakerLevel" and len(args) == 2:
                self._shakers.append(args[0])
        # Unique, stable order
        self._shakers = sorted(set(self._shakers))

        # Extract goal facts (bitmask) and keep only +contains[shot,cocktail]
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "contains" and len(args) == 2:
                # In this domain, goals are of the form contains(shot, cocktail)
                self._goal_deliveries.append((args[0], args[1]))

        # Pre-fetch Fact objects for speed
        self._goal_contains_facts: List[object] = []
        for shot, cocktail in self._goal_deliveries:
            ff = self._fact_index.get(("contains", (shot, cocktail), True))
            if ff is not None:
                self._goal_contains_facts.append(ff)

        # Also pre-fetch shaker related facts per cocktail, per shaker
        # For admissibility we only use: contains(shaker,cocktail) and shaked(shaker)
        self._shaker_contains_fact: Dict[Tuple[str, str], object] = {}
        self._shaker_shaked_fact: Dict[str, object] = {}

        for s in self._shakers:
            shaked_f = self._fact_index.get(("shaked", (s,), True))
            if shaked_f is not None:
                self._shaker_shaked_fact[s] = shaked_f

        # Collect all cocktails mentioned in goals
        cocktails = sorted(set(c for _, c in self._goal_deliveries))
        for s in self._shakers:
            for c in cocktails:
                cf = self._fact_index.get(("contains", (s, c), True))
                if cf is not None:
                    self._shaker_contains_fact[(s, c)] = cf

    # --- Fact parsing helpers (as required by prompt) ---
    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name2 = name[1:]
            bracket_idx = name2.find("[")
            if bracket_idx > 0 and name2.endswith("]"):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_fact_name_full(self, name: str):
        """Return (predicate, args, polarity_plus).

        polarity_plus=True for '+', False for '-'.
        """
        name = name.strip()
        if not name:
            return None, [], True
        if name[0] not in "+-":
            return None, [], True
        pol = (name[0] == "+")
        pred, args = self._parse_fact_name(name)
        return pred, args, pol

    # --- Bit test ---
    @staticmethod
    def _is_true(state_bitwise: int, fact) -> bool:
        return ((state_bitwise >> fact.global_id) & 1) == 1

    # --- Main heuristic ---
    def __call__(self, node) -> int:
        state = node.state

        # If no deliveries in goals, heuristic is 0
        if not self._goal_contains_facts:
            return 0

        h = 0

        # For each goal delivery (shot contains cocktail)
        # - if already true: cost 0
        # - else: at least 1 primitive action to eventually make it true
        #         (pour-shaker-to-shot is necessary in all constructive methods)
        #   plus a very conservative +1 if the cocktail is not already available
        #   as a shaked cocktail in some shaker.
        for (shot, cocktail), goal_fact in zip(self._goal_deliveries, self._goal_contains_facts):
            if self._is_true(state, goal_fact):
                continue

            # Missing delivery
            h += 1

            # Check if cocktail is already prepared in ANY shaker as shaked & contains
            prepared = False
            for s in self._shakers:
                shaked_f = self._shaker_shaked_fact.get(s)
                contains_f = self._shaker_contains_fact.get((s, cocktail))
                if shaked_f is None or contains_f is None:
                    continue
                if self._is_true(state, shaked_f) and self._is_true(state, contains_f):
                    prepared = True
                    break

            # If not prepared, we must at least execute a 'shake' at some point.
            # Counting only 1 here is admissible (true cost >= 1).
            if not prepared:
                h += 1

        return h
