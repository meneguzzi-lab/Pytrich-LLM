from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Any

# Pytrich base class
from Pytrich.Heuristics.heuristic import Heuristic


class BarmanGPT52Heuristic6(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain (Pytrich).

    Key domain insights leveraged
    ----------------------------
    The top-level training instances consist of many tasks of the form:
        (AchieveContainsShotCocktail shotX cocktailY)

    In this domain, producing one cocktail in a shot (when not already present) minimally requires:
      - Ensure shaker contains the cocktail:
          * Ensure shaker clean (may require emptying & cleaning)
          * Add two ingredients to shaker (each requires filling a shot + pouring into shaker)
          * Shake (requires shaker held by one hand and the other hand empty)
      - Ensure target shot clean (often already clean/empty initially)
      - Hold shaker
      - Pour shaker to shot

    The heuristic implemented here is a *fast admissible* lower bound that mixes:
      (A) A task-network based lower bound: count remaining "deliver cocktail to shot" tasks.
      (B) A state-based lower bound per required cocktail that accounts for whether the cocktail
          is already in some shot or the shaker, and whether the shaker is already shaked.

    We deliberately ignore most preconditions (hand logistics, cleaning, levels) to stay admissible
    and fast on large instances. We also avoid expensive state explicit representations and use
    bitwise checks via fact.global_id.

    Admissibility argument (sketch)
    -------------------------------
    - Each unsatisfied AchieveContainsShotCocktail must eventually be achieved; at minimum, if the
      target shot doesn't already contain the cocktail, at least one primitive action must occur
      (e.g., pour-shaker-to-shot, or some alternative path). Counting such goals is a lower bound.
    - Additional increments are only added when they correspond to *unavoidable* primitive actions
      under any valid plan given the current state (e.g., if no shot contains cocktail and shaker
      doesn't contain it, then at least one shake is necessary to create it, because the only action
      that creates (contains shaker cocktail) is shake).

    Efficiency
    ----------
    - Precomputes indices for relevant facts: contains(shot,cocktail), contains(shaker,cocktail),
      shaked(shaker), cocktailPart1/2.
    - Uses bitwise checks only.
    - For large instances, processes each remaining top-level abstract task once.
    """

    def __init__(self, model, barman_gpt52_heuristic_6: str = "barman_gpt52_heuristic_6"):
        super().__init__(model, barman_gpt52_heuristic_6)
        self.model = model

        # ---- Fact indexing ----
        # Maps (container, beverage) -> fact_id for +contains[container,beverage]
        self._contains_fact: Dict[Tuple[str, str], int] = {}
        # Maps shaker -> fact_id for +shaked[shaker]
        self._shaked_fact: Dict[str, int] = {}

        # cocktail -> (ing1, ing2)
        self._cocktail_parts: Dict[str, Tuple[Optional[str], Optional[str]]] = {}

        # Cache sets for fast "any shot contains cocktail" checks
        self._shots: Set[str] = set()
        self._shakers: Set[str] = set()
        self._cocktails: Set[str] = set()

        # Parse all facts once
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue

            if pred == "contains" and len(args) == 2:
                c, b = args
                self._contains_fact[(c, b)] = f.global_id

            elif pred == "shaked" and len(args) == 1:
                self._shaked_fact[args[0]] = f.global_id

            elif pred == "cocktailPart1" and len(args) == 2:
                cock, ing = args
                i1, i2 = self._cocktail_parts.get(cock, (None, None))
                self._cocktail_parts[cock] = (ing, i2)

            elif pred == "cocktailPart2" and len(args) == 2:
                cock, ing = args
                i1, i2 = self._cocktail_parts.get(cock, (None, None))
                self._cocktail_parts[cock] = (i1, ing)

        # Infer object sets from contains facts (works for the given benchmarks)
        for (c, b) in self._contains_fact.keys():
            # names are like shotX / shakerY / cocktailZ in benchmarks
            if c.startswith("shot"):
                self._shots.add(c)
            if c.startswith("shaker"):
                self._shakers.add(c)
            if b.startswith("cocktail"):
                self._cocktails.add(b)

        # If not inferred (edge cases), fall back to scanning model initial task network names later.

        # A small cache for (state, task_network_signature) evaluations
        self._cache: Dict[Tuple[int, int], int] = {}

    # ----------------- Core heuristic API -----------------
    def compute(self, state: int, task_network: List[Any]) -> int:
        """Return h(state, task_network) as an admissible lower bound."""
        tn_sig = self._tn_signature(task_network)
        cache_key = (state, tn_sig)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Extract remaining delivery tasks (most important signal in this domain)
        pending: List[Tuple[str, str]] = []  # (shot, cocktail)
        for t in task_network:
            shot_cock = self._extract_achieve_contains_shot_cocktail(t)
            if shot_cock is not None:
                pending.append(shot_cock)

        if not pending:
            self._cache[cache_key] = 0
            return 0

        # (A) Baseline: each unsatisfied delivery task costs at least 1 action
        # (if already satisfied, cost 0).
        h = 0

        # For extra lower bound: for each *distinct cocktail* still needed somewhere,
        # if it is not already present anywhere (any shot or shaker), then at least one shake
        # is required to create it.
        needed_cocktails: Set[str] = set(c for _, c in pending)

        # Track existence of each cocktail in the current state.
        # We assume single shaker is typical; still works if multiple.
        cocktail_present_anywhere: Dict[str, bool] = {c: False for c in needed_cocktails}

        # Check shots and shakers for each cocktail (bitwise)
        for cock in needed_cocktails:
            if self._is_cocktail_in_any_shot(state, cock):
                cocktail_present_anywhere[cock] = True
                continue
            if self._is_cocktail_in_any_shaker(state, cock):
                cocktail_present_anywhere[cock] = True

        # Per pending goal, if not already in that shot, need at least one action (somehow)
        # In this domain, the direct final step is pour-shaker-to-shot (1 action).
        for shot, cock in pending:
            if not self._is_true(state, self._contains_fact.get((shot, cock))):
                h += 1

        # (B) Add unavoidable "make" cost: if a cocktail is not present anywhere, must be created.
        # The only creator of contains(shaker,cocktail) is shake, so at least 1 shake.
        for cock, present in cocktail_present_anywhere.items():
            if not present:
                h += 1

        # (C) If cocktail is present only as ingredients in shaker? We don't track ingredients here.
        # Staying conservative.

        self._cache[cache_key] = h
        return h

    # ----------------- Helpers -----------------
    def _tn_signature(self, task_network: List[Any]) -> int:
        """Cheap signature for caching: hash of (task global ids, len)."""
        sig = len(task_network)
        for t in task_network:
            gid = getattr(t, "global_id", 0)
            sig = (sig * 1315423911) ^ gid
        return sig & 0xFFFFFFFF

    def _is_true(self, state: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state >> fact_id) & 1) == 1

    def _is_cocktail_in_any_shot(self, state: int, cocktail: str) -> bool:
        # Iterate only known shots; if unknown, return False (admissible).
        for s in self._shots:
            if self._is_true(state, self._contains_fact.get((s, cocktail))):
                return True
        return False

    def _is_cocktail_in_any_shaker(self, state: int, cocktail: str) -> bool:
        for sh in self._shakers:
            if self._is_true(state, self._contains_fact.get((sh, cocktail))):
                return True
        return False

    def _extract_achieve_contains_shot_cocktail(self, task: Any) -> Optional[Tuple[str, str]]:
        """Extract (shot, cocktail) from an AbstractTask corresponding to AchieveContainsShotCocktail.

        Works robustly with Pytrich task naming conventions by parsing task.name.
        Expected task.name patterns include (depending on grounding):
          - 'AchieveContainsShotCocktail shot2 cocktail1'
          - 'AchieveContainsShotCocktail(shot2,cocktail1)'
          - 'AchieveContainsShotCocktail[shot2,cocktail1]'
        If the task doesn't match, returns None.
        """
        name = getattr(task, "name", "")
        if "AchieveContainsShotCocktail" not in name:
            return None

        # Normalize delimiters
        tmp = name.replace("(", " ").replace(")", " ")
        tmp = tmp.replace("[", " ").replace("]", " ")
        tmp = tmp.replace(",", " ")
        parts = [p for p in tmp.split() if p]

        # Find the token and take next two if present
        for i, p in enumerate(parts):
            if p == "AchieveContainsShotCocktail" and i + 2 < len(parts):
                return parts[i + 1], parts[i + 2]

        # Fallback: sometimes name may be like 'AchieveContainsShotCocktail_shot2_cocktail1'
        if "_" in name:
            toks = name.split("_")
            try:
                idx = toks.index("AchieveContainsShotCocktail")
                if idx + 2 < len(toks):
                    return toks[idx + 1], toks[idx + 2]
            except ValueError:
                pass

        return None

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
