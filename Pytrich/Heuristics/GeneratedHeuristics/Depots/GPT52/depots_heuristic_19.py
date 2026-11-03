from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _OnGoal:
    top: str
    bottom: str


class DepotsGPT52Heuristic19(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Rationale (domain insights)
    --------------------------
    Depots goals are exclusively of the form (on crate surface). Achieving one such goal
    typically requires:
      - ensuring both objects are at the same place (may require trucking => Drive),
      - ensuring the destination surface is clear (may require clearing a stack),
      - lifting the crate (possibly from surface or from truck),
      - dropping it onto the destination.

    This heuristic is a *fast lower bound* that combines:
      (1) Goal-count: number of unsatisfied +on[crate,surface] goal atoms.
      (2) Necessary logistics lower bound:
          - If crate not co-located with its target surface, at least one Drive is needed.
            (In this domain there are no road constraints; one Drive suffices.)
      (3) Necessary manipulation lower bound:
          - For each unsatisfied on-goal, at least one Drop will be needed.
          - If crate is currently in a truck, at least one Unload is needed before Drop.
          - If crate is currently on some surface (not lifting/in truck), at least one Lift is needed.

    Additionally, it includes a lightweight *blocking-stack* estimate:
      - If the target surface is not clear, some crate sits on it. To clear it, that blocking
        crate must be moved away, requiring at least one Lift+Load (or Lift+Drop elsewhere).
        We count +1 per unique blocking top-crate (avoids double-counting).

    Properties
    ----------
    - Not guaranteed admissible in all cases (because it sums multiple necessary counts that
      can sometimes be achieved together), but is designed to be informative and very cheap.
    - Uses only bitwise checks on facts (no explicit state repr scanning), suitable for large instances.

    Implementation notes
    --------------------
    - We parse and index facts once in __init__ for O(1) access at evaluation time.
    - Goals are obtained via model.state_explicit_repr(model.goals) as required.
    """

    def __init__(self, model, depots_gpt52_heuristic_19: str = "depots_gpt52_heuristic_19"):
        super().__init__(model, depots_gpt52_heuristic_19)
        self.model = model

        # --- Index facts by predicate for fast lookup ---
        self._fact_id_by_name: Dict[str, int] = {}
        for f in self.model.facts:
            self._fact_id_by_name[f.name] = f.global_id

        # at[x,place]
        self._at_facts: Dict[str, List[Tuple[str, int]]] = {}  # obj -> [(place, fact_id), ...]
        # in[crate,truck]
        self._in_facts: Dict[str, List[Tuple[str, int]]] = {}  # crate -> [(truck, fact_id), ...]
        # on[top,bottom]
        self._on_facts_by_top: Dict[str, List[Tuple[str, int]]] = {}  # top -> [(bottom, fact_id), ...]
        self._on_facts_by_bottom: Dict[str, List[Tuple[str, int]]] = {}  # bottom -> [(top, fact_id), ...]
        # clear[surface]
        self._clear_fact: Dict[str, int] = {}  # surface -> fact_id

        # lifting[hoist,crate]
        self._lifting_by_crate: Dict[str, List[int]] = {}  # crate -> [fact_id,...]

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == "at" and len(args) == 2:
                obj, place = args
                self._at_facts.setdefault(obj, []).append((place, fact.global_id))
            elif pred == "in" and len(args) == 2:
                crate, truck = args
                self._in_facts.setdefault(crate, []).append((truck, fact.global_id))
            elif pred == "on" and len(args) == 2:
                top, bottom = args
                self._on_facts_by_top.setdefault(top, []).append((bottom, fact.global_id))
                self._on_facts_by_bottom.setdefault(bottom, []).append((top, fact.global_id))
            elif pred == "clear" and len(args) == 1:
                surf = args[0]
                self._clear_fact[surf] = fact.global_id
            elif pred == "lifting" and len(args) == 2:
                _hoist, crate = args
                self._lifting_by_crate.setdefault(crate, []).append(fact.global_id)

        # --- Extract relevant goal atoms: +on[crate,surface] ---
        self._on_goals: List[_OnGoal] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self._on_goals.append(_OnGoal(args[0], args[1]))

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
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return ((state_bitwise >> fact_id) & 1) == 1

    def _current_place_of(self, state: int, obj: str) -> Optional[str]:
        """Return the unique place p such that at[obj,p] holds, if any."""
        for place, fid in self._at_facts.get(obj, []):
            if self._is_true(state, fid):
                return place
        return None

    def _is_in_truck(self, state: int, crate: str) -> bool:
        for _truck, fid in self._in_facts.get(crate, []):
            if self._is_true(state, fid):
                return True
        return False

    def _is_lifting(self, state: int, crate: str) -> bool:
        for fid in self._lifting_by_crate.get(crate, []):
            if self._is_true(state, fid):
                return True
        return False

    def _is_on(self, state: int, top: str, bottom: str) -> bool:
        for b, fid in self._on_facts_by_top.get(top, []):
            if b == bottom and self._is_true(state, fid):
                return True
        return False

    def _some_support_of_crate(self, state: int, crate: str) -> bool:
        """True if crate is on some surface (incl. pallet or crate)."""
        for _bottom, fid in self._on_facts_by_top.get(crate, []):
            if self._is_true(state, fid):
                return True
        return False

    def _surface_clear(self, state: int, surface: str) -> Optional[bool]:
        fid = self._clear_fact.get(surface)
        if fid is None:
            return None
        return self._is_true(state, fid)

    def _blocking_top_of_surface(self, state: int, surface: str) -> Optional[str]:
        """If some crate is on 'surface', return one such top-crate (there should be at most one)."""
        for top, fid in self._on_facts_by_bottom.get(surface, []):
            if self._is_true(state, fid):
                return top
        return None

    def __call__(self, node) -> int:
        state = node.state

        # If no relevant goals, trivial.
        if not self._on_goals:
            return 0

        unsat = 0
        need_drive = 0
        need_unload = 0
        need_lift = 0
        need_drop = 0

        blocking_crates: Set[str] = set()

        for g in self._on_goals:
            if self._is_on(state, g.top, g.bottom):
                continue

            unsat += 1
            need_drop += 1  # to establish the on relation in this domain

            # Destination clearing (cheap proxy): if bottom not clear, some crate blocks it.
            clear_val = self._surface_clear(state, g.bottom)
            if clear_val is False:
                blocker = self._blocking_top_of_surface(state, g.bottom)
                if blocker is not None and blocker != g.top:
                    blocking_crates.add(blocker)

            # Location coherence: if crate and destination not at same place => at least one Drive.
            p_crate = self._current_place_of(state, g.top)
            p_dest = self._current_place_of(state, g.bottom)
            if p_crate is not None and p_dest is not None and p_crate != p_dest:
                need_drive += 1

            # If crate is in truck => must unload before drop.
            if self._is_in_truck(state, g.top):
                need_unload += 1
            else:
                # If not already lifting and currently supported somewhere => must lift at least once.
                if (not self._is_lifting(state, g.top)) and self._some_support_of_crate(state, g.top):
                    need_lift += 1

        # Clearing blockers: each unique blocking crate needs to be moved away at least once.
        # Count it as +1 (very conservative proxy for Lift+Load/Drop sequence).
        clear_blockers_lb = len(blocking_crates)

        # Combine with light weights; tuned to guide search but remain cheap.
        # Unsatisfied goals dominates; drives/manipulations add guidance.
        h = (
            unsat
            + need_drive
            + need_drop
            + need_unload
            + need_lift
            + clear_blockers_lb
        )

        return int(h)
