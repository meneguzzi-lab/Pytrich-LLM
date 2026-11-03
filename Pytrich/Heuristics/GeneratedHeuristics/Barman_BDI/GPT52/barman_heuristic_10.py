from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


@dataclass(frozen=True)
class _Atom:
    """Parsed grounded fact representation.

    Grounded facts come as strings like '+predicate[arg1,arg2]' or '-predicate[...]'.
    We ignore the +/- polarity and only use the positive predicate symbol and arguments.
    """

    pred: str
    args: Tuple[str, ...]


class BarmanGPT52Heuristic10(Heuristic):
    """Domain-dependent heuristic for the HDDL Barman domain.

    Key domain insights exploited
    ---------------------------
    The top-level workload is a sequence of tasks AchieveContainsShotCocktail(shot, cocktail).
    Each cocktail typically requires (in the non-null method):
      - build cocktail in shaker using two ingredients
      - shake
      - ensure target shot is clean+empty
      - pour shaker -> shot

    The domain has a *single shaker* (in provided instances), two hands, many shots.
    Shaker capacity is modeled via levels, but in instances 'next level1 level1' and
    'next level2 level2' exist, so the shaker level constraints are effectively easy.

    The main search difficulty comes from:
      - choosing a clean shot, keeping hands free, and avoiding unnecessary grasp/drop
      - ensuring the shaker is in the right condition (clean, empty/unshaked) before mixing

    Heuristic strategy (lower bound, state+task-network aware)
    ---------------------------------------------------------
    We compute an *admissible* estimate as a sum of independent lower bounds:

    1) Remaining "serve" tasks: for each pending AchieveContainsShotCocktail(s,c)
       not already satisfied in the current state, at least one primitive action
       will be needed somewhere in the future, so contribute +1.

    2) Shaker readiness lower bound (only if there exists at least one unsatisfied serve task):
       If the shaker is not clean or not empty or not unshaked, at least one action
       (clean-shaker or empty-shaker or shake-related prep) will be required; contribute
       up to +1 for each missing readiness aspect, but capped conservatively to keep
       the bound safe under potential overlaps.

       In this domain, to make a new cocktail you need: clean(shaker) and then after
       first pour-shot-to-*-shaker, shaker becomes not clean and not empty; however you
       can still add second ingredient without cleaning (AddIngredientToUsedShaker).
       For the next cocktail, you need empty+clean again.

       We therefore only use a very conservative readiness bound: if shaker is currently
       holding some cocktail (contains(shaker, cocktailX)) and shaked(shaker), then at
       least one action is needed before it can be used for a *different* cocktail,
       i.e., either pour-shaker-to-shot (to finish) or empty-shaker (to reset). This
       helps avoid expanding branches that start decomposing next cocktails too early.

    3) Pending abstract tasks in the current task network: each abstract task will
       require at least one refinement step (a decomposition or direct satisfaction via
       a null method). We count only those abstract tasks whose "null" condition is
       not already true in the state, contributing +1 each.

    Efficiency considerations
    -------------------------
    - Uses bitwise checks for facts via precomputed global_ids.
    - Parses fact strings once at initialization to build indices.
    - Works well for small and large instances: the large instances are dominated by
      many independent serve tasks; counting unsatisfied ones gives strong guidance.

    Notes
    -----
    This heuristic is admissible (a lower bound):
      - Each unsatisfied required serve/abstract goal will need at least one step.
      - Shaker "blocking" contributes only when it is certainly blocking future progress.
    """

    def __init__(self, model, barman_gpt52_heuristic_10: bool = True):
        super().__init__(model)
        self.enabled = bool(barman_gpt52_heuristic_10)

        # --- Parse all facts once, create indices for fast lookup ---
        self._fact_by_atom: Dict[_Atom, int] = {}
        self._facts_by_pred: Dict[str, List[Tuple[Tuple[str, ...], int]]] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            atom = _Atom(pred=pred, args=tuple(args))
            self._fact_by_atom[atom] = f.global_id
            self._facts_by_pred.setdefault(pred, []).append((atom.args, f.global_id))

        # Precompute goal facts (bitwise integer stored in model.goals)
        self._goal_fact_ids: List[int] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue
            gid = self._fact_by_atom.get(_Atom(pred=pred, args=tuple(args)))
            if gid is not None:
                self._goal_fact_ids.append(gid)

        # Cache some useful predicate-specific global_id lists
        self._contains_facts = self._facts_by_pred.get("contains", [])
        self._shaked_facts = self._facts_by_pred.get("shaked", [])
        self._unshaked_facts = self._facts_by_pred.get("unshaked", [])
        self._clean_facts = self._facts_by_pred.get("clean", [])
        self._empty_facts = self._facts_by_pred.get("empty", [])

        # Identify shakers from facts mentioning them (instances use shaker1)
        self._shakers: List[str] = self._collect_objects_of_type_from_facts("shaker")

    # ------------------------- Core heuristic API -------------------------

    def __call__(self, state: int, task_network: Optional[list] = None) -> int:
        if not self.enabled:
            return 0
        if task_network is None:
            task_network = []

        h = 0

        # (A) Count unsatisfied "serve" tasks in the current task network
        #     AchieveContainsShotCocktail(shot, cocktail)
        pending_serves = self._count_unsatisfied_serve_tasks(state, task_network)
        h += pending_serves

        # (B) Count other abstract tasks that are not already satisfied by a null method condition.
        h += self._count_unsatisfied_abstract_tasks_by_null_conditions(state, task_network)

        # (C) If we still have work, add conservative shaker blocking lower bound
        if pending_serves > 0:
            h += self._shaker_blocking_lower_bound(state, task_network)

        return h

    # ------------------------- Task-network analysis -------------------------

    def _count_unsatisfied_serve_tasks(self, state: int, task_network: list) -> int:
        cnt = 0
        for t in task_network:
            # AbstractTask instances have .name
            name = getattr(t, "name", "")
            if not name.startswith("AchieveContainsShotCocktail"):
                continue

            # Pytrich typically encodes grounded task names like:
            # 'AchieveContainsShotCocktail shot2 cocktail1' or 'AchieveContainsShotCocktail(shot2,cocktail1)'
            shot, cocktail = self._parse_task_args(name)
            if shot is None or cocktail is None:
                # If parsing fails, still count it (admissible, weak)
                cnt += 1
                continue

            # If already contains(shot,cocktail), the null method applies => 0 needed
            fact_id = self._fact_by_atom.get(_Atom("contains", (shot, cocktail)))
            if fact_id is None:
                cnt += 1
                continue
            if not self._is_true(state, fact_id):
                cnt += 1
        return cnt

    def _count_unsatisfied_abstract_tasks_by_null_conditions(self, state: int, task_network: list) -> int:
        """Very lightweight HTN-aware component.

        For each abstract task in the current TN, if its corresponding null-method
        precondition is NOT true, then at least one refinement/action is necessary.

        We only implement null-conditions that are explicit and cheap to test:
          - AchieveContainsShotCocktail(s,c): contains(s,c)
          - AchieveContainsShakerCocktail(sh,c): contains(sh,c)
          - AchieveContainsShotIngredient(shot,i): contains(shot,i)
          - AchieveContainsShakerIngredient(sh,i): contains(sh,i)
          - AchieveCleanShot(shot): clean(shot)
          - AchieveCleanShaker(sh): clean(sh)
          - AchieveHolding(hand,container): holding(hand,container)
          - AchieveOnTable(container): ontable(container)
          - AchieveHandEmpty(hand): handEmpty(hand)
        """

        null_checks = {
            "AchieveContainsShotCocktail": "contains",
            "AchieveContainsShakerCocktail": "contains",
            "AchieveContainsShotIngredient": "contains",
            "AchieveContainsShakerIngredient": "contains",
            "AchieveCleanShot": "clean",
            "AchieveCleanShaker": "clean",
            "AchieveHolding": "holding",
            "AchieveOnTable": "ontable",
            "AchieveHandEmpty": "handEmpty",
        }

        cnt = 0
        for t in task_network:
            tname = getattr(t, "name", "")
            base = self._task_base_name(tname)
            if base not in null_checks:
                continue
            pred = null_checks[base]
            args = self._parse_task_all_args(tname)
            if args is None:
                cnt += 1
                continue
            atom = _Atom(pred=pred, args=tuple(args))
            fid = self._fact_by_atom.get(atom)
            if fid is None:
                cnt += 1
                continue
            if not self._is_true(state, fid):
                cnt += 1
        return cnt

    # ------------------------- Shaker-specific bound -------------------------

    def _shaker_blocking_lower_bound(self, state: int, task_network: list) -> int:
        """Conservative lower bound capturing 'shaker is currently busy' situations.

        If any shaker currently contains some cocktail AND is shaked, then before making
        a different cocktail you must at least perform one primitive action to progress
        (typically pour-shaker-to-shot for some pending serve, or empty-shaker).

        We add 1 if:
          - exists shaker s, cocktail c where contains(s,c) and shaked(s)
          - and there exists a pending AchieveContainsShotCocktail(_, c2) with c2 != c
            OR no pending serve for c (meaning shaker cocktail is not directly requested)

        This is still admissible: at least one action is required to resolve the mismatch.
        """

        # Collect pending cocktails in TN
        pending_cocktails: List[str] = []
        for t in task_network:
            name = getattr(t, "name", "")
            if not name.startswith("AchieveContainsShotCocktail"):
                continue
            _, cocktail = self._parse_task_args(name)
            if cocktail is not None:
                pending_cocktails.append(cocktail)

        if not pending_cocktails:
            return 0

        pending_set = set(pending_cocktails)

        # Check current shaker contents
        for (args, fid) in self._contains_facts:
            # contains(container, beverage)
            if len(args) != 2:
                continue
            cont, bev = args
            if cont not in self._shakers:
                continue

            if not self._is_true(state, fid):
                continue

            # Is shaker shaked?
            shaked_id = self._fact_by_atom.get(_Atom("shaked", (cont,)))
            if shaked_id is None or not self._is_true(state, shaked_id):
                continue

            # If shaker holds cocktail 'bev'
            # If pending contains some different cocktail or doesn't contain bev, then we're blocked
            if bev not in pending_set or any(c != bev for c in pending_set):
                return 1

        return 0

    # ------------------------- Helpers -------------------------

    def _is_true(self, state: int, fact_global_id: int) -> bool:
        return ((state >> fact_global_id) & 1) == 1

    def _parse_fact_name(self, name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _task_base_name(self, task_name: str) -> str:
        s = task_name.strip()
        # Possible formats:
        #  - 'AchieveCleanShot shot1'
        #  - 'AchieveCleanShot(shot1)'
        #  - 'AchieveContainsShotCocktail shot2 cocktail1'
        #  - 'AchieveContainsShotCocktail(shot2,cocktail1)'
        if "(" in s:
            return s.split("(", 1)[0].strip()
        return s.split()[0] if s else s

    def _parse_task_all_args(self, task_name: str) -> Optional[List[str]]:
        s = task_name.strip()
        if not s:
            return None

        if "(" in s and s.endswith(")"):
            inside = s[s.find("(") + 1 : -1].strip()
            if inside == "":
                return []
            return [a.strip() for a in inside.split(",")]

        parts = s.split()
        if len(parts) < 2:
            return []
        return parts[1:]

    def _parse_task_args(self, task_name: str) -> Tuple[Optional[str], Optional[str]]:
        args = self._parse_task_all_args(task_name)
        if args is None or len(args) < 2:
            return None, None
        return args[0], args[1]

    def _collect_objects_of_type_from_facts(self, type_hint: str) -> List[str]:
        """Best-effort object collection.

        Pytrich does not directly expose typed objects here, so we infer from fact strings.
        For this domain, shaker objects appear as first argument of shaker-only predicates
        like 'shakerLevel', 'shaked', 'unshaked', 'shakerEmptyLevel'.
        """
        candidates = set()
        for pred in ("shakerLevel", "shakerEmptyLevel", "shaked", "unshaked"):
            for args, _ in self._facts_by_pred.get(pred, []):
                if len(args) >= 1:
                    candidates.add(args[0])
        # Fallback: any object name containing 'shaker'
        if not candidates:
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred is None:
                    continue
                for a in args:
                    if "shaker" in a:
                        candidates.add(a)
        return sorted(candidates)
