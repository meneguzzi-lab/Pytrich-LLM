from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set, Union

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic4(Heuristic):
    """Domain-dependent heuristic for the HDDL *Depots* domain.

    Design rationale (domain insights)
    -------------------------------
    In this domain, all interesting goals are of the form (on c s) where s is a surface
    (pallet or crate). Achieving (on c s) minimally requires:
      - a hoist at the goal place to drop the crate onto the target surface
      - the target surface to be clear
      - the crate to be lifted by the hoist at that place

    From the methods:
      do_put_on(c, s2) can be achieved via:
        m0: already on -> 0
        m1: crate at place -> clear c and s2, lift, drop
        m2: crate in truck -> get truck to place, clear s2, unload, drop
        m3: otherwise -> load to truck, drive, unload+drop

    A strong, cheap lower bound for search guidance is therefore obtained by counting
    *missing essential facts* for each goal (on c s), plus an additional penalty if the
    crate is not currently at the same place as the target surface (requires at least
    one Drive). We also add a small hierarchical component: for each pending abstract
    task do_put_on in the current task network, add a minimal remaining cost estimate.

    This heuristic is NOT guaranteed admissible (it mixes fact-counting and minimal
    action estimates), but it is designed to reduce node expansions in practice while
    keeping computation strictly bitwise (no explicit state decoding).

    Efficiency considerations
    -------------------------
    - All checks use bitwise operations on fact global ids.
    - We pre-index facts by predicate and (arg1,arg2) pairs for O(1) lookups.
    - We avoid iterating all facts/operators per node; per node work is proportional
      to number of goal facts + number of relevant TN tasks.
    """

    def __init__(self, model, depots_gpt52_heuristic_4: str = "depots_gpt52_heuristic_4"):
        super().__init__(model, depots_gpt52_heuristic_4)
        self.model = model

        # --- Precompute goal on-facts ---
        self._goal_on_pairs: List[Tuple[str, str]] = []  # (crate, surface)
        self._goal_on_fact_ids: List[int] = []

        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for fn in goal_fact_names:
            pred, args = self._parse_fact_name(fn)
            if pred == "on" and len(args) == 2:
                self._goal_on_pairs.append((args[0], args[1]))

        # --- Index all relevant facts for fast lookup ---
        # Maps: (pred, a1, a2) -> global_id
        self._fact_id: Dict[Tuple[str, str, str], int] = {}
        # Maps: (pred, a1) -> list of (a2, global_id)
        self._by_pred_arg1: Dict[Tuple[str, str], List[Tuple[str, int]]] = {}

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue
            if len(args) == 2:
                key = (pred, args[0], args[1])
                self._fact_id[key] = f.global_id
                self._by_pred_arg1.setdefault((pred, args[0]), []).append((args[1], f.global_id))

        for c, s in self._goal_on_pairs:
            fid = self._fact_id.get(("on", c, s))
            if fid is not None:
                self._goal_on_fact_ids.append(fid)

        # Cache hoist locations (static objects but at() changes; still we need query per state)
        # We'll just keep list of all hoists encountered via at(hoist, place) facts.
        self._all_hoists: Set[str] = set()
        self._all_trucks: Set[str] = set()
        self._all_places: Set[str] = set()
        self._all_surfaces: Set[str] = set()

        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred == "at" and len(args) == 2:
                obj, place = args
                self._all_places.add(place)
                # object type is not directly accessible here; infer by name prefixes used in benchmarks
                if obj.startswith("hoist"):
                    self._all_hoists.add(obj)
                elif obj.startswith("truck"):
                    self._all_trucks.add(obj)
                elif obj.startswith("pallet") or obj.startswith("crate"):
                    self._all_surfaces.add(obj)
            elif pred in ("clear",) and len(args) == 1:
                if args[0].startswith("pallet") or args[0].startswith("crate"):
                    self._all_surfaces.add(args[0])

        # Weight tuning: chosen to strongly prioritize finishing goal stacks and avoiding relocation.
        self.w_missing_goal_on = 6
        self.w_need_clear_surface = 2
        self.w_need_crate_at_place = 2
        self.w_need_move_between_places = 3
        self.w_need_hoist_at_place = 1
        self.w_need_available_hoist = 1
        self.w_need_lifting_for_drop = 2
        self.w_pending_put_on_task = 4

    # --------------------- Parsing helpers ---------------------
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

    def _is_true(self, state: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state >> fact_id) & 1) == 1

    def _find_place_of(self, state: int, obj: str) -> Optional[str]:
        """Return some place p such that at(obj,p) holds in state, else None."""
        entries = self._by_pred_arg1.get(("at", obj), [])
        for place, fid in entries:
            if ((state >> fid) & 1) == 1:
                return place
        return None

    def _any_hoist_at(self, state: int, place: str) -> Tuple[bool, bool]:
        """Return (exists_hoist_at_place, exists_available_hoist_at_place)."""
        exists = False
        exists_available = False
        for h in self._all_hoists:
            fid_at = self._fact_id.get(("at", h, place))
            if self._is_true(state, fid_at):
                exists = True
                fid_av = self._fact_id.get(("available", h, ""))  # won't exist (available is unary)
                # handle unary available below by scanning facts is too expensive; instead build unary map on demand
                # We implement unary lookup by storing as (pred,arg1,arg2) with arg2="" during init.
        # Fallback: use unary index created lazily
        if not hasattr(self, "_unary_fact_id"):
            self._unary_fact_id = {}
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred and len(args) == 1:
                    self._unary_fact_id[(pred, args[0])] = f.global_id
        for h in self._all_hoists:
            fid_at = self._fact_id.get(("at", h, place))
            if self._is_true(state, fid_at):
                exists = True
                fid_av = self._unary_fact_id.get(("available", h))
                if self._is_true(state, fid_av):
                    exists_available = True
                    break
        return exists, exists_available

    def _is_clear(self, state: int, surf: str) -> bool:
        if not hasattr(self, "_unary_fact_id"):
            self._unary_fact_id = {}
            for f in self.model.facts:
                pred, args = self._parse_fact_name(f.name)
                if pred and len(args) == 1:
                    self._unary_fact_id[(pred, args[0])] = f.global_id
        return self._is_true(state, self._unary_fact_id.get(("clear", surf)))

    def _is_in_truck(self, state: int, crate: str) -> bool:
        # check any in(crate, truck)
        entries = self._by_pred_arg1.get(("in", crate), [])
        for truck, fid in entries:
            if ((state >> fid) & 1) == 1:
                return True
        return False

    def _is_lifting_any(self, state: int, crate: str) -> bool:
        # check any lifting(hoist, crate)
        # lifting is binary in facts: lifting[h,c]
        # We'll search all hoists for which lifting(hoist,crate) fact exists.
        for h in self._all_hoists:
            fid = self._fact_id.get(("lifting", h, crate))
            if fid is not None and ((state >> fid) & 1) == 1:
                return True
        return False

    # --------------------- Heuristic computation ---------------------
    def __call__(self, node) -> int:
        state = node.state
        tn = getattr(node, "task_network", []) or []

        h_val = 0

        # 1) Goal-directed estimate over all (on c s) goals.
        for (c, s) in self._goal_on_pairs:
            fid_goal = self._fact_id.get(("on", c, s))
            if self._is_true(state, fid_goal):
                continue

            # Missing the goal itself is the main signal.
            h_val += self.w_missing_goal_on

            # Target surface must be clear to Drop onto it.
            if not self._is_clear(state, s):
                h_val += self.w_need_clear_surface

            # For Drop, crate must be lifted by some hoist (at the same place as target surface).
            # If not already lifting, at least one Lift/Unload is needed.
            if not self._is_lifting_any(state, c):
                h_val += self.w_need_lifting_for_drop

            # If crate is in a truck, it must be unloaded somewhere; if not in truck, it must be at a place.
            if not self._is_in_truck(state, c):
                # crate should be at some place (otherwise it's being lifted or inconsistent)
                if self._find_place_of(state, c) is None and not self._is_lifting_any(state, c):
                    h_val += self.w_need_crate_at_place

            # Penalize if crate place differs from target surface place (implies at least one Drive).
            place_s = self._find_place_of(state, s)
            place_c = self._find_place_of(state, c)

            if place_s is not None:
                # Need some hoist at the goal place to execute Drop.
                exists_hoist, exists_av = self._any_hoist_at(state, place_s)
                if not exists_hoist:
                    h_val += self.w_need_hoist_at_place
                elif not exists_av and not self._is_lifting_any(state, c):
                    # availability matters mainly when we need to perform Lift/Unload.
                    h_val += self.w_need_available_hoist

            if place_s is not None and place_c is not None and place_s != place_c:
                h_val += self.w_need_move_between_places

        # 2) Hierarchical TN component: if there are pending do_put_on tasks, add small lower bound.
        # In Pytrich, tasks/operators in the TN have .name; abstract tasks likely have names like 'do_put_on ...'
        for t in tn:
            name = getattr(t, "name", "")
            # robust matching: task name contains "do_put_on"
            if "do_put_on" in name:
                # If the corresponding on-fact already holds, cost can be ~0; otherwise add.
                # Try to extract args from something like "do_put_on(crateX,palletY)" (format can vary).
                c, s = self._try_parse_task_args(name)
                if c is not None and s is not None:
                    fid_goal = self._fact_id.get(("on", c, s))
                    if not self._is_true(state, fid_goal):
                        h_val += self.w_pending_put_on_task
                else:
                    h_val += 1  # minimal guidance even if parsing fails

        return int(h_val)

    def _try_parse_task_args(self, task_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort parse of task string to extract (crate,surface).

        Different encodings may appear; we handle common ones:
          - 'do_put_on crate1 pallet2'
          - 'do_put_on(crate1,pallet2)'
          - '(do_put_on crate1 pallet2)'
        """
        s = task_name.strip()
        # remove surrounding parentheses
        if s.startswith("(") and s.endswith(")"):
            s = s[1:-1].strip()

        if "(" in s and ")" in s:
            # function-like
            head = s[: s.find("(")].strip()
            if "do_put_on" not in head:
                return None, None
            inside = s[s.find("(") + 1 : s.rfind(")")]
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[0], parts[1]
            return None, None

        parts = [p for p in s.replace("(", " ").replace(")", " ").replace(",", " ").split() if p]
        if len(parts) >= 3 and parts[0].endswith("do_put_on"):
            return parts[1], parts[2]
        # Sometimes the name may include decomposition prefix; search token 'do_put_on'
        if "do_put_on" in parts:
            i = parts.index("do_put_on")
            if i + 2 < len(parts):
                return parts[i + 1], parts[i + 2]
        return None, None
