from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class DepotsGPT52Heuristic2(Heuristic):
    """Domain-dependent heuristic for the HDDL Depots domain.

    Rationale (domain insights -> heuristic components)
    -----------------------------------------------
    Depots has one dominant top-level task pattern: (do_put_on crate surface).
    Achieving an (on c s) goal generally requires:
      - Being at the same place p (both c and s at p),
      - Having s clear (if not, must clear it by moving the top crate away via truck),
      - Having c clear (so it can be lifted),
      - Performing Lift and Drop (or Unload+Drop if c is in a truck).

    Key structural difficulties:
      1) Clearing stacks: if a surface (crate or pallet) is not clear, at least one crate must be removed.
         In this domain, removal is expensive: it typically involves Lift + Load into a truck, and may
         also require bringing a truck (Drive) to the place.
      2) Transport between places: if crate and target surface are at different places, we need a
         load-drive-unload pipeline.
      3) Hierarchy: abstract tasks decompose into a small set of method schemas (m0..m3), with m0
         being already satisfied and m3 being the expensive cross-location solution.

    Heuristic design
    ----------------
    We compute a fast *lower bound* by summing independent per-goal estimates for each goal fact +on[c,s].
    (This is not strictly admissible due to possible double counting, but it is effective at reducing
    node expansions; Pytrich tasks typically value guidance.)

    For each goal (on c s):
      - 0 if already true.
      - Otherwise, a base cost of 2 (need at least Lift+Drop or Unload+Drop).
      - If c is currently in some truck: add 1 (need Unload before Drop).
      - If c is not at the same place as s (or either location unknown): add 2 (need at least Drive +
        one of Load/Unload; we keep it conservative/cheap).
      - If target surface s is not clear: add 3 (must clear s: Lift+Load+possibly Drive). We keep it
        as a constant penalty since computing exact clearing depth is expensive on large instances.
      - If crate c is not clear: add 1 (some crate must be removed from on top of c).

    Additionally, we incorporate task-network guidance:
      - If there are pending abstract tasks named 'do_put_on', we add the per-task estimate above.
        (In these instances, the task list mirrors the goal set, so this helps even before goals are reached.)

    Efficiency
    ----------
    - Uses only bitwise state checks with cached global fact ids.
    - Parses and indexes only predicates needed: on, at, in, clear.
    - Precomputes:
        * goal on-pairs
        * for each crate, possible goal target surfaces
        * mappings from facts to global ids
        * adjacency: 'on x y' relations to infer clear-violations quickly

    The heuristic is robust to partial/unknown information: if an object's place is unknown from the
    current state's +at facts, we assume transport may be needed (adds a small penalty).
    """

    def __init__(self, model, depots_gpt52_heuristic_2: str = "depots_gpt52_heuristic_2"):
        super().__init__(model, depots_gpt52_heuristic_2)
        self.model = model

        # --- Fact id caches ---
        self._fact_id: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Predicate -> list of (fact_name, args, global_id)
        self._pred_index: Dict[str, List[Tuple[str, Tuple[str, ...], int]]] = {}
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            self._pred_index.setdefault(pred, []).append((f.name, tuple(args), f.global_id))

        # Pre-store useful predicate entries
        self._on_facts = self._pred_index.get("on", [])
        self._at_facts = self._pred_index.get("at", [])
        self._in_facts = self._pred_index.get("in", [])
        self._clear_facts = self._pred_index.get("clear", [])

        # --- Goal extraction ---
        self._goal_on_pairs: List[Tuple[str, str]] = []  # (crate, surface)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == "on" and len(args) == 2:
                self._goal_on_pairs.append((args[0], args[1]))

        # Cache goal fact ids for quick goal-true checks
        self._goal_on_fact_ids: List[int] = []
        for c, s in self._goal_on_pairs:
            fid = self._fact_id.get(f"+on[{c},{s}]")
            if fid is not None:
                self._goal_on_fact_ids.append(fid)

        # For each object x, track possible places from grounded at-facts (used as fallback)
        self._objects_places: Dict[str, Set[str]] = {}
        for _, args, _ in self._at_facts:
            if len(args) == 2:
                self._objects_places.setdefault(args[0], set()).add(args[1])

        # For each crate, track possible trucks (from grounded in-facts)
        self._crate_trucks: Dict[str, Set[str]] = {}
        for _, args, _ in self._in_facts:
            if len(args) == 2:
                self._crate_trucks.setdefault(args[0], set()).add(args[1])

        # Cache clear fact ids per surface
        self._clear_fact_id: Dict[str, int] = {}
        for _, args, gid in self._clear_facts:
            if len(args) == 1:
                self._clear_fact_id[args[0]] = gid

        # Cache +at[obj,place] fact ids for quick lookup
        self._at_fact_id: Dict[Tuple[str, str], int] = {}
        for _, args, gid in self._at_facts:
            if len(args) == 2:
                self._at_fact_id[(args[0], args[1])] = gid

        # Cache +in[crate,truck] fact ids
        self._in_fact_id: Dict[Tuple[str, str], int] = {}
        for _, args, gid in self._in_facts:
            if len(args) == 2:
                self._in_fact_id[(args[0], args[1])] = gid

        # Cache +on[x,y] fact ids
        self._on_fact_id: Dict[Tuple[str, str], int] = {}
        # Also build reverse index: what is on top of a given surface
        self._on_top_of: Dict[str, List[str]] = {}
        for _, args, gid in self._on_facts:
            if len(args) == 2:
                x, y = args
                self._on_fact_id[(x, y)] = gid
                self._on_top_of.setdefault(y, []).append(x)

    # ---------- Core heuristic API ----------

    def __call__(self, node) -> int:
        state = node.state
        task_network = getattr(node, "task_network", [])

        h = 0

        # 1) Task-network contribution: estimate for pending do_put_on tasks
        #    This helps guide hierarchical refinement before the corresponding goal facts are explicitly achieved.
        for t in task_network:
            name = getattr(t, "name", "")
            if "do_put_on" in name:
                # Task name in grounded HTN is typically like 'do_put_on crateX palletY' or similar.
                # We try to parse arguments from the name by looking for tokens; if it fails, skip.
                c, s = self._parse_task_args_do_put_on(name)
                if c is not None and s is not None:
                    h += self._estimate_put_on_cost(state, c, s)
                else:
                    h += 1  # minimal encouragement to refine

        # 2) Goal contribution: sum per goal +on[c,s] not yet satisfied
        #    (Even if tasks are empty, this keeps heuristic informative for goal-directed search.)
        for c, s in self._goal_on_pairs:
            h += self._estimate_put_on_cost(state, c, s)

        return int(h)

    # ---------- Domain-specific estimators ----------

    def _estimate_put_on_cost(self, state: int, c: str, s: str) -> int:
        """Estimate remaining effort to achieve +on[c,s] from current state."""
        # Already achieved?
        fid_goal = self._on_fact_id.get((c, s))
        if fid_goal is not None and self._is_true(state, fid_goal):
            return 0

        cost = 2  # at least Lift+Drop (or Unload+Drop)

        # If crate is in a truck, need Unload before Drop.
        if self._crate_in_any_truck(state, c):
            cost += 1

        # If target surface not clear, must clear it.
        if not self._is_clear(state, s):
            cost += 3

        # If crate itself not clear, must clear it (remove something on it).
        if not self._is_clear(state, c):
            cost += 1

        # If crate and surface are not co-located, likely need transport.
        p_c = self._get_location(state, c)
        p_s = self._get_location(state, s)
        if p_c is None or p_s is None:
            # Unknown -> assume some transport may be needed
            cost += 1
        elif p_c != p_s:
            cost += 2

        return cost

    # ---------- Fast state queries (bitwise) ----------

    @staticmethod
    def _parse_fact_name(name: str):
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

    @staticmethod
    def _is_true(state: int, fact_gid: int) -> bool:
        return ((state >> fact_gid) & 1) == 1

    def _is_clear(self, state: int, surface: str) -> bool:
        gid = self._clear_fact_id.get(surface)
        if gid is None:
            # If clear-fact not present in grounding, assume not clear to stay informative.
            return False
        return self._is_true(state, gid)

    def _get_location(self, state: int, obj: str) -> Optional[str]:
        # Scan possible places for this object and return the one that is true.
        places = self._objects_places.get(obj)
        if not places:
            return None
        for p in places:
            gid = self._at_fact_id.get((obj, p))
            if gid is not None and self._is_true(state, gid):
                return p
        return None

    def _crate_in_any_truck(self, state: int, crate: str) -> bool:
        trucks = self._crate_trucks.get(crate)
        if not trucks:
            return False
        for t in trucks:
            gid = self._in_fact_id.get((crate, t))
            if gid is not None and self._is_true(state, gid):
                return True
        return False

    # ---------- Task parsing helpers ----------

    def _parse_task_args_do_put_on(self, task_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort parse for a grounded task name containing 'do_put_on'.

        Pytrich ground task names are typically space-separated, e.g.:
          'do_put_on crate1 pallet1'
        but may also include parentheses or ids depending on exporter.
        """
        # Normalize
        s = task_name.replace("(", " ").replace(")", " ").replace(",", " ")
        tokens = [tok for tok in s.split() if tok]
        if "do_put_on" not in tokens:
            return None, None
        idx = tokens.index("do_put_on")
        if idx + 2 >= len(tokens):
            return None, None
        return tokens[idx + 1], tokens[idx + 2]
