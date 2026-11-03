from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any

try:
    # Pytrich style
    from Pytrich.Heuristics.heuristic import Heuristic
except Exception:
    # Fallback for other package layouts
    from heuristics.heuristic import Heuristic


class RoverGPT52Heuristic13(Heuristic):
    """Domain-dependent heuristic for the HDDL Rover domain.

    Design rationale (domain insights -> heuristic components)
    ---------------------------------------------------------
    The training instances show a classic Rover benchmark pattern:
      * Many independent high-level tasks (get_soil_data wp), (get_rock_data wp), (get_image_data o m).
      * Each such task typically requires a navigation to a relevant waypoint, then a sensing action
        (sample/take_image), then communication to a lander-visible waypoint.
      * Navigation is the dominant cost driver on large instances; but computing true shortest paths per
        state is expensive.

    This heuristic provides a *fast* and *useful* lower bound by combining:

    (A) Goal-fact counting: remaining goal facts not yet true. This is admissible as a unit-cost
        relaxation lower bound (each missing goal fact requires at least one action to add it).

    (B) Communication feasibility + minimal motion proxy:
        For each missing communicated_* goal, we estimate at least:
          - 1 action to communicate,
          - plus a lower bound on navigation steps required to reach *some* waypoint from which
            communication is possible (visibility to a lander waypoint).
        We approximate navigation steps by BFS distances over the rover's can_traverse graph.
        Crucially, we compute BFS only from the *current rover positions* to the subset of waypoints
        that can communicate to the lander.

    (C) Image calibration requirement proxy:
        If an image communication goal is missing and the rover currently has no matching have_image,
        then at least one take_image is needed (1), and in this domain take_image consumes calibration.
        We include a small constant lower bound for the imaging acquisition chain (calibrate+take_image)
        but still keep it admissible by only charging when the corresponding enabling facts are
        definitely missing.

    Hierarchy handling:
    -------------------
    Pytrich is HTN: nodes include a task network. We incorporate the task network in a lightweight way:
      * For each remaining abstract task in the task network, add a cheap admissible estimate of the
        minimal primitive actions needed for *any* decomposition.

    Efficiency:
    -----------
    * No state->explicit fact list conversion.
    * All checks use bitwise membership via global_id.
    * BFS graphs built once from static facts (can_traverse).

    Notes on admissibility:
    -----------------------
    This heuristic is intended as a *lower bound*. We only add costs that are certainly necessary:
      * missing goal facts >= 1 action each (unit-cost assumption matches typical rover instances)
      * to communicate you must execute a communicate_* action (>=1)
      * if you are not at any communication waypoint, you must navigate at least the shortest-path
        number of navigate actions to reach one.
      * abstract tasks: we use minimal primitive counts across methods (safe lower bounds).

    If operator costs are not uniform, the heuristic still remains a lower bound as long as all costs
    are >= 1. (This holds in the provided domain where actions have implicit unit costs in benchmarks.)
    """

    def __init__(self, model, rover_gpt52_heuristic_13: str = "rover_gpt52_heuristic_13"):
        super().__init__(model, rover_gpt52_heuristic_13)
        self.model = model

        # --- Fact indexing helpers
        self._fact_name_to_gid: Dict[str, int] = {f.name: f.global_id for f in self.model.facts}

        # Parse all fact names once for static analysis
        self._parsed_fact: Dict[int, Tuple[str, Tuple[str, ...], bool]] = {}
        # gid -> (pred, args, is_positive)
        for fact in self.model.facts:
            pred, args, sign = self._parse_fact(fact.name)
            self._parsed_fact[fact.global_id] = (pred, args, sign)

        # --- Static world structures
        self._lander_at: List[Tuple[str, str]] = []  # (lander, waypoint)
        self._visible_pairs: Set[Tuple[str, str]] = set()  # (w1, w2)
        self._can_traverse_edges: Dict[str, Set[str]] = defaultdict(set)  # w1 -> {w2}

        # --- Camera / imaging capabilities
        self._camera_on_board: Dict[str, str] = {}  # camera -> rover
        self._camera_supports: Dict[str, Set[str]] = defaultdict(set)  # camera -> {mode}
        self._camera_calib_target: Dict[str, str] = {}  # camera -> objective

        # --- Objective visibility
        self._objective_visible_from: Dict[str, Set[str]] = defaultdict(set)  # objective -> {waypoint}

        # --- Rover equipment predicates (static from init, but still facts; we treat as checkable per state)
        # We'll just look them up by predicate during evaluation.

        # Gather from initial state's facts (static environment predicates are usually invariant)
        # But we can safely gather from *all* facts list by predicate/args; these are domain atoms.
        for gid, (pred, args, is_pos) in self._parsed_fact.items():
            if not is_pos:
                continue
            if pred == "at_lander" and len(args) == 2:
                self._lander_at.append((args[0], args[1]))
            elif pred == "visible" and len(args) == 2:
                self._visible_pairs.add((args[0], args[1]))
            elif pred == "can_traverse" and len(args) == 3:
                # can_traverse[rover, w1, w2] is rover-dependent; for nav LB we ignore rover id and
                # union edges (admissible: more edges can only reduce distance).
                self._can_traverse_edges[args[1]].add(args[2])
            elif pred == "on_board" and len(args) == 2:
                self._camera_on_board[args[0]] = args[1]
            elif pred == "supports" and len(args) == 2:
                self._camera_supports[args[0]].add(args[1])
            elif pred == "calibration_target" and len(args) == 2:
                self._camera_calib_target[args[0]] = args[1]
            elif pred == "visible_from" and len(args) == 2:
                self._objective_visible_from[args[0]].add(args[1])

        # Precompute set of lander waypoints
        self._lander_waypoints: Set[str] = {wp for _, wp in self._lander_at}

        # Precompute communication-capable waypoints:
        # a rover at p1 can communicate if visible(p1, lander_wp) for some lander_wp.
        self._comm_waypoints: Set[str] = set()
        for (p1, p2) in self._visible_pairs:
            if p2 in self._lander_waypoints:
                self._comm_waypoints.add(p1)

        # Also if rover is at lander waypoint itself and visible is required (visible p1 p2);
        # datasets usually include visible(wp, wp2) not reflexive, so keep as above.

        # --- Task-based lower bounds (minimal primitive steps per abstract task)
        # Extract by name patterns used after grounding in Pytrich:
        # Abstract task names are typically like "(get_soil_data waypoint0)" or similar;
        # but we avoid parsing that and instead use the AbstractTask.name field.
        self._abt_min_steps: Dict[int, int] = {}
        for abt in self.model.abstract_tasks:
            n = abt.name
            # Safe minimal primitive action counts implied by the domain methods:
            # do_navigate1: at least 0 (if already there) else >=1 navigate (but we can't know)
            # empty_store: 0 or 1
            # send_*: at least 1 communicate and maybe navigation
            # get_*: includes navigation, maybe drop, sample/take, send
            if "get_soil_data" in n:
                self._abt_min_steps[abt.global_id] = 2  # sample_soil + communicate (nav could be 0)
            elif "get_rock_data" in n:
                self._abt_min_steps[abt.global_id] = 2
            elif "get_image_data" in n:
                self._abt_min_steps[abt.global_id] = 2  # take_image + communicate (calibrate/nav could be 0)
            elif "send_soil_data" in n or "send_rock_data" in n or "send_image_data" in n:
                self._abt_min_steps[abt.global_id] = 1
            elif "do_calibrate" in n:
                self._abt_min_steps[abt.global_id] = 1
            elif "do_navigate" in n:
                self._abt_min_steps[abt.global_id] = 0
            elif "empty_store" in n:
                self._abt_min_steps[abt.global_id] = 0
            else:
                self._abt_min_steps[abt.global_id] = 0

        # Cache for BFS distances per (start_wp) to comm_waypoints
        self._bfs_cache_to_comm: Dict[str, int] = {}

        # Pre-cache goal facts parsed
        self._goal_fact_gids: List[int] = []
        self._goal_fact_info: List[Tuple[str, Tuple[str, ...]]] = []  # (pred, args)
        for fname in self.model.state_explicit_repr(self.model.goals):
            gid = self._fact_name_to_gid.get(fname)
            if gid is None:
                continue
            pred, args, is_pos = self._parse_fact(fname)
            if is_pos:
                self._goal_fact_gids.append(gid)
                self._goal_fact_info.append((pred, args))

    # ------------------------ parsing utilities ------------------------
    def _parse_fact(self, name: str) -> Tuple[str, Tuple[str, ...], bool]:
        name = name.strip()
        is_pos = True
        if name.startswith("+"):
            name = name[1:]
            is_pos = True
        elif name.startswith("-"):
            name = name[1:]
            is_pos = False

        b = name.find("[")
        if b > 0 and name.endswith("]"):
            pred = name[:b]
            args_str = name[b + 1 : -1]
            if args_str:
                args = tuple(args_str.split(","))
            else:
                args = tuple()
            return pred, args, is_pos
        return name, tuple(), is_pos

    # ------------------------ bitwise state checks ------------------------
    @staticmethod
    def _is_true(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    # ------------------------ BFS distance (static graph) ------------------------
    def _min_steps_to_any_comm_waypoint(self, start_wp: str) -> int:
        """Return shortest number of traverse edges from start_wp to any comm waypoint.

        Uses a graph that unions can_traverse edges across rovers (admissible LB).
        If unreachable, returns a large number.
        """
        if start_wp in self._bfs_cache_to_comm:
            return self._bfs_cache_to_comm[start_wp]

        if start_wp in self._comm_waypoints:
            self._bfs_cache_to_comm[start_wp] = 0
            return 0

        q = deque()
        q.append(start_wp)
        dist = {start_wp: 0}

        while q:
            u = q.popleft()
            du = dist[u]
            for v in self._can_traverse_edges.get(u, ()):
                if v in dist:
                    continue
                dist[v] = du + 1
                if v in self._comm_waypoints:
                    self._bfs_cache_to_comm[start_wp] = dist[v]
                    return dist[v]
                q.append(v)

        # unreachable
        self._bfs_cache_to_comm[start_wp] = 10**6
        return 10**6

    def _extract_rover_positions(self, state_bitwise: int) -> Dict[str, str]:
        """Extract at[rover, waypoint] facts from the state (bitwise scan over set bits).

        We do a set-bit scan (fast) and pick only predicate 'at'.
        """
        rover_pos: Dict[str, str] = {}
        bits = state_bitwise
        while bits:
            lsb = bits & -bits
            gid = (lsb.bit_length() - 1)
            pred, args, is_pos = self._parsed_fact.get(gid, ("", tuple(), True))
            if is_pos and pred == "at" and len(args) == 2:
                rover_pos[args[0]] = args[1]
            bits ^= lsb
        return rover_pos

    # ------------------------ main heuristic ------------------------
    def __call__(self, parent_node, node) -> int:
        state = node.state
        task_network = node.task_network

        # (1) Missing goal facts count (admissible)
        missing_goal = 0
        missing_comm_goals: List[Tuple[str, Tuple[str, ...]]] = []

        for gid, (pred, args) in zip(self._goal_fact_gids, self._goal_fact_info):
            if not self._is_true(state, gid):
                missing_goal += 1
                if pred.startswith("communicated_"):
                    missing_comm_goals.append((pred, args))

        # (2) Task network lower bound (very lightweight)
        tn_lb = 0
        for t in task_network:
            # primitive Operator has global_id in operator range, AbstractTask has global_id in abt range
            try:
                gid = t.global_id
            except Exception:
                continue

            # If it's an abstract task, add a minimal primitive estimate
            if gid >= self.model.iabt_init and gid <= self.model.iabt_end:
                tn_lb += self._abt_min_steps.get(gid, 0)
            # If it's an operator, at least its cost (usually 1)
            elif gid >= self.model.iop_init and gid <= self.model.iop_end:
                # keep admissible: add minimal possible cost (>=0). operators in benchmarks cost 1.
                try:
                    c = getattr(t, "cost", 1)
                    if c > 0:
                        tn_lb += 0  # don't double-count; goal counting already drives search
                except Exception:
                    pass

        # (3) Communication + navigation proxy for missing communicated goals
        # We compute a lower bound to get to a comm waypoint for *some* rover.
        comm_nav_lb = 0
        if missing_comm_goals and self._comm_waypoints:
            rover_pos = self._extract_rover_positions(state)
            if rover_pos:
                best_to_comm = 10**6
                for _, wp in rover_pos.items():
                    d = self._min_steps_to_any_comm_waypoint(wp)
                    if d < best_to_comm:
                        best_to_comm = d
                if best_to_comm >= 10**6:
                    best_to_comm = 0
                # If there is at least one missing communicated goal, at least one communicate action
                # is needed AND rover must be at a comm waypoint.
                comm_nav_lb += 1
                comm_nav_lb += best_to_comm

        # (4) Imaging acquisition proxy: if communicated_image_data is missing and no have_image exists,
        # then at least 1 take_image is needed somewhere. Detecting existence of have_image for the
        # specific (o,m) is cheap via fact lookup.
        image_acq_lb = 0
        for pred, args in missing_comm_goals:
            if pred != "communicated_image_data":
                continue
            if len(args) != 2:
                continue
            o, m = args
            # If any rover already has the image, acquisition is not needed.
            # Ground fact format: +have_image[rover,objective,mode]
            have_any = False
            # We don't know which rover; check all possible have_image facts for this (o,m)
            # by scanning facts once via parsed_fact index is too expensive each call.
            # Instead, try a small scan over state set bits for have_image with matching o,m.
            bits = state
            while bits and not have_any:
                lsb = bits & -bits
                gid = (lsb.bit_length() - 1)
                p2, a2, is_pos2 = self._parsed_fact.get(gid, ("", tuple(), True))
                if is_pos2 and p2 == "have_image" and len(a2) == 3 and a2[1] == o and a2[2] == m:
                    have_any = True
                    break
                bits ^= lsb
            if not have_any:
                # lower bound: need take_image (1). calibrate may or may not be needed depending on state,
                # but we can't safely assume it's needed. So only add 1.
                image_acq_lb += 1

        # Combine with max to avoid overcounting independent relaxations too aggressively.
        # missing_goal already counts each communicated_* once; comm_nav_lb adds structural cost for
        # at least one comm chain. Using max keeps admissible and informative.
        h = max(missing_goal, comm_nav_lb)
        h = max(h, image_acq_lb)

        # Add a small TN component as additional guidance but keep admissible by max.
        h = max(h, tn_lb)

        return int(h)
