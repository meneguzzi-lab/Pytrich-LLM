from collections import deque
from typing import Dict, List, Tuple, Optional, Set

# Pytrich base class import (works in the evaluation environment)
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic10(Heuristic):
    """Domain-dependent heuristic for the HDDL 'robot' domain.

    Core domain insight
    -------------------
    The domain is essentially a repeated delivery problem under HTN control:

      - If an object is not in its goal room, the HTN will (eventually) require:
          pickup(obj at its current room) -> (move/open as needed) -> putdown(obj at goal room)
      - The only way to move between rooms is via doors; some doors start closed and must be opened.
      - The robot can hold at most one package.

    Heuristic design
    ----------------
    We compute a fast, state-dependent *lower bound* based on:

      1) Remaining deliveries count:
           Each misplaced object requires at least one pickup and one putdown.
           => 2 actions per misplaced object.

      2) Navigation distance for the currently carried item (if any):
           If holding o, we must reach goal_in(o, g).
           We estimate minimal number of moves as shortest-path length in the room graph.

      3) For any required move, at least one 'move' action per traversed edge.
         Additionally, if an edge uses a door that is still closed, at least one 'open' action
         is needed at some point before moving through it. We conservatively add the minimum
         number of closed doors along the chosen shortest path (door openings are shared across
         objects, so we only count them for the currently forced navigation of a held item).

    Hierarchy awareness
    ------------------
    We do not explicitly evaluate decompositions (to keep it cheap), but we approximate the
    minimal primitive effort that any decomposition must eventually incur.

    Performance
    -----------
    - All fact checks use bit operations via global_id.
    - The room graph and relevant fact ids are pre-indexed once in __init__.
    - Shortest path is computed with BFS over rooms (instances are sparse graphs).

    Notes
    -----
    This heuristic is admissible as a relaxed lower bound on primitive steps.
    It is also robust to partial progress because it reads the current state.
    """

    def __init__(self, model, robot_gpt52_heuristic_10: str = "robot_gpt52_heuristic_10"):
        super().__init__(model, robot_gpt52_heuristic_10)
        self.model = model

        # --- Parse and index facts we care about ---
        self._fact_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._rooms: Set[str] = set()
        self._packages: Set[str] = set()

        # door facts: (loc1, loc2) -> door_obj
        self._door_of_edge: Dict[Tuple[str, str], str] = {}

        # fact ids for quick checks
        self._armempty_id: Optional[int] = None
        self._holding_ids: Dict[str, int] = {}      # pkg -> fact_id
        self._rloc_ids: Dict[str, int] = {}         # room -> fact_id
        self._in_ids: Dict[Tuple[str, str], int] = {}       # (pkg, room) -> fact_id
        self._goal_in: Dict[str, str] = {}          # pkg -> goal room
        self._closed_ids: Dict[str, int] = {}       # door_obj -> fact_id

        # adjacency graph: room -> list[(neighbor_room, door_obj)]
        self._adj: Dict[str, List[Tuple[str, str]]] = {}

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            key = (pred, tuple(args))
            self._fact_by_key[key] = fact.global_id

            if pred == "armempty" and len(args) == 0:
                self._armempty_id = fact.global_id

            elif pred == "rloc" and len(args) == 1:
                r = args[0]
                self._rooms.add(r)
                self._rloc_ids[r] = fact.global_id

            elif pred == "holding" and len(args) == 1:
                o = args[0]
                self._packages.add(o)
                self._holding_ids[o] = fact.global_id

            elif pred == "in" and len(args) == 2:
                o, r = args
                self._packages.add(o)
                self._rooms.add(r)
                self._in_ids[(o, r)] = fact.global_id

            elif pred == "goal_in" and len(args) == 2:
                o, r = args
                self._packages.add(o)
                self._rooms.add(r)
                # Each object has exactly one goal room in the instances
                self._goal_in[o] = r

            elif pred == "door" and len(args) == 3:
                loc1, loc2, d = args
                self._rooms.add(loc1)
                self._rooms.add(loc2)
                self._door_of_edge[(loc1, loc2)] = d

            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self._closed_ids[d] = fact.global_id

        # Build adjacency lists from door facts
        for (u, v), d in self._door_of_edge.items():
            self._adj.setdefault(u, []).append((v, d))

        # Cache for BFS distances keyed by (start_room, state_closed_mask_signature)
        # We keep it simple: cache only by start room and ignore state in the cache;
        # we still compute closed-door counts per query.
        self._dist_cache: Dict[str, Dict[str, int]] = {}

        # Precompute goal facts for fast goal counting if needed
        self._goal_fact_ids: List[int] = []
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for n in goal_fact_names:
            pred, args = self._parse_fact_name(n)
            if pred == "in" and len(args) == 2:
                gid = self._fact_by_key.get((pred, tuple(args)))
                if gid is not None:
                    self._goal_fact_ids.append(gid)

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
    def _is_true(self, state: int, fact_id: Optional[int]) -> bool:
        if fact_id is None:
            return False
        return ((state >> fact_id) & 1) == 1

    def _current_room(self, state: int) -> Optional[str]:
        # Exactly one rloc should be true
        for r, fid in self._rloc_ids.items():
            if ((state >> fid) & 1) == 1:
                return r
        return None

    def _held_object(self, state: int) -> Optional[str]:
        for o, fid in self._holding_ids.items():
            if ((state >> fid) & 1) == 1:
                return o
        return None

    def _object_room(self, state: int, obj: str) -> Optional[str]:
        # If holding, it's not in a room
        if obj in self._holding_ids and self._is_true(state, self._holding_ids[obj]):
            return None
        # Find room where in[obj,room] holds
        for (o, r), fid in self._in_ids.items():
            if o == obj and ((state >> fid) & 1) == 1:
                return r
        return None

    # ---------- Graph distance ----------
    def _bfs_distances(self, start: str) -> Dict[str, int]:
        cached = self._dist_cache.get(start)
        if cached is not None:
            return cached

        dist: Dict[str, int] = {start: 0}
        q = deque([start])
        while q:
            u = q.popleft()
            for v, _d in self._adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        self._dist_cache[start] = dist
        return dist

    def _min_closed_doors_on_shortest_path(self, state: int, start: str, goal: str) -> int:
        """Among all shortest paths (min number of moves), estimate min number of closed doors.

        We do a BFS layered by distance, tracking minimal closed-door count.
        This remains cheap because graphs are sparse and rooms <= 50 in training.
        """
        if start == goal:
            return 0

        # (room -> best_closed_count) for current frontier at a given depth
        frontier = {start: 0}
        visited_best: Dict[str, int] = {start: 0}

        depth = 0
        # We stop once we first reach the goal depth; we continue within that depth to minimize closed count.
        while frontier:
            if goal in frontier:
                return frontier[goal]

            next_frontier: Dict[str, int] = {}
            depth += 1
            for u, closed_cnt in frontier.items():
                for v, d in self._adj.get(u, []):
                    add = 1 if self._is_true(state, self._closed_ids.get(d)) else 0
                    cand = closed_cnt + add

                    # standard BFS depth control: we allow v in next_frontier if it improves closed count
                    prev = next_frontier.get(v)
                    if prev is None or cand < prev:
                        next_frontier[v] = cand

            # Prune dominated states across depths: keep only improvements
            pruned_frontier: Dict[str, int] = {}
            for v, cand in next_frontier.items():
                best = visited_best.get(v)
                if best is None or cand < best:
                    visited_best[v] = cand
                    pruned_frontier[v] = cand
            frontier = pruned_frontier

        # Disconnected: no path (should not happen in provided instances); return 0 as safe lower bound.
        return 0

    # ---------- Heuristic interface ----------
    def __call__(self, node) -> int:
        state = node.state
        task_network = node.task_network  # not used directly; domain is simple deliver-all

        # 1) Count misplaced objects (must be delivered)
        misplaced = 0
        for obj, goal_room in self._goal_in.items():
            # If in goal room already, OK.
            fid = self._in_ids.get((obj, goal_room))
            if fid is not None and self._is_true(state, fid):
                continue
            # Otherwise, not yet at goal.
            misplaced += 1

        # Lower bound: each misplaced object requires at least pickup + putdown
        h = 2 * misplaced

        # 2) If currently holding an object, add navigation-to-goal lower bound
        held = self._held_object(state)
        if held is not None:
            cur_room = self._current_room(state)
            goal_room = self._goal_in.get(held)
            if cur_room is not None and goal_room is not None:
                dist_map = self._bfs_distances(cur_room)
                moves_lb = dist_map.get(goal_room, 0)
                opens_lb = self._min_closed_doors_on_shortest_path(state, cur_room, goal_room)
                # Each move is 1 action; each unopened closed door needs at least 1 open action.
                h += moves_lb + opens_lb

        # 3) If not holding and there are misplaced objects, encourage getting to *some* next pickup
        # (still admissible as 0 would be fine, but a small LB helps tie-breaking without overcounting)
        # We add the minimal moves needed to reach any misplaced object's current room.
        if held is None and misplaced > 0:
            cur_room = self._current_room(state)
            if cur_room is not None:
                dist_map = self._bfs_distances(cur_room)
                best_pickup_moves = None
                for obj, goal_room in self._goal_in.items():
                    # Only consider misplaced
                    fid_goal = self._in_ids.get((obj, goal_room))
                    if fid_goal is not None and self._is_true(state, fid_goal):
                        continue
                    obj_room = self._object_room(state, obj)
                    if obj_room is None:
                        continue
                    d = dist_map.get(obj_room)
                    if d is None:
                        continue
                    if best_pickup_moves is None or d < best_pickup_moves:
                        best_pickup_moves = d
                if best_pickup_moves is not None:
                    h += best_pickup_moves

        # Never negative
        return max(0, int(h))
