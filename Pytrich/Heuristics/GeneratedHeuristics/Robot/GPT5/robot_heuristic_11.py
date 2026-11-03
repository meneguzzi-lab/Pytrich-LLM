from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set

# Flexible import of the framework's base Heuristic class
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic11(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Core idea (admissible, state-aware, hierarchy-aware via task semantics):
    - Each package not at its goal location requires, at minimum:
        * One pickup and one putdown action (2 primitive actions total),
          except for a package currently held: it only needs the putdown (1 action).
        * A number of move actions at least the shortest path length (in edges) from
          its current room to its goal room. For a held package, this is from the
          current robot location to the package goal.
        * A number of opens at least the minimal number of CLOSED doors that must be
          opened along that path (0-1 BFS over the door graph w.r.t. current state's
          closed[...] bits). Since opening a door persists, summing opens over all
          packages would be inadmissible; we take the maximum across required segments
          as a safe lower bound.
    - Additionally, when the robot is not holding any package, to start progressing
      it must first reach some package that is not at its goal. We add the minimal
      robot-to-package shortest-path distance (in edges) as an extra move lower bound,
      and also consider the minimal number of closed doors along that initial reach.

    Final heuristic h(s) =
        sum_over_packages[ required_moves(curr->goal) ]
      + (1 if holding else 0) * required_moves(robot_loc->held_goal)
      + (0 if holding else min_robot_to_unsat_pkg_moves)
      + action_count (pickups/putdowns: 2 per unsatisfied, except held package counts 1)
      + max_required_opens_across_considered_segments

    All components are lower bounds, summed or maximized in an admissible way.

    Implementation details:
    - Facts are accessed via bitwise checks using global_id bits.
    - Goals (target in[obj,loc]) are parsed from model.goals using model.state_explicit_repr.
    - Door graph is built from 'door[loc1,loc2,d]' facts; each edge stores the global_id of 'closed[d]'.
    - All-pairs unweighted distances between rooms are precomputed (ignoring door closed/open), as the
      minimal number of move actions does not depend on closed status (open adds separate actions).
    - Minimal number of opens along a path is computed per query with a 0-1 BFS using current closed bits;
      results are cached, keyed by (closed_signature, src, dst) where closed_signature = state & closed_mask.

    Notes:
    - This heuristic is computationally efficient for the provided problem scales (<= ~50 rooms, ~100 packages).
    - It leverages domain structure (doors/open/move/pickup/putdown semantics), remains admissible, and
      uses only bitwise-state operations.

    Constructor parameter name requirement: 'robot_gpt5_heuristic_11'.
    """

    def __init__(self, model, robot_gpt5_heuristic_11: bool = True):
        super().__init__(model)

        self.enabled = robot_gpt5_heuristic_11

        # Internal structures
        self._fact_id_by_key: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        self._rooms: Set[str] = set()
        self._packages: Set[str] = set()

        # Maps and indices for quick state queries
        self._in_fact_ids_by_object: Dict[str, Dict[str, int]] = defaultdict(dict)  # obj -> {room: fact_id}
        self._holding_fact_id_by_object: Dict[str, int] = {}
        self._rloc_fact_id_by_room: Dict[str, int] = {}

        # Door graph: room -> list of (neighbor_room, closed_fact_id)
        self._edges_out: Dict[str, List[Tuple[str, Optional[int]]]] = defaultdict(list)
        self._closed_fact_ids: Set[int] = set()
        self._closed_bits_mask: int = 0

        # Goal map: object -> goal_room (parsed from model.goals' +in[obj,goal] facts)
        self._obj_goal_room: Dict[str, str] = {}

        # All-pairs unweighted room distance (number of move actions ignoring door closed/open)
        self._dist_unweighted: Dict[str, Dict[str, int]] = {}

        # Cache for 0-1 BFS results: (closed_signature, src, dst) -> min_opens
        self._min_closed_cache: Dict[Tuple[int, str, str], int] = {}

        # Parse and precompute
        self._index_facts()
        self._build_goal_map_from_goals()
        self._build_graph()
        self._precompute_all_pairs_unweighted()

    # ------------------ Parsing and indexing ------------------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
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

    def _index_facts(self) -> None:
        # Build fact key -> id mapping and collect domain symbols
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            key = (pred, tuple(args))
            self._fact_id_by_key[key] = fact.global_id

        # Scan once to collect objects, rooms, and build index maps
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "in" and len(args) == 2:
                obj, room = args
                self._packages.add(obj)
                self._rooms.add(room)
                self._in_fact_ids_by_object[obj][room] = fact.global_id

            elif pred == "rloc" and len(args) == 1:
                room = args[0]
                self._rooms.add(room)
                self._rloc_fact_id_by_room[room] = fact.global_id

            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                self._packages.add(obj)
                self._holding_fact_id_by_object[obj] = fact.global_id

            elif pred == "door" and len(args) == 3:
                # rooms gathered when building the graph; keep the names available
                loc1, loc2, d = args
                self._rooms.add(loc1)
                self._rooms.add(loc2)

            elif pred == "goal_in" and len(args) == 2:
                # May be used as a fallback if goals parsing fails
                obj, room = args
                self._packages.add(obj)
                self._rooms.add(room)

        # Precompute mask for all closed[...] facts
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "closed" and len(args) == 1:
                self._closed_fact_ids.add(fact.global_id)
                self._closed_bits_mask |= (1 << fact.global_id)

    def _build_goal_map_from_goals(self) -> None:
        # Build object -> goal_room from goal state's explicit representation (list of '+in[obj,room]' facts)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == "in" and len(args) == 2:
                obj, goal_room = args
                self._obj_goal_room[obj] = goal_room
        # Fallback: if empty (should not happen), try goal_in from facts
        if not self._obj_goal_room:
            for fact in self.model.facts:
                pred, args = self._parse_fact_name(fact.name)
                if pred == "goal_in" and len(args) == 2:
                    obj, room = args
                    self._obj_goal_room[obj] = room

    def _build_graph(self) -> None:
        # Build directed graph edges from door facts, with per-edge closed[...] fact id
        # For each door(loc1,loc2,d), find corresponding closed(d) fact id (may or may not be true at runtime)
        def get_closed_id(door_name: str) -> Optional[int]:
            key = ("closed", (door_name,))
            return self._fact_id_by_key.get(key, None)

        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == "door" and len(args) == 3:
                loc1, loc2, d = args
                closed_id = get_closed_id(d)
                self._edges_out[loc1].append((loc2, closed_id))

        # Ensure all rooms appear in adjacency, even if no outgoing edges (defensive)
        for r in list(self._rooms):
            _ = self._edges_out[r]  # touch to create key

    def _precompute_all_pairs_unweighted(self) -> None:
        # BFS from each room over directed edges to compute minimal number of moves (edges)
        rooms = list(self._rooms)
        self._dist_unweighted = {r: {rr: (0 if r == rr else 10 ** 9) for rr in rooms} for r in rooms}

        for src in rooms:
            dist = {r: 10 ** 9 for r in rooms}
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                du = dist[u]
                for v, _closed_id in self._edges_out.get(u, []):
                    if dist[v] > du + 1:
                        dist[v] = du + 1
                        q.append(v)
            self._dist_unweighted[src] = dist

    # ------------------ Bitwise state queries ------------------
    @staticmethod
    def _is_true(state_bitwise: int, fact_id: int) -> bool:
        return (state_bitwise >> fact_id) & 1 == 1

    def _get_robot_location(self, state_bitwise: int) -> Optional[str]:
        # Exactly one rloc[room] should be true
        for room, fid in self._rloc_fact_id_by_room.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _get_holding_object(self, state_bitwise: int) -> Optional[str]:
        # At most one holding[obj] can be true
        for obj, fid in self._holding_fact_id_by_object.items():
            if self._is_true(state_bitwise, fid):
                return obj
        return None

    def _get_object_location(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Returns the unique room such that in[obj,room] is true; None if not found (e.g., when held)
        for room, fid in self._in_fact_ids_by_object.get(obj, {}).items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    # ------------------ 0-1 BFS for minimal opens ------------------
    def _min_closed_opens_between(self, state_bitwise: int, src: str, dst: str) -> int:
        if src == dst:
            return 0
        # Cache key based on closed-door signature (only closed[...] facts matter)
        closed_signature = state_bitwise & self._closed_bits_mask
        cache_key = (closed_signature, src, dst)
        cached = self._min_closed_cache.get(cache_key)
        if cached is not None:
            return cached

        # 0-1 BFS: edge weight = 1 if door is currently closed, else 0
        dist: Dict[str, int] = {r: 10 ** 9 for r in self._rooms}
        dist[src] = 0
        dq: deque = deque([src])

        while dq:
            u = dq.popleft()
            du = dist[u]
            if u == dst:
                break
            for v, closed_fid in self._edges_out.get(u, []):
                w = 1 if (closed_fid is not None and self._is_true(state_bitwise, closed_fid)) else 0
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)

        res = dist.get(dst, 10 ** 9)
        if res == 10 ** 9:
            res = 0  # unreachable => return 0 as a safe (very weak) admissible bound
        self._min_closed_cache[cache_key] = res
        return res

    # ------------------ Heuristic evaluation ------------------
    def evaluate(self, state_bitwise: int, task_network: List = None) -> int:
        if not self.enabled:
            return 0

        # Gather current situation
        robot_room = self._get_robot_location(state_bitwise)
        holding_obj = self._get_holding_object(state_bitwise)

        # Defensive guards
        if robot_room is None:
            # If unknown, we cannot estimate movement accurately; return 0 (admissible, but weak)
            return 0

        # Action lower bound: pickups/putdowns
        action_lb = 0
        # Movement lower bound: sum of required carries (and initial approach)
        move_lb = 0
        # Open-door lower bound: maximum across relevant segments
        opens_lb = 0

        # Build list of unsatisfied packages (excluding the held one which we handle specially)
        unsat_pkgs: List[Tuple[str, str, str]] = []  # (obj, curr_room, goal_room)
        for obj in self._obj_goal_room.keys():
            goal_room = self._obj_goal_room[obj]
            if obj == holding_obj:
                continue  # handled separately
            curr_room = self._get_object_location(state_bitwise, obj)
            if curr_room is None:
                # If not found and not the held object, be conservative: skip (treat as satisfied)
                continue
            if curr_room != goal_room:
                unsat_pkgs.append((obj, curr_room, goal_room))

        # Handle held object specially
        if holding_obj is not None:
            goal_room = self._obj_goal_room.get(holding_obj)
            if goal_room is not None:
                # Must at least put it down
                action_lb += 1
                # Must move at least distance from robot's current room to goal_room
                d = self._dist_unweighted.get(robot_room, {}).get(goal_room, 10 ** 9)
                if d == 10 ** 9:
                    d = 0
                move_lb += d
                # Opens lower bound along this segment
                opens_lb = max(opens_lb, self._min_closed_opens_between(state_bitwise, robot_room, goal_room))
        
        # For other unsatisfied packages: each needs pickup + putdown (2 actions) and carry distance curr->goal
        for _obj, curr_room, goal_room in unsat_pkgs:
            action_lb += 2
            d = self._dist_unweighted.get(curr_room, {}).get(goal_room, 10 ** 9)
            if d == 10 ** 9:
                d = 0
            move_lb += d
            opens_lb = max(opens_lb, self._min_closed_opens_between(state_bitwise, curr_room, goal_room))

        # Initial approach when not holding: need to reach at least one unsatisfied package to pick it up
        if holding_obj is None and unsat_pkgs:
            # Minimal robot->package distance over all unsatisfied packages
            best_d = 10 ** 9
            best_opens = 10 ** 9
            for _obj, curr_room, _goal_room in unsat_pkgs:
                d = self._dist_unweighted.get(robot_room, {}).get(curr_room, 10 ** 9)
                if d < best_d:
                    best_d = d
                # minimal opens along approach paths (we will take min across candidates)
                o = self._min_closed_opens_between(state_bitwise, robot_room, curr_room)
                if o < best_opens:
                    best_opens = o
            if best_d == 10 ** 9:
                best_d = 0
            move_lb += best_d
            opens_lb = max(opens_lb, best_opens)

        # Heuristic is sum of actions + moves + (max)opens lower bounds
        h_val = action_lb + move_lb + opens_lb
        if h_val < 0:
            h_val = 0
        return int(h_val)

    # Expose common alias used by some planners
    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.evaluate(state_bitwise, task_network)


# Alias compliant with potential dynamic loading conventions
def get_heuristic(model):
    return RobotGPT5Heuristic11(model)
