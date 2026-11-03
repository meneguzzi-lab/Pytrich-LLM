from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set

# Robust import of the base Heuristic class
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic0(Heuristic):
    """
    Domain-dependent heuristic for the HDDL Robot domain.

    Design rationale (admissible lower bound):
    - Each package that is not yet at its goal requires at least:
      - 1 pickup and 1 putdown (2 actions), except if already held (only 1 putdown).
      - Being carried from its current room to its goal room, contributing at least the
        number of move actions along a shortest path (ignoring door closures). This is a strict lower bound because
        these carrying moves are necessary and cannot be shared across different packages due to the single-arm constraint.
    - If the robot is not currently holding an object, it must at least move from its current room to the room
      of the first package it will pick up. We add the minimum move distance from the robot's location to any unfinished
      package's current room (again ignoring door closures). This deadhead travel is necessary at least once and does not
      overlap with the carrying distances already counted.
    - Door openings (open actions) are ignored for all but the held object (if any) to keep the bound admissible and cheap.
      For the currently held object, we add a minimal number of opens (0-1 BFS) along a path to its goal reflecting the
      current state's opened/closed doors. This is still admissible because these opens are mandatory for delivering the
      currently carried package.

    This heuristic balances accuracy and speed:
    - Precomputes the undirected room graph from initial door facts ignoring closure for move distances (BFS all-pairs)
    - Precomputes indices for relevant facts to allow fast bitwise state checks
    - Uses only necessary per-state scans with bitwise tests, avoiding slow explicit state representations

    Expected benefits:
    - Strong guidance on large instances by summing unavoidable carrying distances and action counts
    - Minimal overhead per node thanks to precomputation
    - Remains admissible and robust across hierarchical decompositions
    """

    def __init__(self, model, robot_gpt5_heuristic_0: Optional[dict] = None):
        super().__init__(model)
        self.robot_gpt5_heuristic_0 = robot_gpt5_heuristic_0 or {}

        # Internal structures
        self._rooms: Set[str] = set()
        self._room_to_idx: Dict[str, int] = {}
        self._idx_to_room: List[str] = []

        # Adjacency with door-closure info per edge: room_idx -> List[(nbr_idx, closed_fact_id)]
        self._adj_edges: List[List[Tuple[int, int]]] = []

        # All-pairs shortest move distances ignoring closed doors: moves only
        self._moves_dist: List[List[int]] = []

        # Fact indexes for quick bitwise checks
        self._rloc_fact_id: Dict[str, int] = {}           # +rloc[room] -> fact_id
        self._holding_fact_id: Dict[str, int] = {}        # +holding[obj] -> fact_id
        self._in_fact_id: Dict[str, Dict[str, int]] = {}  # obj -> {room -> fact_id} for +in[obj,room]
        self._closed_fact_id: Dict[str, int] = {}         # +closed[door] -> fact_id

        # Goals: obj -> goal_room
        self._goal_room_of: Dict[str, str] = {}
        self._objects: List[str] = []

        # Build all precomputations
        self._precompute_static_structures()

    # ---------- Core Heuristic API ----------
    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.h(state_bitwise, task_network or [])

    def h(self, state_bitwise: int, task_network: List) -> int:
        # If goals and tasks are both achieved, cost is zero
        if self.model.goal_reached(state_bitwise, task_network):
            return 0

        # Determine robot location
        robot_room = self._get_robot_room(state_bitwise)
        robot_room_idx = self._room_to_idx.get(robot_room, -1) if robot_room is not None else -1

        # Determine held object (if any)
        held_obj = None
        for obj, fid in self._holding_fact_id.items():
            if self._is_true(state_bitwise, fid):
                held_obj = obj
                break

        total = 0

        # Sum of carrying move distances ignoring closed doors
        # + 2 actions (pickup+putdown) for each unfinished non-held object,
        # + 1 action (putdown) for the held object, and carry distance from current robot room to goal.
        unfinished_obj_rooms: List[Tuple[str, str]] = []  # (obj, obj_current_room) for non-held, unfinished

        for obj in self._objects:
            goal_room = self._goal_room_of[obj]

            # If object already at goal room: +in[obj,goal_room] true
            in_goal_fid = self._in_fact_id.get(obj, {}).get(goal_room, None)
            if in_goal_fid is not None and self._is_true(state_bitwise, in_goal_fid):
                continue  # already done

            if held_obj == obj:
                # Held object: need to carry from current robot room to goal, and putdown once
                if robot_room is not None:
                    carry_moves = self._get_moves_dist(robot_room, goal_room)
                else:
                    carry_moves = 0
                total += carry_moves
                total += 1  # putdown

                # Add minimal opens required along any path (0-1 BFS on current closed facts)
                if robot_room_idx != -1:
                    min_opens = self._min_closed_edges_along_path(state_bitwise, robot_room_idx, self._room_to_idx[goal_room])
                    if min_opens != float('inf'):
                        total += min_opens
                continue

            # Not held: find current room of object
            obj_room = self._get_object_room(state_bitwise, obj)
            if obj_room is None:
                # Fallback: if unknown, skip (keeps admissible by not overestimating)
                continue

            # Carrying moves from its current room to goal room (lower bound)
            carry_moves = self._get_moves_dist(obj_room, goal_room)
            total += carry_moves
            # pickup + putdown
            total += 2

            # For potential deadhead calculation later
            unfinished_obj_rooms.append((obj, obj_room))

        # Initial deadhead (only if not holding): robot must at least reach some unfinished object's room once
        if held_obj is None and robot_room_idx != -1 and unfinished_obj_rooms:
            min_deadhead = float('inf')
            for _, obj_room in unfinished_obj_rooms:
                d = self._get_moves_dist(robot_room, obj_room)
                if d < min_deadhead:
                    min_deadhead = d
            if min_deadhead != float('inf'):
                # Optionally include necessary opens along the chosen deadhead path (still admissible)
                # Choose the object minimizing (moves + opens) to strengthen the bound.
                best_extra = min_deadhead
                for _, obj_room in unfinished_obj_rooms:
                    dst_idx = self._room_to_idx.get(obj_room, -1)
                    if dst_idx == -1:
                        continue
                    opens = self._min_closed_edges_along_path(state_bitwise, robot_room_idx, dst_idx)
                    moves = self._get_moves_dist(robot_room, obj_room)
                    if moves == float('inf'):
                        continue
                    cand = moves + (opens if opens != float('inf') else 0)
                    if cand < best_extra:
                        best_extra = cand
                total += int(best_extra)

        # Ensure non-negative integer
        if total < 0:
            total = 0
        return int(total)

    # ---------- Precomputation ----------
    def _precompute_static_structures(self):
        # 1) Parse all facts to index those we need quickly
        self._index_relevant_facts()

        # 2) Build goals: from model.goals bitset we can get +in[obj,room] goals
        self._build_goals_from_model_goals()

        # 3) Build room graph ignoring closed (from initial +door facts) and edge closed-fact ids
        self._build_room_graph()

        # 4) All-pairs shortest move distances ignoring closed doors
        self._precompute_all_pairs_moves()

    def _index_relevant_facts(self):
        # Map closed[d] fact ids first
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue
            if predicate == 'closed' and len(args) == 1:
                self._closed_fact_id[args[0]] = fact.global_id

        # Index rloc, holding, in facts; also collect room names
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue
            if predicate == 'rloc' and len(args) == 1:
                room = args[0]
                self._rooms.add(room)
                self._rloc_fact_id[room] = fact.global_id
            elif predicate == 'holding' and len(args) == 1:
                obj = args[0]
                self._holding_fact_id[obj] = fact.global_id
            elif predicate == 'in' and len(args) == 2:
                obj, room = args[0], args[1]
                self._rooms.add(room)
                dct = self._in_fact_id.setdefault(obj, {})
                dct[room] = fact.global_id

        # Build room index maps
        self._idx_to_room = sorted(list(self._rooms))
        self._room_to_idx = {r: i for i, r in enumerate(self._idx_to_room)}

    def _build_goals_from_model_goals(self):
        # Goals are provided as +in[obj,room] in the bitset self.model.goals
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            predicate, args = self._parse_fact_name(gname)
            if predicate == 'in' and len(args) == 2:
                obj, room = args[0], args[1]
                self._goal_room_of[obj] = room
        self._objects = sorted(list(self._goal_room_of.keys()))

    def _build_room_graph(self):
        # Build adjacency edges with closure fact per edge using initial +door facts
        init_fact_names = self.model.state_explicit_repr(self.model.initial_state)

        # Temporarily collect door edges as (room1, room2, door_id)
        temp_edges: List[Tuple[str, str, str]] = []
        for fname in init_fact_names:
            predicate, args = self._parse_fact_name(fname)
            if predicate == 'door' and len(args) == 3:
                r1, r2, d = args
                # Ensure rooms exist in index set (in case doors include rooms unseen elsewhere)
                if r1 not in self._room_to_idx:
                    self._rooms.add(r1)
                if r2 not in self._room_to_idx:
                    self._rooms.add(r2)
                temp_edges.append((r1, r2, d))

        # Rebuild indices if we added rooms via doors
        self._idx_to_room = sorted(list(self._rooms))
        self._room_to_idx = {r: i for i, r in enumerate(self._idx_to_room)}

        # Initialize adjacency with closure info
        n = len(self._idx_to_room)
        self._adj_edges = [[] for _ in range(n)]

        for r1, r2, d in temp_edges:
            i = self._room_to_idx.get(r1, -1)
            j = self._room_to_idx.get(r2, -1)
            if i == -1 or j == -1:
                continue
            closed_fid = self._closed_fact_id.get(d, -1)
            self._adj_edges[i].append((j, closed_fid))
            # Note: In domains provided, doors are given both directions as facts. We add directed edges as encountered.

    def _precompute_all_pairs_moves(self):
        # BFS from every room ignoring closed doors (i.e., every edge has cost 1)
        n = len(self._idx_to_room)
        INF = 10**9
        self._moves_dist = [[INF] * n for _ in range(n)]

        for s in range(n):
            dist = [INF] * n
            dq = deque([s])
            dist[s] = 0
            while dq:
                u = dq.popleft()
                for v, _closed_fid in self._adj_edges[u]:
                    if dist[v] == INF:
                        dist[v] = dist[u] + 1
                        dq.append(v)
            self._moves_dist[s] = dist

    # ---------- Helpers (state queries and distances) ----------
    @staticmethod
    def _parse_fact_name(name: str):
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
        if fact_id is None or fact_id < 0:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        # Check which +rloc[room] fact is true
        for room, fid in self._rloc_fact_id.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _get_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # If holding[obj] -> no current room via +in (object is with robot)
        fid_hold = self._holding_fact_id.get(obj, None)
        if fid_hold is not None and self._is_true(state_bitwise, fid_hold):
            return None
        # Otherwise find the unique room r where +in[obj,r] is true
        room_to_fid = self._in_fact_id.get(obj, {})
        for room, fid in room_to_fid.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _get_moves_dist(self, room_a: str, room_b: str) -> int:
        if room_a is None or room_b is None:
            return 0
        ia = self._room_to_idx.get(room_a, -1)
        ib = self._room_to_idx.get(room_b, -1)
        if ia == -1 or ib == -1:
            return 0
        d = self._moves_dist[ia][ib]
        if d == 10**9:
            # Disconnected (should not happen in provided instances); keep admissible by returning 0
            return 0
        return d

    def _min_closed_edges_along_path(self, state_bitwise: int, src_idx: int, dst_idx: int) -> int:
        # 0-1 BFS minimizing number of still-closed doors along path
        n = len(self._adj_edges)
        INF = float('inf')
        dist = [INF] * n
        dq = deque()
        dist[src_idx] = 0
        dq.append(src_idx)
        while dq:
            u = dq.popleft()
            if u == dst_idx:
                return dist[u]
            du = dist[u]
            for v, closed_fid in self._adj_edges[u]:
                w = 1 if (closed_fid is not None and closed_fid >= 0 and self._is_true(state_bitwise, closed_fid)) else 0
                nd = du + w
                if nd < dist[v]:
                    dist[v] = nd
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return INF

    # Optional: human-readable name
    @property
    def name(self) -> str:
        return "RobotGPT5Heuristic0"
