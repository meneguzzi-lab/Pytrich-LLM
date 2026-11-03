from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional

# Pytrich interface
from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT52Heuristic17(Heuristic):
    """Domain-dependent heuristic for the HDDL Robot domain.

    Domain summary (grounded):
      - Facts: rloc[room], in[obj,room], holding[obj], armempty, closed[door], door[room,room,door], goal_in[obj,room]
      - Primitive actions: open, move, pickup, putdown
      - HTN structure: achieve-goals repeatedly picks some misplaced object, picks it up, then release recursively opens/moves until it can putdown at goal.

    Key domain insights leveraged:
      1) Each object that is not already at its goal room must be delivered.
         In this domain, delivering an object necessarily requires at least:
           - pickup (1)
           - putdown (1)
           - moving from pickup-room to goal-room (>= shortest-path moves)
           - opening at least the closed doors encountered on that path (>= closed doors along a shortest path)
         Additionally, if the robot is not currently in the pickup room, it must at least travel there first.

      2) Doors, rooms, and connectivity are static (door predicate), while door-closed is dynamic.
         We precompute shortest-path distances on the room graph (ignoring closed) once.
         For any (roomA, roomB), we also precompute the minimum number of *door traversals* that correspond to currently-closed doors along a path.
         This is done per state (since closed changes), but efficiently with a 0-1 BFS per relevant source.

      3) HTN task network: typically contains achieve-goals and release/open/move abstractions.
         We do not try to exactly price decompositions (would be expensive), but we include a small task-network-aware lower bound:
           - Each pending (pickup_abstract obj) implies at least 1 pickup.
           - Each pending (putdown_abstract) implies at least 1 putdown.
           - Each pending (move_abstract) implies at least 1 move.
           - Each pending (open_abstract) implies at least 1 open.
         This helps guide search when the task network already committed to certain abstract steps.

    The heuristic returned is admissible as it is a lower bound formed by:
      - selecting the single cheapest next object-delivery (or the held-object completion) cost lower bound, PLUS
      - a task-network minimal primitive-count lower bound.

    We intentionally do NOT sum costs over all remaining objects (would overcount shared travel / openings and break admissibility).
    Instead, we take a max of independent lower bounds (still admissible) but here we use:
      h = max(h_object_delivery_lb, h_task_network_lb)

    Efficiency:
      - Precomputes all-pairs room distances and the door-id used for each directed edge.
      - Per state, runs 0-1 BFS for only a few sources (robot location + rooms that contain misplaced objects + holding goal room).
      - Uses bitwise checks only (no state_explicit_repr on current state).
    """

    def __init__(self, model, robot_gpt52_heuristic_17: str = "robot_gpt52_heuristic_17"):
        super().__init__(model, robot_gpt52_heuristic_17)
        self.model = model

        # --- Fact indices we care about (by parsing grounded fact names once) ---
        self.f_armempty: Optional[int] = None
        self.rloc_facts: Dict[str, int] = {}                 # room -> fact_id
        self.in_facts: Dict[Tuple[str, str], int] = {}       # (obj, room) -> fact_id
        self.holding_facts: Dict[str, int] = {}              # obj -> fact_id
        self.closed_facts: Dict[str, int] = {}               # door -> fact_id
        self.goal_in_pairs: Dict[str, str] = {}              # obj -> goal_room

        # door graph
        self.room_nodes: set = set()
        self.edge_to_door: Dict[Tuple[str, str], str] = {}   # (room1, room2) -> door_id
        self.adj: Dict[str, List[str]] = defaultdict(list)

        # derived
        self.objects: set = set()
        self.rooms: set = set()
        self.doors: set = set()

        # task primitive lower bounds
        self._opname_cost_lb = {
            "pickup_abstract": 1,
            "putdown_abstract": 1,
            "move_abstract": 1,
            "open_abstract": 1,
        }

        # Parse all facts and build static maps
        for f in self.model.facts:
            pred, args = self._parse_fact_name(f.name)
            if pred is None:
                continue
            if pred == "armempty" and len(args) == 0:
                self.f_armempty = f.global_id
            elif pred == "rloc" and len(args) == 1:
                room = args[0]
                self.rloc_facts[room] = f.global_id
                self.rooms.add(room)
            elif pred == "in" and len(args) == 2:
                obj, room = args
                self.in_facts[(obj, room)] = f.global_id
                self.objects.add(obj)
                self.rooms.add(room)
            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                self.holding_facts[obj] = f.global_id
                self.objects.add(obj)
            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self.closed_facts[d] = f.global_id
                self.doors.add(d)
            elif pred == "door" and len(args) == 3:
                r1, r2, d = args
                self.edge_to_door[(r1, r2)] = d
                self.adj[r1].append(r2)
                self.room_nodes.add(r1)
                self.room_nodes.add(r2)
                self.rooms.add(r1)
                self.rooms.add(r2)
                self.doors.add(d)
            elif pred == "goal_in" and len(args) == 2:
                obj, room = args
                self.goal_in_pairs[obj] = room
                self.objects.add(obj)
                self.rooms.add(room)

        # Precompute unweighted shortest path distance between all rooms (ignoring door closed)
        self._dist: Dict[str, Dict[str, int]] = {}
        for src in self.rooms:
            self._dist[src] = self._bfs_room_distance(src)

        # Cache for per-state closed-door 0-1 BFS results
        self._cache_closed_signature: Optional[int] = None
        self._cache_closed01: Dict[str, Dict[str, int]] = {}  # src_room -> {dst_room: min_closed_doors_along_path}

    def __call__(self, state_bitwise: int, task_network=None) -> int:
        if task_network is None:
            task_network = []

        # Task-network primitive LB
        h_tn = self._task_network_lb(task_network)

        # Object delivery LB (one-step lookahead lower bound for the next necessary delivery)
        h_obj = self._one_object_delivery_lb(state_bitwise)

        return max(h_tn, h_obj)

    # -------------------------- Core lower bounds --------------------------

    def _one_object_delivery_lb(self, state: int) -> int:
        """Admissible LB for completing at least one still-required delivery.

        If holding an object that is not yet at its goal, estimate cost to finish it.
        Else pick the cheapest misplaced object and estimate cost to pick+deliver it.

        Returns 0 if all objects are already at their goal rooms (goal facts satisfied).
        """
        robot_room = self._get_robot_room(state)
        if robot_room is None:
            return 0

        held_obj = self._get_held_object(state)
        if held_obj is not None:
            goal_room = self.goal_in_pairs.get(held_obj)
            if goal_room is None:
                return 0
            # If already in goal room, only putdown remains (1)
            if robot_room == goal_room:
                return 1
            # Must travel to goal and open closed doors on the way, then putdown
            move_d = self._dist_get(robot_room, goal_room)
            if move_d is None:
                return 0  # disconnected; heuristic stays safe
            closed_need = self._min_closed_doors_on_path(state, robot_room, goal_room)
            return move_d + closed_need + 1  # moves + opens + putdown

        # Not holding anything: choose best misplaced object to deliver next
        best = None
        for obj, goal_room in self.goal_in_pairs.items():
            # If already at goal, skip
            if self._is_obj_in_room(state, obj, goal_room):
                continue

            cur_room = self._find_obj_room(state, obj)
            if cur_room is None:
                continue

            # Lower bound cost components:
            # travel to object + (pickup) + travel to goal + (putdown)
            # plus openings needed along these paths
            d1 = self._dist_get(robot_room, cur_room)
            d2 = self._dist_get(cur_room, goal_room)
            if d1 is None or d2 is None:
                continue

            open1 = self._min_closed_doors_on_path(state, robot_room, cur_room)
            open2 = self._min_closed_doors_on_path(state, cur_room, goal_room)

            cost_lb = d1 + open1 + 1 + d2 + open2 + 1
            if best is None or cost_lb < best:
                best = cost_lb

        return 0 if best is None else best

    def _task_network_lb(self, task_network) -> int:
        """Very cheap LB derived from current task network.

        Counts abstract tasks that directly map to one primitive in this domain.
        This is admissible because each such abstract task must decompose to at least one primitive operator.
        """
        if not task_network:
            return 0

        lb = 0
        for t in task_network:
            # Operators are already primitive; counting them would be admissible but can overcount with costs if non-unit.
            # Here all primitive costs are 1 in instances, but we stay conservative:
            if hasattr(t, "cost"):
                # primitive operator
                lb += 0
            else:
                # abstract task
                name = getattr(t, "name", "")
                # name examples typically: 'pickup_abstract o1' or 'pickup_abstract(o1)' depending on grounding;
                # we use prefix match.
                for k, v in self._opname_cost_lb.items():
                    if name.startswith(k):
                        lb += v
                        break
        return lb

    # -------------------------- State queries (bitwise only) --------------------------

    def _get_robot_room(self, state: int) -> Optional[str]:
        for room, fid in self.rloc_facts.items():
            if (state >> fid) & 1:
                return room
        return None

    def _get_held_object(self, state: int) -> Optional[str]:
        for obj, fid in self.holding_facts.items():
            if (state >> fid) & 1:
                return obj
        return None

    def _is_obj_in_room(self, state: int, obj: str, room: str) -> bool:
        fid = self.in_facts.get((obj, room))
        if fid is None:
            return False
        return ((state >> fid) & 1) == 1

    def _find_obj_room(self, state: int, obj: str) -> Optional[str]:
        # Fast scan over rooms; facts are pre-indexed
        for room in self.rooms:
            fid = self.in_facts.get((obj, room))
            if fid is not None and ((state >> fid) & 1):
                return room
        return None

    def _closed_signature(self, state: int) -> int:
        """A compact signature of which door-closed facts are true, used to cache 0-1 BFS results."""
        sig = 0
        # fold bits for each closed fact into an integer signature
        # (not perfect hashing, but deterministic and fast)
        for d, fid in self.closed_facts.items():
            if (state >> fid) & 1:
                # mix in door fact id
                sig ^= (1 << (fid % 61))
                sig = (sig * 1099511628211) & ((1 << 64) - 1)
        return sig

    # -------------------------- Graph distances --------------------------

    def _bfs_room_distance(self, src: str) -> Dict[str, int]:
        dist = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist[u]
            for v in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _dist_get(self, a: str, b: str) -> Optional[int]:
        dmap = self._dist.get(a)
        if not dmap:
            return None
        return dmap.get(b)

    def _min_closed_doors_on_path(self, state: int, src: str, dst: str) -> int:
        """Minimum number of currently-closed doors that must be opened to travel from src to dst.

        We treat traversing an edge whose associated door is currently closed as cost 1, else 0.
        Then compute shortest path cost with 0-1 BFS.

        Admissible as a lower bound on required open actions along a route.
        """
        if src == dst:
            return 0

        sig = self._closed_signature(state)
        if self._cache_closed_signature != sig:
            self._cache_closed_signature = sig
            self._cache_closed01.clear()

        if src not in self._cache_closed01:
            self._cache_closed01[src] = self._zero_one_bfs_closed_cost(state, src)

        return self._cache_closed01[src].get(dst, 0)

    def _zero_one_bfs_closed_cost(self, state: int, src: str) -> Dict[str, int]:
        INF = 10**9
        dist = {src: 0}
        dq = deque([src])

        while dq:
            u = dq.popleft()
            du = dist[u]
            for v in self.adj.get(u, []):
                door_id = self.edge_to_door.get((u, v))
                w = 0
                if door_id is not None:
                    cfid = self.closed_facts.get(door_id)
                    if cfid is not None and ((state >> cfid) & 1):
                        w = 1
                nd = du + w
                if nd < dist.get(v, INF):
                    dist[v] = nd
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist

    # -------------------------- Fact parsing --------------------------

    def _parse_fact_name(self, name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
            # unary with no args should still have [] in Pytrich, but be robust:
            return name, []
        return None, []
