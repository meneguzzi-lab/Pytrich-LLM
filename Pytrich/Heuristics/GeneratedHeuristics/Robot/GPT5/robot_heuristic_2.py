from collections import deque, defaultdict
from typing import Dict, Tuple, List, Set, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic2(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Design rationale and domain insights:
    - Goals: Each package must end up in its goal room. Hierarchy achieves this by:
        1) Selecting a package not yet at goal (achieve-goals-pickup),
        2) Picking it up (pickup),
        3) Navigating to its goal (release with move/open as needed),
        4) Putting it down at goal (putdown).
      The task network recursively repeats until all packages are at their goals.

    - Action lower bounds:
        * Every not-at-goal package requires at least one putdown.
        * Every not-at-goal package not currently held also requires a pickup.
        * Movement is required to travel between rooms: we estimate using shortest-path distances
          while IGNORING door closed/open statuses for movement (admissible underestimation).

    - Travel lower bound structure used here:
        * If holding some package H: add distance(robot_loc -> goal(H)).
        * For every other not-delivered package P: add distance(cur_loc(P) -> goal(P)).
        * If not holding any package and there exists a not-delivered package: add
          min over P of distance(robot_loc -> cur_loc(P)) (first approach must happen at least once).

      This sum of move distances is a safe lower bound, ignoring potential inter-package repositioning.

    - Optional admissible opening contribution (used conservatively):
        * For the CURRENTLY HELD package only (or, if none is held, for the initial approach to the first package),
          we compute the minimal number of CLOSED doors along some path (0/1 BFS) and add it. This is still
          admissible as we never count opening costs for more than one phase in a single estimate, avoiding
          double counting persistent door openings across multiple packages.

    - Resulting heuristic:
        h = (#pickups) + (#putdowns) + (sum of move distances) + (closed-doors contribution for at most one leg)

    Efficiency:
        - We precompute room adjacency (ignoring door states) and all-pairs shortest paths once per model.
        - We map facts to fast bit indices and use bitwise checks.
        - At evaluation, we do O(#packages) lightweight checks and at most one 0/1 BFS for closed-doors contribution.

    Consistency/Admissibility:
        - Each component is a lower bound on the true remaining cost. Summing these components remains a lower bound.
        - We avoid over-counting opening costs across different deliveries to keep admissibility.
    """

    def __init__(self, robot_gpt5_heuristic_2: Optional[dict] = None):
        super().__init__()
        self.model = None
        self._initialized = False

        # Caches derived from the model structure (static w.r.t. the model)
        self.rooms: Set[str] = set()
        self.objects: Set[str] = set()

        # Fact mappings
        self.rloc_fact_by_room: Dict[str, int] = {}           # room -> fact.global_id for +rloc[room]
        self.in_facts_by_obj: Dict[str, List[Tuple[str, int]]] = defaultdict(list)  # obj -> list of (room, fact.global_id)
        self.holding_fact_by_obj: Dict[str, int] = {}         # obj -> fact.global_id for +holding[obj]
        self.closed_fact_by_door: Dict[str, int] = {}         # doorId -> fact.global_id for +closed[door]

        # Door / graph structures
        self.edge_closed_gid: Dict[Tuple[str, str], int] = {} # (room1, room2) -> closed fact gid
        self.adj_ignoring_closed: Dict[str, List[str]] = defaultdict(list)  # room -> neighbors

        # Goal mapping (obj -> goal_room) from model.goals
        self.goal_room_by_obj: Dict[str, str] = {}

        # All-pairs shortest path distances (ignoring door states); computed once per model
        self.apsp_moves: Dict[str, Dict[str, int]] = defaultdict(dict)  # room -> (room -> dist)

    # ------------------------- Framework integration helpers -------------------------
    def _ensure_model(self, model):
        if self.model is not model or not self._initialized:
            self.model = model
            self._build_static_caches()
            self._initialized = True

    def evaluate(self, model, state_bitwise: int, task_network: List) -> int:
        self._ensure_model(model)
        return self._heuristic(state_bitwise, task_network)

    # Provide common aliases to improve compatibility with different planner calls
    def __call__(self, model, state_bitwise: int, task_network: List) -> int:
        return self.evaluate(model, state_bitwise, task_network)

    def h(self, model, task_network: List, state_bitwise: int) -> int:
        return self.evaluate(model, state_bitwise, task_network)

    # ------------------------------ Core heuristic ------------------------------
    def _heuristic(self, state_bitwise: int, task_network: List) -> int:
        # If framework exposes goal_reached, short-circuit
        try:
            if self.model.goal_reached(state_bitwise, task_network):
                return 0
        except Exception:
            pass

        # Identify current robot room
        robot_room = self._get_robot_room(state_bitwise)
        if robot_room is None:
            # If unknown, return 0 to stay admissible
            return 0

        # Extract which object is currently held (if any)
        held_obj = self._get_held_object(state_bitwise)

        # Build current locations for packages that are not being held
        cur_loc_by_obj = self._get_current_locations(state_bitwise)

        # Compute not-delivered packages set
        not_delivered: List[str] = []
        for obj in self.objects:
            goal_r = self.goal_room_by_obj.get(obj)
            if goal_r is None:
                # No explicit goal -> nothing to do
                continue
            if obj == held_obj:
                # Held object is not yet at goal (not in any room until putdown)
                not_delivered.append(obj)
            else:
                cur_r = cur_loc_by_obj.get(obj)
                if cur_r is None:
                    # If we don't know where it is (shouldn't happen), be conservative
                    not_delivered.append(obj)
                else:
                    if cur_r != goal_r:
                        not_delivered.append(obj)

        if not not_delivered:
            return 0

        # Base action counts
        num_putdowns = len(not_delivered)
        num_pickups = len(not_delivered) - (1 if held_obj is not None else 0)

        # Travel distances (moves only, ignoring door states)
        total_moves = 0

        # If holding an object, add dist(robot -> goal(held))
        closed_bonus = 0  # closed-doors contribution for ONE leg only (to keep admissible vs. persistence)
        if held_obj is not None:
            goal_r = self.goal_room_by_obj.get(held_obj)
            if goal_r is not None:
                total_moves += self._dist_moves(robot_room, goal_r)
                # Add closed-doors minimal count along some path for this leg only
                closed_bonus += self._min_closed_along_path(state_bitwise, robot_room, goal_r)

            # For all other not-delivered packages, add dist(cur -> goal)
            for obj in not_delivered:
                if obj == held_obj:
                    continue
                cur_r = cur_loc_by_obj.get(obj)
                goal_r = self.goal_room_by_obj.get(obj)
                if cur_r is None or goal_r is None:
                    continue
                if cur_r != goal_r:
                    total_moves += self._dist_moves(cur_r, goal_r)
        else:
            # Not holding: add min approach distance robot -> cur_loc(obj)
            candidates: List[Tuple[int, int]] = []  # list of (moves, closed)
            for obj in not_delivered:
                cur_r = cur_loc_by_obj.get(obj)
                if cur_r is None:
                    continue
                moves_d = self._dist_moves(robot_room, cur_r)
                if moves_d < 10**12:
                    closed_d = self._min_closed_along_path(state_bitwise, robot_room, cur_r)
                    candidates.append((moves_d, closed_d))
            if candidates:
                best_moves, best_closed = min(candidates, key=lambda x: (x[0], x[1]))
                total_moves += best_moves
                closed_bonus += best_closed

            # Sum per-package cur -> goal moves
            for obj in not_delivered:
                cur_r = cur_loc_by_obj.get(obj)
                goal_r = self.goal_room_by_obj.get(obj)
                if cur_r is None or goal_r is None:
                    continue
                if cur_r != goal_r:
                    total_moves += self._dist_moves(cur_r, goal_r)

        # Final admissible estimate
        h_val = num_pickups + num_putdowns + total_moves + closed_bonus
        if h_val < 0:
            return 0
        return int(h_val)

    # ------------------------------ Static cache building ------------------------------
    def _build_static_caches(self):
        # Reset
        self.rooms.clear()
        self.objects.clear()
        self.rloc_fact_by_room.clear()
        self.in_facts_by_obj.clear()
        self.holding_fact_by_obj.clear()
        self.closed_fact_by_door.clear()
        self.edge_closed_gid.clear()
        self.adj_ignoring_closed.clear()
        self.goal_room_by_obj.clear()
        self.apsp_moves.clear()

        # First pass: collect predicate-specific facts
        for fact in self.model.facts:
            name = fact.name
            pred, args = self._parse_fact_name(name)
            if pred is None:
                continue

            if pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_by_room[room] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                obj, room = args[0], args[1]
                self.objects.add(obj)
                self.rooms.add(room)
                self.in_facts_by_obj[obj].append((room, fact.global_id))

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.objects.add(obj)
                self.holding_fact_by_obj[obj] = fact.global_id

            elif pred == 'closed' and len(args) == 1:
                door_id = args[0]
                self.closed_fact_by_door[door_id] = fact.global_id

        # Second pass: build door edges (adjacency + closed gid per edge)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'door' and len(args) == 3:
                loc1, loc2, door_id = args
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.adj_ignoring_closed[loc1].append(loc2)
                # Map edge to the closed-bit of this door id (exists even if false in state)
                closed_gid = self.closed_fact_by_door.get(door_id)
                if closed_gid is not None:
                    self.edge_closed_gid[(loc1, loc2)] = closed_gid

        # Build goal mapping from model.goals (bitwise int -> explicit names)
        goal_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'goal_in' and len(args) == 2:
                obj, room = args
                self.objects.add(obj)
                self.rooms.add(room)
                self.goal_room_by_obj[obj] = room

        # Precompute all-pairs shortest path distances (ignoring door closed/open statuses)
        self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        # BFS from each room
        for src in self.rooms:
            self.apsp_moves[src] = self._bfs_ignoring_closed(src)

    def _bfs_ignoring_closed(self, src: str) -> Dict[str, int]:
        dist = {r: 10**12 for r in self.rooms}
        if src not in self.rooms:
            return dist
        q = deque([src])
        dist[src] = 0
        while q:
            u = q.popleft()
            for v in self.adj_ignoring_closed.get(u, []):
                if dist[v] > dist[u] + 1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    # ------------------------------ State queries ------------------------------
    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        # There should be exactly one rloc true
        for room, gid in self.rloc_fact_by_room.items():
            if ((state_bitwise >> gid) & 1) == 1:
                return room
        return None

    def _get_held_object(self, state_bitwise: int) -> Optional[str]:
        for obj, gid in self.holding_fact_by_obj.items():
            if ((state_bitwise >> gid) & 1) == 1:
                return obj
        return None

    def _get_current_locations(self, state_bitwise: int) -> Dict[str, str]:
        # Returns current room for each obj that is not being held (i.e., in[obj,room] is true)
        cur: Dict[str, str] = {}
        for obj, entries in self.in_facts_by_obj.items():
            for room, gid in entries:
                if ((state_bitwise >> gid) & 1) == 1:
                    cur[obj] = room
                    break
        return cur

    # ------------------------------ Distances and closed-door contributions ------------------------------
    def _dist_moves(self, r1: Optional[str], r2: Optional[str]) -> int:
        if r1 is None or r2 is None:
            return 0
        if r1 == r2:
            return 0
        row = self.apsp_moves.get(r1)
        if not row:
            return 0
        d = row.get(r2, 10**12)
        if d >= 10**12:
            # Unreachable in the static graph (shouldn't happen in given instances)
            return 0
        return d

    def _min_closed_along_path(self, state_bitwise: int, src: Optional[str], tgt: Optional[str]) -> int:
        # Compute minimal number of CLOSED doors along any path from src to tgt using 0/1 BFS.
        if src is None or tgt is None or src == tgt:
            return 0
        if src not in self.rooms or tgt not in self.rooms:
            return 0

        INF = 10**9
        dist_closed: Dict[str, int] = {r: INF for r in self.rooms}
        dq = deque()
        dist_closed[src] = 0
        dq.append(src)

        while dq:
            u = dq.popleft()
            if u == tgt:
                break
            for v in self.adj_ignoring_closed.get(u, []):
                closed_gid = self.edge_closed_gid.get((u, v))
                edge_cost = 0
                if closed_gid is not None:
                    edge_cost = 1 if ((state_bitwise >> closed_gid) & 1) == 1 else 0
                nd = dist_closed[u] + edge_cost
                if nd < dist_closed[v]:
                    dist_closed[v] = nd
                    # 0-1 BFS optimization
                    if edge_cost == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)

        val = dist_closed.get(tgt, INF)
        if val >= INF:
            return 0
        return val

    # ------------------------------ Parsing helpers ------------------------------
    @staticmethod
    def _parse_fact_name(name: str):
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
