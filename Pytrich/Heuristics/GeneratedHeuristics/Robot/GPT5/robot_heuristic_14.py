from Pytrich.Heuristics.heuristic import Heuristic
import heapq
from collections import defaultdict, deque


class RobotGPT5Heuristic14(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Key idea:
    - Each unsatisfied package requires at least one pickup and one putdown: contributes 2 per object.
    - If currently holding an unsatisfied package, we can save one pickup across the whole problem.
    - Movement lower bound: at least the cheapest next-object travel cost considering door openings on the first leg only.
      We compute:
        movement_lb = min over all unsatisfied objects i of
            dist_closed(current_room -> obj_room_i) + dist_unit(obj_room_i -> goal_room_i)
      Additionally, if holding an unsatisfied package h, we also consider
            dist_closed(current_room -> goal_room_h)
      where dist_closed treats every closed door traversal as cost 2 (1 to open, 1 to move) and open door as 1,
      and dist_unit is an all-edges-open unit-cost distance (precomputed, static).

    This yields an admissible and fast-to-compute heuristic that leverages the door structure and current door states.
    """

    def __init__(self, model, robot_gpt5_heuristic_14=True):
        super().__init__(model)
        self.model = model
        self.name = "RobotGPT5Heuristic14"

        # Internal caches and structures
        self.rooms = []                 # List[str]
        self.room_index = {}            # str -> int
        self.edges = []                 # List[List[Tuple[int, str]]] neighbor index, door_id
        self.closed_fact_by_door = {}   # door_id -> Fact.global_id (for +closed[door]) or None if not present
        self.rloc_fact_by_room = {}     # room_idx -> global_id of +rloc[room]
        self.holding_fact_by_obj = {}   # obj -> global_id of +holding[obj]
        self.in_fact_ids = defaultdict(dict)  # obj -> {room_idx: global_id of +in[obj,room]}
        self.goal_room_of_obj = {}      # obj -> room_idx
        self.unit_dist = []             # All-pairs unit distances between rooms (ignoring closed doors)

        # Build all static structures from model facts
        self._prepare_static_structures()
        self._precompute_unit_distances()

    # -------------------------- Core API --------------------------
    def h(self, state_bitwise: int, task_network=None) -> int:
        if task_network is None:
            task_network = []

        # Quick goal test: if planner already reached goals and no tasks, return 0
        try:
            if self.model.goal_reached(state_bitwise, task_network):
                return 0
        except Exception:
            # If goal_reached not reliable with HTN TN here, continue heuristic computation
            pass

        # 1) Locate current robot room
        current_room_idx = self._get_current_robot_room_idx(state_bitwise)

        # 2) Identify held object (if any)
        held_obj = self._get_held_object(state_bitwise)

        # 3) Determine unsatisfied objects and count
        unsat_objects = []
        for obj, goal_ridx in self.goal_room_of_obj.items():
            fid = self.in_fact_ids.get(obj, {}).get(goal_ridx, None)
            if fid is None:
                # If the +in[obj, goal_room] fact doesn't exist at all, be conservative: treat as unsatisfied
                unsat_objects.append(obj)
            else:
                if not self._is_true(state_bitwise, fid):
                    unsat_objects.append(obj)

        n_unsat = len(unsat_objects)
        if n_unsat == 0:
            return 0

        # 4) Base action lower bound for pickups and putdowns (per unsatisfied object)
        #    - Each unsatisfied object requires one putdown
        #    - Minimal number of pickups is n_unsat, unless we are already holding one of them
        pickups_lb = n_unsat - (1 if held_obj in unsat_objects else 0)
        putdowns_lb = n_unsat
        base_actions = pickups_lb + putdowns_lb

        # 5) Movement lower bound: cheapest next-object travel cost
        #    Dijkstra from current room with door-aware edge weights (1 if open, 2 if closed)
        dist_from_current_closed = self._dijkstra_closed(state_bitwise, current_room_idx)

        movement_options = []

        # Option A: If holding an unsatisfied object, consider finishing it next
        if held_obj in unsat_objects:
            goal_idx = self.goal_room_of_obj[held_obj]
            d = dist_from_current_closed[goal_idx]
            if d < float('inf'):
                movement_options.append(d)

        # Option B: Consider any other unsatisfied object: go to it (closed-aware), then to goal (unit/open-aware)
        for obj in unsat_objects:
            obj_room_idx = self._get_object_room_idx(state_bitwise, obj, current_room_idx, held_obj)
            if obj_room_idx is None:
                continue
            goal_idx = self.goal_room_of_obj.get(obj, None)
            if goal_idx is None:
                continue
            d1 = dist_from_current_closed[obj_room_idx]
            d2 = self.unit_dist[obj_room_idx][goal_idx]
            total = d1 + d2
            if d1 < float('inf') and d2 < float('inf'):
                movement_options.append(total)

        if movement_options:
            movement_lb = int(min(movement_options))
        else:
            # If no movement option is feasible (disconnected), keep LB minimal to stay admissible
            movement_lb = 0

        h_val = base_actions + movement_lb
        if h_val < 0:
            return 0
        return int(h_val)

    # -------------------------- Helpers: State Queries --------------------------
    def _is_true(self, state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) == 1

    def _get_current_robot_room_idx(self, state_bitwise: int):
        # There should be exactly one rloc true
        for ridx, fid in self.rloc_fact_by_room.items():
            if self._is_true(state_bitwise, fid):
                return ridx
        # Fallback if missing (should not happen): return index 0
        return 0

    def _get_held_object(self, state_bitwise: int):
        for obj, fid in self.holding_fact_by_obj.items():
            if self._is_true(state_bitwise, fid):
                return obj
        return None

    def _get_object_room_idx(self, state_bitwise: int, obj: str, current_room_idx: int, held_obj: str):
        # If holding this object, its location equals the robot location
        if held_obj == obj:
            return current_room_idx
        # Otherwise, find the true +in[obj, room]
        room_to_fid = self.in_fact_ids.get(obj, {})
        for ridx, fid in room_to_fid.items():
            if self._is_true(state_bitwise, fid):
                return ridx
        # Not found; might be an inconsistent state; return None conservatively
        return None

    # -------------------------- Graph and Distances --------------------------
    def _dijkstra_closed(self, state_bitwise: int, start_idx: int):
        # Edge weight = 1 if door open, else 2 if door closed in current state
        n = len(self.rooms)
        dist = [float('inf')] * n
        dist[start_idx] = 0
        pq = [(0, start_idx)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist[u]:
                continue
            for v, door_id in self.edges[u]:
                closed_fid = self.closed_fact_by_door.get(door_id, None)
                is_closed = 1 if (closed_fid is not None and self._is_true(state_bitwise, closed_fid)) else 0
                w = 1 + is_closed
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def _precompute_unit_distances(self):
        # BFS from each room with unit cost on all edges (ignore closed/open)
        n = len(self.rooms)
        self.unit_dist = [[float('inf')] * n for _ in range(n)]
        for s in range(n):
            self.unit_dist[s][s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                du = self.unit_dist[s][u]
                for v, _ in self.edges[u]:
                    if self.unit_dist[s][v] == float('inf'):
                        self.unit_dist[s][v] = du + 1
                        q.append(v)

    # -------------------------- Static Model Parsing --------------------------
    def _prepare_static_structures(self):
        # Parse facts into structures: rooms, doors, closed map, rloc, holding, in, goal_in
        init_state = self.model.initial_state

        # Temporary holders
        rooms_set = set()
        door_edges = []  # tuples: (room1, room2, door_id)
        closed_candidates = set()  # door_ids that appear in closed facts

        # First pass: identify all relevant facts and construct mappings
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == 'door':
                # Use only doors that are true in init (static predicate)
                if self._is_true(init_state, fact.global_id):
                    r1, r2, d = args
                    rooms_set.add(r1)
                    rooms_set.add(r2)
                    door_edges.append((r1, r2, d))

            elif pred == 'rloc':
                r = args[0]
                rooms_set.add(r)
                # Store mapping from room -> rloc fact id later after we create room_index

            elif pred == 'closed':
                d = args[0]
                closed_candidates.add(d)

            elif pred == 'holding':
                obj = args[0]
                self.holding_fact_by_obj[obj] = fact.global_id

            elif pred == 'in':
                obj, r = args
                rooms_set.add(r)
                # Save later when room_index ready
                # Temporarily store as is; we'll post-process once room_index is built

            elif pred == 'goal_in':
                # We'll only record those that are true in initial state (they are static throughout)
                if self._is_true(init_state, fact.global_id):
                    obj, r = args
                    rooms_set.add(r)

        # Create room index mapping
        self.rooms = sorted(rooms_set)
        self.room_index = {r: i for i, r in enumerate(self.rooms)}

        # Initialize graph edges
        self.edges = [[] for _ in self.rooms]
        for r1, r2, d in door_edges:
            if r1 in self.room_index and r2 in self.room_index:
                i = self.room_index[r1]
                j = self.room_index[r2]
                self.edges[i].append((j, d))

        # Map closed facts by door id (if a door id has a closed fact, store its fact id)
        # We need a search over facts again to link door ids -> closed fact ids
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'closed':
                d = args[0]
                self.closed_fact_by_door[d] = fact.global_id
        # Ensure every door has an entry (None if no closed fact found -> treated as always open)
        for _, _, d in door_edges:
            if d not in self.closed_fact_by_door:
                self.closed_fact_by_door[d] = None

        # Fill rloc facts per room
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'rloc':
                r = args[0]
                ridx = self.room_index.get(r, None)
                if ridx is not None:
                    self.rloc_fact_by_room[ridx] = fact.global_id

        # Fill in[obj, room] mappings per object
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'in':
                obj, r = args
                ridx = self.room_index.get(r, None)
                if ridx is not None:
                    self.in_fact_ids[obj][ridx] = fact.global_id

        # Map goal_in[obj] -> room_idx
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred == 'goal_in' and self._is_true(init_state, fact.global_id):
                obj, r = args
                ridx = self.room_index.get(r, None)
                if ridx is not None:
                    self.goal_room_of_obj[obj] = ridx

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []
