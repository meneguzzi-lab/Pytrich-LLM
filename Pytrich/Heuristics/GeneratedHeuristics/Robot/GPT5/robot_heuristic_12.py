from typing import Dict, List, Set, Tuple
from collections import deque, defaultdict

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic12(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Design rationale (admissible lower-bound):
    - Each misplaced package requires at least:
      - 2 primitive actions: pickup + putdown (unless it is currently held, then only 1 putdown), and
      - A number of move actions equal to the shortest path length from its current room to its goal room
        while carrying it (ignoring door-closed constraints to stay optimistic/admissible).
    - Additionally, to make any progress if nothing is currently held, the robot must first reach some
      misplaced package's location: we add the minimal distance from the robot's current location to any
      misplaced package's location as a one-time repositioning lower bound.
    - We ignore door opening costs (treat all doors as traversable) to preserve admissibility and keep
      computation cheap and scalable.

    Implementation details:
    - Preprocess the room graph from all 'door' facts (directed edges), ignoring 'closed' status.
    - Precompute all-pairs shortest path distances (in number of moves) via BFS from each room once.
    - Parse the goal facts from model.goals (bitwise) to get, for each package, its target room.
    - At evaluation time, use fast bitwise checks to:
        * Find current robot room (unique rloc[room] true fact)
        * Determine held object (if any) using holding[obj]
        * For each goal (in[obj,goal_room]) check whether it's already satisfied
        * For each unsatisfied object, determine its current room (unless held)
    - Heuristic value h(state) =
        sum over not-held, unsatisfied objects of (2 + dist(obj_room, goal_room))
        + if a held, unsatisfied object exists: dist(robot_room, its_goal_room) + 1
          else if there are unsatisfied objects: min over them of dist(robot_room, obj_room)

    Notes:
    - Uses only bitwise membership checks for speed as required by the framework.
    - Considers hierarchical structure implicitly by focusing on primitive cost lower bounds;
      decomposition steps are cost-free and not counted.
    """

    def __init__(self, model, robot_gpt5_heuristic_12: bool = True):
        super().__init__(model)
        self.model = model

        # Internal indices
        self.rooms: Set[str] = set()
        self.objects: Set[str] = set()

        # Parse and index facts
        self.rloc_fact_ids_by_room: Dict[str, int] = {}
        self.holding_fact_id_by_obj: Dict[str, int] = {}
        self.in_fact_ids_by_obj: Dict[str, Dict[str, int]] = defaultdict(dict)

        # Door adjacency (directed) ignoring 'closed' to keep admissible
        self.adj: Dict[str, Set[str]] = defaultdict(set)

        # Helper maps
        self._fact_name_to_id: Dict[str, int] = {}

        # Build base indices from all facts
        for fact in self.model.facts:
            name = fact.name
            self._fact_name_to_id[name] = fact.global_id
            pred, args = self._parse_fact_name(name)
            if not pred:
                continue

            if pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_ids_by_room[room] = fact.global_id

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.objects.add(obj)
                self.holding_fact_id_by_obj[obj] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                obj, room = args
                self.objects.add(obj)
                self.rooms.add(room)
                self.in_fact_ids_by_obj[obj][room] = fact.global_id

            elif pred == 'door' and len(args) == 3:
                loc1, loc2, d = args
                # Directed adjacency; admissible graph ignores 'closed' status
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.adj[loc1].add(loc2)

        # Parse goal state facts (bitwise)
        self.goal_of_obj: Dict[str, str] = {}
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gname in goal_fact_names:
            pred, args = self._parse_fact_name(gname)
            if pred == 'in' and len(args) == 2:
                obj, room = args
                self.goal_of_obj[obj] = room

        # For quick satisfied-check: precompute goal 'in' fact id per object, if available
        self.in_goal_fact_id_by_obj: Dict[str, int] = {}
        for obj, groom in self.goal_of_obj.items():
            fid = self.in_fact_ids_by_obj.get(obj, {}).get(groom)
            if fid is not None:
                self.in_goal_fact_id_by_obj[obj] = fid

        # Precompute all-pairs shortest path distances with unit edge costs over directed graph
        self._dist: Dict[str, Dict[str, int]] = {}
        self._precompute_all_pairs_shortest_paths()

    # -------------- Heuristic interface --------------
    def get_value(self, state_bitwise: int, task_network: List = None) -> int:
        # Quick goal check
        try:
            if self.model.goal_reached(state_bitwise, task_network or []):
                return 0
        except Exception:
            # If goal_reached signature differs, fall back to state-only check
            if (self.model.goals & state_bitwise) == self.model.goals:
                return 0

        # Determine current robot room
        robot_room = self._get_robot_room(state_bitwise)

        # Determine held object (if any)
        held_obj = None
        for obj, fid in self.holding_fact_id_by_obj.items():
            if self._is_true(state_bitwise, fid):
                held_obj = obj
                break

        total = 0
        # Collect candidate distances from robot to first object location if nothing is held
        first_pick_dists: List[int] = []

        # When holding an object that is not yet at goal, we handle it specially
        held_contrib_added = False

        for obj, goal_room in self.goal_of_obj.items():
            # If goal already satisfied: skip
            goal_fid = self.in_goal_fact_id_by_obj.get(obj)
            if goal_fid is not None and self._is_true(state_bitwise, goal_fid):
                continue

            if obj == held_obj:
                # Need to move (optimistically) from robot_room to goal_room and then putdown
                move_cost = self._get_dist(robot_room, goal_room)
                total += move_cost + 1  # putdown
                held_contrib_added = True
                continue

            # Object not at goal and not held: need pickup + carry + putdown
            # First, find object's current room
            obj_room = self._get_object_room(state_bitwise, obj)
            if obj_room is None:
                # If we can't find its room and it's not held, be optimistic: assume 0 additional
                # However, to stay informative, still add the action costs minimally
                total += 2  # pickup + putdown
                continue

            carry_moves = self._get_dist(obj_room, goal_room)
            total += 2 + carry_moves  # pickup + putdown + moves while holding

            # Candidate distance for initial reposition (only if nothing is held)
            if held_obj is None and robot_room is not None:
                d = self._get_dist(robot_room, obj_room)
                first_pick_dists.append(d)

        # If no object is currently held, we must at least reach the first object's location once
        if held_obj is None and len(first_pick_dists) > 0:
            total += min(first_pick_dists)

        return int(total)

    # Provide __call__ as alias for frameworks expecting callable heuristics
    def __call__(self, state_bitwise: int, task_network: List = None) -> int:
        return self.get_value(state_bitwise, task_network)

    # -------------- Helpers --------------
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

    def _is_true(self, state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) == 1

    def _get_robot_room(self, state_bitwise: int) -> str:
        # Exactly one rloc should be true
        for room, fid in self.rloc_fact_ids_by_room.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None  # Fallback if unspecified

    def _get_object_room(self, state_bitwise: int, obj: str) -> str:
        # Find the true 'in[obj,room]' fact
        room_map = self.in_fact_ids_by_obj.get(obj, {})
        for room, fid in room_map.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _precompute_all_pairs_shortest_paths(self):
        # Compute BFS distances from each room over directed adjacency
        for src in self.rooms:
            self._dist[src] = self._bfs_distances_from(src)

    def _bfs_distances_from(self, src: str) -> Dict[str, int]:
        INF = 10 ** 9
        dist: Dict[str, int] = {r: INF for r in self.rooms}
        if src is None:
            return dist
        q = deque()
        dist[src] = 0
        q.append(src)
        while q:
            u = q.popleft()
            du = dist[u]
            for v in self.adj.get(u, ()):  # neighbors via any door
                if dist[v] > du + 1:
                    dist[v] = du + 1
                    q.append(v)
        return dist

    def _get_dist(self, a: str, b: str) -> int:
        # Optimistic distance (ignores closed doors), returns 0 if unknown to stay admissible
        if a is None or b is None:
            return 0
        da = self._dist.get(a)
        if not da:
            return 0
        d = da.get(b)
        if d is None or d >= 10 ** 9:
            # If unreachable even in optimistic graph, return 0 to remain a valid lower bound
            return 0
        return d
