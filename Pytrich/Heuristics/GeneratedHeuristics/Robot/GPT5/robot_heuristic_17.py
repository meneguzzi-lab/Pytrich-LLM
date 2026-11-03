from collections import deque, defaultdict
from typing import Dict, List, Set, Tuple, Optional

# Attempt to import the planner's Heuristic base class with robust fallbacks
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic17(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Key idea (admissible lower bound):
    - Each undelivered package requires exactly one putdown (necessary) and, unless it's already being held, one pickup.
    - Any package not yet at its goal must be transported from its current location to the goal room. The minimal
      number of move actions to transport it is at least the shortest-path distance (in number of doors traversed)
      between its current room and its goal room. We compute this distance on the undirected room graph defined by
      'door' facts while IGNORING the 'closed' predicate. Ignoring closed doors makes the estimate optimistic
      (admissible), because in reality some doors may first need to be opened.
    - We sum these per-package carrying distances (each package must be carried independently as capacity is 1),
      and add the minimal number of necessary pickups and putdowns.
    - We do not add travel needed to get to the first package, nor inter-package repositioning travel, nor any open
      actions. This keeps the heuristic admissible and cheap to compute.

    h(s) = Sum_over_remaining_packages shortest_path_length(curr_room(o, s), goal_room(o))
           + (#remaining_packages)  [putdowns]
           + (#remaining_packages_not_currently_held)  [pickups]

    Where curr_room(o, s) is:
      - the robot room if o is currently held
      - otherwise the unique room r with in(o, r) true in s

    Implementation details:
    - Precompute room graph from 'door' facts (undirected).
    - Precompute all-pairs shortest paths (BFS from each room) on that graph.
    - Precompute goal mapping goal_room_by_obj from model.goals (which encode '+in[obj,room]') as provided.
    - Use fast bitwise checks against precomputed Fact global_ids for rloc, in, holding.

    This heuristic is cheap, scales well to large instances (100 packages, ~50 rooms), and significantly reduces
    node expansions by correlating the remaining transport work with the HTN's required actions.
    """

    def __init__(self, model, robot_gpt5_heuristic_17: bool = True):
        super().__init__(model)

        # Storage
        self.room_names: Set[str] = set()
        self.adj: Dict[str, Set[str]] = defaultdict(set)  # undirected adjacency: room -> neighbors

        # Fact id indices for fast bitwise checks
        self.rloc_fact_id_by_room: Dict[str, int] = {}            # rloc[room] -> fact id
        self.in_fact_id_by_obj_room: Dict[Tuple[str, str], int] = {}  # in[obj, room] -> fact id
        self.holding_fact_id_by_obj: Dict[str, int] = {}          # holding[obj] -> fact id

        # Goal mapping: obj -> goal_room
        self.goal_room_by_obj: Dict[str, str] = {}

        # All-pairs shortest path distances (in number of moves), ignoring closed doors
        self._dist: Dict[str, Dict[str, int]] = {}

        # Parse static model facts (doors, rlocs, ins, holdings)
        for fact in self.model.facts:
            predicate, args = self._parse_fact_name(fact.name)
            if not predicate:
                continue

            if predicate == 'door' and len(args) == 3:
                r1, r2, _ = args
                # Build undirected graph on rooms
                self.room_names.add(r1)
                self.room_names.add(r2)
                self.adj[r1].add(r2)
                self.adj[r2].add(r1)

            elif predicate == 'rloc' and len(args) == 1:
                r = args[0]
                self.room_names.add(r)
                self.rloc_fact_id_by_room[r] = fact.global_id

            elif predicate == 'in' and len(args) == 2:
                obj, r = args
                self.room_names.add(r)
                self.in_fact_id_by_obj_room[(obj, r)] = fact.global_id

            elif predicate == 'holding' and len(args) == 1:
                obj = args[0]
                self.holding_fact_id_by_obj[obj] = fact.global_id

        # Ensure rooms seen in 'in' or 'rloc' are present in adjacency dict
        for r in self.room_names:
            if r not in self.adj:
                self.adj[r] = set()

        # Precompute all-pairs shortest path distances (BFS per room) ignoring closed
        self._compute_all_pairs_shortest_paths()

        # Extract goals: model.goals is a bitwise int. Use explicit repr ONCE during init as recommended.
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for gf in goal_fact_names:
            predicate, args = self._parse_fact_name(gf)
            if predicate == 'in' and len(args) == 2:
                obj, goal_room = args
                # If multiple goals per obj exist, last one wins (domain instances provide 1 per obj)
                self.goal_room_by_obj[obj] = goal_room

        # Cache list of rooms for iteration order
        self.rooms_list: List[str] = list(self.room_names)

    def h(self, state_bitwise: int, task_network: List) -> int:
        # If goal already reached and no tasks left, heuristic is 0
        if self.model.goal_reached(state_bitwise, task_network):
            return 0

        # Determine current robot location
        robot_room = self._get_robot_room(state_bitwise)

        total_moves_lb = 0
        pickups_needed = 0
        putdowns_needed = 0

        # Iterate over goal objects
        for obj, goal_room in self.goal_room_by_obj.items():
            # If already at goal, skip
            in_goal_fact_id = self.in_fact_id_by_obj_room.get((obj, goal_room))
            if in_goal_fact_id is not None and self._is_true(state_bitwise, in_goal_fact_id):
                continue

            # Determine if currently holding this object
            is_holding = self._is_true(state_bitwise, self.holding_fact_id_by_obj.get(obj, -1))

            # Determine the object's current room
            if is_holding and robot_room is not None:
                curr_room = robot_room
            else:
                curr_room = self._get_object_room(state_bitwise, obj)

            # Lower bound on moves to deliver this package: shortest path curr_room -> goal_room (ignoring closed)
            if curr_room is not None and goal_room in self._dist.get(curr_room, {}):
                d = self._dist[curr_room][goal_room]
                if d < float('inf'):
                    total_moves_lb += d
                # If distance unavailable/infinite, contribute 0 to keep admissibility and robustness
            # else unknown rooms -> contribute 0

            # Each undelivered package requires a putdown
            putdowns_needed += 1
            # Plus a pickup unless it is already being held
            if not is_holding:
                pickups_needed += 1

        # Final heuristic value (admissible lower bound)
        h_value = total_moves_lb + putdowns_needed + pickups_needed
        if h_value < 0:
            return 0
        return int(h_value)

    # ------------------------ Helper methods ------------------------
    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith('+') or name.startswith('-'):
            name2 = name[1:]
            bracket_idx = name2.find('[')
            if bracket_idx > 0 and name2.endswith(']'):
                predicate = name2[:bracket_idx]
                args_str = name2[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_id: int) -> bool:
        if fact_id is None or fact_id < 0:
            return False
        return ((state_bitwise >> fact_id) & 1) == 1

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        for r, fid in self.rloc_fact_id_by_room.items():
            if self._is_true(state_bitwise, fid):
                return r
        return None

    def _get_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Scan known rooms for a true in[obj, room]
        # O(|rooms|) bitwise checks; cheap (<= ~60)
        for r in self.rooms_list:
            fid = self.in_fact_id_by_obj_room.get((obj, r))
            if fid is not None and self._is_true(state_bitwise, fid):
                return r
        return None

    def _compute_all_pairs_shortest_paths(self) -> None:
        # BFS from each room; edges are unit cost (one move per door traversal)
        self._dist = {r: {s: float('inf') for s in self.room_names} for r in self.room_names}
        for r in self.room_names:
            self._dist[r][r] = 0
            q = deque([r])
            while q:
                u = q.popleft()
                du = self._dist[r][u]
                for v in self.adj.get(u, []):
                    if self._dist[r][v] == float('inf'):
                        self._dist[r][v] = du + 1
                        q.append(v)

# Alias for external registration if needed
RobotGPT5Heuristic17Class = RobotGPT5Heuristic17
