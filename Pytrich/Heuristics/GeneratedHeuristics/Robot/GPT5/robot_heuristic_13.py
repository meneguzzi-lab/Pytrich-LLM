from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Set

from Pytrich.Heuristics.heuristic import Heuristic


class RobotGPT5Heuristic13(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Design rationale (admissible lower bound):
    - Each misplaced package requires at least:
        * pickup (1) if not currently held
        * putdown (1)
      So: 2 actions if not held, 1 if currently held (regardless of position).
    - Carrying moves: To relocate a package from its current location to its goal
      location, the robot must perform at least the number of move actions equal to the
      graph shortest-path length between those two rooms. When the package is held,
      its current location is the robot's current room.
        * We ignore door-opening costs for admissibility (assuming doors are already open)
          because openings only increase the real cost; ignoring them preserves a lower bound.
    - Initial approach move (when not holding anything): To start working on misplaced packages,
      the robot must at least reach one of them. We add the minimal distance from the robot's
      current location to any misplaced, non-held package's current room. This is a necessary
      (non-carrying) movement before the first pickup and is not counted in the carrying distances.

    Overall heuristic value h(s) =
        sum_over_misplaced_packages( carrying_shortest_path )
      + sum_over_misplaced_packages( 2 if not held else 1 )
      + (0 if holding any package OR no misplaced non-held packages else
         min_dist(robot_loc, any_misplaced_non_held_pkg_loc))

    Notes:
    - Goals are extracted from model.goals (bitwise integer) using model.state_explicit_repr.
    - All state membership checks use bitwise operations on fact.global_id for speed.
    - Distances computed on the static undirected room graph induced by 'door' facts, ignoring 'closed'.
    - This heuristic is admissible and fast; it purposefully underestimates by ignoring door openings and
      deadhead moves between successive packages (beyond the first initial approach), but it captures key
      domain structure to guide search effectively, especially in large instances.
    """

    def __init__(self, model, robot_gpt5_heuristic_13: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.config = robot_gpt5_heuristic_13 or {}

        # Indexes
        self.in_facts_by_obj_room: Dict[Tuple[str, str], int] = {}
        self.in_facts_by_obj: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self.holding_fact_by_obj: Dict[str, int] = {}
        self.rloc_fact_by_room: Dict[str, int] = {}
        self.rooms: Set[str] = set()
        self.packages_in_goals: Set[str] = set()
        self.goal_room_of_obj: Dict[str, str] = {}
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.dist: Dict[str, Dict[str, int]] = {}

        # Preprocess model facts to build indices and graph
        self._preprocess_model()

    # ---------- Core Heuristic API ----------
    def h(self, state_bitwise: int, task_network: List = None) -> int:
        # Fast goal reached check (if the planner provides task_network)
        try:
            if self.model.goal_reached(state_bitwise, task_network or []):
                return 0
        except Exception:
            # Fallback: if goals subset of state, we still return 0
            if (self.model.goals & ~state_bitwise) == 0:
                return 0

        robot_loc = self._get_robot_loc(state_bitwise)
        held_pkg = self._get_held_object(state_bitwise)

        total_moves_carrying = 0
        total_pick_put = 0

        misplaced_nonheld_rooms: List[str] = []  # current rooms of misplaced, non-held packages

        for obj in self.packages_in_goals:
            goal_room = self.goal_room_of_obj.get(obj)
            if goal_room is None:
                continue  # Safety

            # If already at goal room, skip
            in_goal_gid = self.in_facts_by_obj_room.get((obj, goal_room))
            if in_goal_gid is not None and self._is_true(state_bitwise, in_goal_gid):
                continue

            if held_pkg == obj:
                # Package currently held: carrying distance is from robot_loc to goal_room
                d = self._dist(robot_loc, goal_room)
                total_moves_carrying += d
                total_pick_put += 1  # putdown eventually
            else:
                # Not held: find current location
                cur_room = self._get_object_room(state_bitwise, obj)
                if cur_room is None:
                    # If not found among 'in' facts, it's probably held (handled above), else ignore safely
                    continue
                d = self._dist(cur_room, goal_room)
                total_moves_carrying += d
                total_pick_put += 2  # pickup + putdown
                misplaced_nonheld_rooms.append(cur_room)

        # Initial approach: if not holding any package and there exists at least one misplaced non-held package,
        # the robot must reach one of them before pickup. Add minimal such distance.
        initial_approach = 0
        if held_pkg is None and misplaced_nonheld_rooms:
            # If robot already at one misplaced package location, cost 0; else add min distance
            initial_approach = min(self._dist(robot_loc, r) for r in misplaced_nonheld_rooms)

        h_val = total_moves_carrying + total_pick_put + initial_approach
        # Ensure heuristic is non-negative integer
        return max(0, int(h_val))

    # ---------- Preprocessing ----------
    def _preprocess_model(self):
        # Build goal mapping from model.goals (bitwise integer)
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'in' and len(args) == 2:
                obj, room = args[0], args[1]
                self.packages_in_goals.add(obj)
                self.goal_room_of_obj[obj] = room

        # Iterate all facts once to collect global_ids and static structures
        init_state = getattr(self.model, 'initial_state', 0)
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if not pred:
                continue

            if pred == 'in' and len(args) == 2:
                obj, room = args[0], args[1]
                self.in_facts_by_obj_room[(obj, room)] = fact.global_id
                self.in_facts_by_obj[obj].append((room, fact.global_id))

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.holding_fact_by_obj[obj] = fact.global_id

            elif pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rloc_fact_by_room[room] = fact.global_id
                self.rooms.add(room)

            elif pred == 'door' and len(args) == 3:
                # Only include true door facts from the initial state (doors are static)
                if self._is_true(init_state, fact.global_id):
                    r1, r2 = args[0], args[1]
                    self.rooms.add(r1)
                    self.rooms.add(r2)
                    self.graph[r1].add(r2)
                    self.graph[r2].add(r1)

        # Compute all-pairs shortest path lengths over room graph (ignoring closed)
        self._compute_all_pairs_shortest_paths()

    def _compute_all_pairs_shortest_paths(self):
        # BFS from each room
        for src in self.rooms:
            self.dist[src] = {}
            # Standard BFS over unweighted graph
            q = deque([src])
            self.dist[src][src] = 0
            while q:
                u = q.popleft()
                for v in self.graph.get(u, []):
                    if v not in self.dist[src]:
                        self.dist[src][v] = self.dist[src][u] + 1
                        q.append(v)

    # ---------- Helpers ----------
    @staticmethod
    def _parse_fact_name(name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        if name.startswith('+') or name.startswith('-'):
            name = name[1:]
            bracket_idx = name.find('[')
            if bracket_idx > 0 and name.endswith(']'):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(',') if args_str else []
                return predicate, args
        return None, []

    @staticmethod
    def _is_true(state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    def _get_robot_loc(self, state_bitwise: int) -> Optional[str]:
        for room, gid in self.rloc_fact_by_room.items():
            if self._is_true(state_bitwise, gid):
                return room
        return None

    def _get_held_object(self, state_bitwise: int) -> Optional[str]:
        for obj, gid in self.holding_fact_by_obj.items():
            if self._is_true(state_bitwise, gid):
                return obj
        return None

    def _get_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Returns the room where obj is currently located, or None if not in any (likely held)
        for room, gid in self.in_facts_by_obj.get(obj, []):
            if self._is_true(state_bitwise, gid):
                return room
        return None

    def _dist(self, r1: Optional[str], r2: Optional[str]) -> int:
        if r1 is None or r2 is None:
            return 0
        if r1 == r2:
            return 0
        return self.dist.get(r1, {}).get(r2, 0)

    # Optional: name accessor for logging/debug
    def name(self) -> str:
        return 'RobotGPT5Heuristic13'
