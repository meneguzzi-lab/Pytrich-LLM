# Domain-dependent heuristic for the HDDL Robot domain in the Pytrich Planner framework
# Heuristic name: RobotGPT5Heuristic19
#
# Rationale (high level):
# - Each unsatisfied package must, at minimum, be put down at its goal room (1 action), and if not
#   already carried, first picked up (1 action). That yields a baseline of 2 actions per remaining
#   package except for the currently-held one, which only needs a putdown.
# - While carrying, the robot must traverse a path from the package's pickup room to its goal room.
#   Summing these per-package carrying distances (in number of room-to-room moves) is a valid lower
#   bound on the total number of move actions, since each package’s carrying segment must be executed.
#   For a currently held package, the segment is from the robot’s current room to its goal room.
# - Before moving the first (not-yet-held) package, the robot must reach some pickup room; so we add
#   the minimal distance from the robot’s current room to any needed pickup location (lower bound).
# - Doors may be closed and must be opened at least once before a move can cross them. To keep the
#   heuristic admissible yet informative, we add only the minimal number of opens required to handle
#   the first package (either the currently held one, or the best first pick), computed with a 0-1 BFS
#   where edges cost 1 if their door is closed and 0 if open. We avoid summing open counts over all
#   packages since that can overestimate (inadmissible) when the same opened door is reused.
# - This combines hierarchical awareness (pickup/putdown counts) with domain navigation structure
#   (graph distances and closed-door costs) while remaining computationally light.
#
# Important implementation details for Pytrich:
# - Facts are accessed via bitwise checks using fact.global_id. NEVER use explicit string checks on
#   the state representation (too slow).
# - Goals are retrieved from model.goals via model.state_explicit_repr, but for efficiency we parse
#   static goal_in facts once to map each package to its goal room. We check goal satisfaction per
#   package via the single in[obj,goal_room] fact bit.
# - We precompute the directed adjacency graph from door facts and all-pairs shortest path distances
#   (edge count, ignoring closed/open). For open counts, we perform on-the-fly 0-1 BFS per start room.
#
# Notes:
# - The heuristic is designed to be a consistent lower bound. It purposefully does not count moves
#   between completing one package and starting the next (beyond the initial min distance to a first
#   pickup). It also only counts open actions for the first package. Both choices keep it admissible.

from collections import deque, defaultdict
from typing import Dict, Tuple, List, Set, Optional

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic19(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    h(state, tn) combines:
    - 2 per unsatisfied package (pickup + putdown), except the currently-held package needs only 1 (putdown).
    - Sum over packages of the minimal number of moves (room-to-room steps) while carrying each package:
      For held: dist(robot_room, goal_room) ; otherwise: dist(current_room_of_obj, goal_room).
    - Plus the minimal distance from the robot's current room to any first pickup room (if no package held).
    - Plus the minimal number of door opens needed to complete the very first package (held or best first pick),
      computed with a 0-1 BFS over the door graph using current closed-door facts.

    All components are lower bounds => heuristic is admissible. It is also efficient for up to 100 packages / 50 rooms.
    """

    def __init__(self, model, robot_gpt5_heuristic_19: bool = True):
        super().__init__(model)

        # Internal parsed structures
        self.rooms: Set[str] = set()
        self.objects: Set[str] = set()
        self.doors: Set[str] = set()

        # Fact dictionaries
        self.in_fact_by_obj_room: Dict[Tuple[str, str], object] = {}
        self.in_facts_per_obj: Dict[str, List[Tuple[str, object]]] = defaultdict(list)  # obj -> list of (room, Fact)
        self.rloc_fact_by_room: Dict[str, object] = {}
        self.holding_fact_by_obj: Dict[str, object] = {}
        self.closed_fact_by_door: Dict[str, object] = {}
        self.goal_room_by_obj: Dict[str, str] = {}

        # Adjacency: room -> List[(neighbor_room, door_id)]
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        # Precompute: parse model facts
        self._parse_model_facts()

        # Precompute all-pairs shortest path distances (#edges) ignoring closed/open status
        self._dist_edges: Dict[str, Dict[str, int]] = self._compute_all_pairs_unweighted_distances()

    # --------------- Parsing helpers ---------------
    @staticmethod
    def _parse_fact_name(name: str):
        # Name format: '+predicate[arg1,arg2]' (as per problem statement)
        s = name.strip()
        if s.startswith("+") or s.startswith("-"):
            s = s[1:]
            idx = s.find("[")
            if idx > 0 and s.endswith("]"):
                predicate = s[:idx]
                args_str = s[idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _parse_model_facts(self):
        # Iterate once through all facts in the model
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == "in" and len(args) == 2:
                obj, room = args[0], args[1]
                self.objects.add(obj)
                self.rooms.add(room)
                self.in_fact_by_obj_room[(obj, room)] = fact
                self.in_facts_per_obj[obj].append((room, fact))

            elif pred == "rloc" and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_fact_by_room[room] = fact

            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                self.objects.add(obj)
                self.holding_fact_by_obj[obj] = fact

            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self.doors.add(d)
                self.closed_fact_by_door[d] = fact

            elif pred == "door" and len(args) == 3:
                loc1, loc2, d = args[0], args[1], args[2]
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.doors.add(d)
                # Directed edge as defined by the presence of this fact
                self.adjacency[loc1].append((loc2, d))

            elif pred == "goal_in" and len(args) == 2:
                obj, room = args[0], args[1]
                self.objects.add(obj)
                self.rooms.add(room)
                # If multiple goal_in facts exist for the same obj (unlikely), keep the last one
                self.goal_room_by_obj[obj] = room

        # Ensure all rooms are keys in adjacency
        for r in list(self.rooms):
            if r not in self.adjacency:
                self.adjacency[r] = []

    # --------------- Utility bitwise checks ---------------
    @staticmethod
    def _is_true(bit_state: int, fact) -> bool:
        return ((bit_state >> fact.global_id) & 1) == 1

    def _current_robot_room(self, state: int) -> Optional[str]:
        # Find which rloc[room] is true
        for room, fact in self.rloc_fact_by_room.items():
            if self._is_true(state, fact):
                return room
        return None

    def _current_holding_object(self, state: int) -> Optional[str]:
        for obj, fact in self.holding_fact_by_obj.items():
            if self._is_true(state, fact):
                return obj
        return None

    def _object_current_room(self, obj: str, state: int, robot_room: str, held_obj: Optional[str]) -> Optional[str]:
        # If currently holding this object, treat its location as the robot's room (carried)
        if held_obj is not None and obj == held_obj:
            return robot_room
        # Else find which in[obj, room] fact is true
        for room, fact in self.in_facts_per_obj.get(obj, []):
            if self._is_true(state, fact):
                return room
        return None

    # --------------- Distances (unweighted edges) ---------------
    def _compute_all_pairs_unweighted_distances(self) -> Dict[str, Dict[str, int]]:
        # BFS from each room using adjacency edges, ignoring closed/open (presence of edges only)
        dists: Dict[str, Dict[str, int]] = {r: {r2: (0 if r == r2 else 10**9) for r2 in self.rooms} for r in self.rooms}
        for src in self.rooms:
            q = deque([src])
            dists[src][src] = 0
            while q:
                u = q.popleft()
                du = dists[src][u]
                for v, _door in self.adjacency[u]:
                    if dists[src][v] > du + 1:
                        dists[src][v] = du + 1
                        q.append(v)
        return dists

    # --------------- 0-1 BFS for minimal closed-door opens ---------------
    def _compute_zero_one_bfs(self, start_room: str, state: int) -> Dict[str, int]:
        # Edge cost is 1 if the door is currently closed in this state, else 0.
        # 0-1 BFS over small graphs is very fast and precise for this component.
        from collections import deque as _deque

        # Precompute door-closed boolean for quick lookup
        is_closed: Dict[str, bool] = {}
        for d, f in self.closed_fact_by_door.items():
            is_closed[d] = self._is_true(state, f)

        INF = 10**9
        dist_closed: Dict[str, int] = {r: INF for r in self.rooms}
        dist_closed[start_room] = 0
        dq = _deque([start_room])

        while dq:
            u = dq.popleft()
            du = dist_closed[u]
            for v, d in self.adjacency[u]:
                w = 1 if is_closed.get(d, False) else 0
                nd = du + w
                if nd < dist_closed[v]:
                    dist_closed[v] = nd
                    if w == 0:
                        dq.appendleft(v)
                    else:
                        dq.append(v)
        return dist_closed

    # --------------- Main heuristic ---------------
    def __call__(self, state_bitwise: int, task_network: List[int]) -> int:
        return self.evaluate(state_bitwise, task_network)

    def evaluate(self, state_bitwise: int, task_network: List[int]) -> int:
        # Current robot room and possibly held object
        robot_room = self._current_robot_room(state_bitwise)
        if robot_room is None:
            # If unknown, return 0 to stay safe (admissible). Should not happen in valid instances.
            return 0
        held_obj = self._current_holding_object(state_bitwise)

        # Gather all objects that have a goal
        goal_objs = [obj for obj in self.objects if obj in self.goal_room_by_obj]

        # Identify unsatisfied objects, and simultaneously compute their current room if needed later
        unsatisfied: List[str] = []
        obj_current_room: Dict[str, Optional[str]] = {}

        for obj in goal_objs:
            goal_room = self.goal_room_by_obj[obj]
            in_goal_fact = self.in_fact_by_obj_room.get((obj, goal_room))
            satisfied = False
            if in_goal_fact is not None:
                satisfied = self._is_true(state_bitwise, in_goal_fact)
            # If object is held, it cannot be "in" any room -> unsatisfied
            if not satisfied:
                unsatisfied.append(obj)

        if not unsatisfied:
            # All goals met in the current state; remaining abstract tasks (if any) can be decomposed to empty at zero cost
            return 0

        # Compute pick/put action count (lower bound)
        putdowns_needed = len(unsatisfied)
        pickups_needed = len(unsatisfied) - (1 if held_obj is not None else 0)
        if pickups_needed < 0:
            pickups_needed = 0
        action_cost_baseline = putdowns_needed + pickups_needed

        # For move distances, we need each object's current room (held: robot_room)
        # and its goal room. Compute lazily.
        total_carry_moves = 0
        for obj in unsatisfied:
            goal_room = self.goal_room_by_obj[obj]
            if held_obj is not None and obj == held_obj:
                src_room = robot_room
            else:
                # Find current room of the object
                src_room = self._object_current_room(obj, state_bitwise, robot_room, held_obj)
                if src_room is None:
                    # If unknown (shouldn't happen), skip contribution (keeps admissible)
                    continue
            # Distance in edges ignoring closed/open
            d = self._dist_edges.get(src_room, {}).get(goal_room, 10**9)
            if d >= 10**9:
                # If unreachable per our static graph (should not happen in solvable problems),
                # do not add anything (keep lower bound finite).
                d = 0
            total_carry_moves += d

        # Minimal distance from robot to any first pickup room (if no object is held)
        reach_first_pick = 0
        if held_obj is None:
            best = 10**9
            for obj in unsatisfied:
                # Need current room for the object (not held here)
                cur = self._object_current_room(obj, state_bitwise, robot_room, held_obj)
                if cur is None:
                    continue
                d = self._dist_edges.get(robot_room, {}).get(cur, 10**9)
                if d < best:
                    best = d
            if best >= 10**9:
                best = 0
            reach_first_pick = best

        # Minimal number of OPEN actions needed for the first package only (admissible)
        min_first_open_cost = 0
        if held_obj is not None:
            # First package is the held one: need opens on path robot_room -> goal_room(held)
            goal_room = self.goal_room_by_obj[held_obj]
            dist_closed_from_robot = self._compute_zero_one_bfs(robot_room, state_bitwise)
            c = dist_closed_from_robot.get(goal_room, 10**9)
            if c >= 10**9:
                c = 0
            min_first_open_cost = c
        else:
            # Choose the object that minimizes opens: robot->pickup + pickup->goal
            best = 10**9
            dist_closed_from_robot = self._compute_zero_one_bfs(robot_room, state_bitwise)
            for obj in unsatisfied:
                pick_room = self._object_current_room(obj, state_bitwise, robot_room, held_obj)
                if pick_room is None:
                    continue
                # If robot cannot reach pick_room without opens considered, we still compute opens via 0-1 BFS
                c1 = dist_closed_from_robot.get(pick_room, 10**9)
                if c1 >= 10**9:
                    # No path -> skip; but in solvable instances, there should be one
                    continue
                # Opens from pick to goal
                dist_closed_from_pick = self._compute_zero_one_bfs(pick_room, state_bitwise)
                goal_room = self.goal_room_by_obj[obj]
                c2 = dist_closed_from_pick.get(goal_room, 10**9)
                if c2 >= 10**9:
                    continue
                if c1 + c2 < best:
                    best = c1 + c2
            if best >= 10**9:
                best = 0
            min_first_open_cost = best

        heuristic_value = action_cost_baseline + total_carry_moves + reach_first_pick + min_first_open_cost
        if heuristic_value < 0:
            heuristic_value = 0
        return int(heuristic_value)


# Alias for potential external reference (optional)
HeuristicClass = RobotGPT5Heuristic19
