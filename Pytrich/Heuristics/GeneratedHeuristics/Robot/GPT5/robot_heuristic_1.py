from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import math

from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic1(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Rationale (admissible lower bound):
    - Each package that is not already at its goal must be manipulated:
      at least a pickup (1 action) and a putdown (1 action). If the robot is
      already holding that specific package, only a putdown is needed (1 action).
      Summing these manipulation lower bounds over all packages is admissible.
    - Movement is a significant cost driver. To keep the heuristic admissible
      and efficient, we add a conservative movement lower bound for only the
      next inevitable movement segment (not for all packages):
        * If holding some package p, add the shortest path length (in number of
          room-to-room moves, ignoring door closures) from the robot's current
          room to p's goal room.
        * Otherwise (not holding), add the minimum, over all packages that are
          not at goal, of [distance(robot_room, package_room) +
          distance(package_room, package_goal_room)].
      - Distances ignore closed doors (i.e., assume all are open) and count 1
        per move. Ignoring door openings and sharing of opened doors keeps the
        estimate a lower bound (admissible) and avoids double counting.

    This heuristic is designed to be:
      - Admissible and consistent in practice
      - Fast: precomputes an all-pairs shortest path (BFS on directed graph)
      - Scalable to large instances (50 rooms, 100 packages)

    Notes on implementation details:
      - Facts are accessed via bitwise checks using their global_id (fast).
      - Goal facts are parsed from model.goals via model.state_explicit_repr().
      - The robot-room graph is built from 'door[room1,room2,doorId]' facts.
      - Distances are computed once (ignoring door closures) and cached.
      - The heuristic gracefully handles different calling styles by trying to
        extract (state_bitwise, task_network) from a variety of node formats.

    Class name and init param follow the task requirements:
      - Class name: RobotGPT5Heuristic1
      - __init__ parameter name: robot_gpt5_heuristic_1
    """

    def __init__(self, model, robot_gpt5_heuristic_1: bool = True):
        super().__init__(model=model, name="RobotGPT5Heuristic1")

        # Containers
        self.rooms: List[str] = []
        self.room_index: Dict[str, int] = {}
        self.objects: List[str] = []

        # Fast fact-id maps
        self.armempty_id: Optional[int] = None
        self.holding_id_by_obj: Dict[str, int] = {}
        self.rloc_id_by_room: Dict[str, int] = {}
        self.in_id_by_obj_room: Dict[Tuple[str, str], int] = {}
        self.in_ids_per_obj: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

        # Door graph info (for completeness; closures are ignored in distance comp)
        self.closed_id_by_door: Dict[str, int] = {}
        self.adj: Dict[str, List[str]] = defaultdict(list)  # directed adjacency

        # Goals: obj -> goal_room (parsed from model.goals via state_explicit_repr)
        self.goal_room_of: Dict[str, str] = {}

        # Parse facts and build internal structures
        self._scan_facts_and_build_maps()
        self._parse_goals_from_model()

        # Precompute all-pairs shortest paths on the directed room graph,
        # ignoring door closures (each move has cost 1)
        self._all_pairs_dist: Dict[str, Dict[str, int]] = self._precompute_all_pairs_shortest_paths()

    # ===================== Core Heuristic API =====================
    def evaluate(self, *args, **kwargs) -> int:
        state_bitwise, task_network = self._extract_state_and_tn(*args, **kwargs)
        return self._heuristic(state_bitwise, task_network)

    __call__ = evaluate

    # ===================== Heuristic computation =====================
    def _heuristic(self, state_bitwise: int, task_network: Optional[List[Any]]) -> int:
        if state_bitwise is None:
            return 0

        # 1) Identify robot's current room
        robot_room = self._get_robot_room(state_bitwise)

        # 2) Identify if holding an object, and build baseline manipulation LB
        held_obj = None
        baseline = 0
        for obj in self.objects:
            # If object already at its goal: 0
            goal_r = self.goal_room_of.get(obj)
            if goal_r is None:
                # If goals were not provided for some obj, be safe (treat as 0)
                continue

            in_goal_fact_id = self.in_id_by_obj_room.get((obj, goal_r))
            is_at_goal = in_goal_fact_id is not None and self._is_true(state_bitwise, in_goal_fact_id)

            if is_at_goal:
                # No manipulation needed
                continue

            # If holding this object: only need putdown
            hid = self.holding_id_by_obj.get(obj)
            if hid is not None and self._is_true(state_bitwise, hid):
                baseline += 1
                # Domain has single-arm; record the held object
                if held_obj is None:
                    held_obj = obj
                continue

            # Otherwise, at least pickup + putdown (ignore movement for now)
            baseline += 2

        # Early exit: if all packages are at goals and no one is held, movement lb = 0
        if baseline == 0:
            return 0

        # 3) Add a conservative movement LB for just the next inevitable motion
        move_lb = 0
        if robot_room is not None:
            if held_obj is not None:
                # Need to reach the goal room of the held object
                g_r = self.goal_room_of.get(held_obj)
                if g_r is not None:
                    d = self._dist(robot_room, g_r)
                    if d < math.inf:
                        move_lb += d
            else:
                # Not holding: consider the cheapest object to complete next:
                # min over objects not at goal and not held of
                #   dist(robot_room, obj_room) + dist(obj_room, goal_room)
                best = math.inf
                for obj in self.objects:
                    g_r = self.goal_room_of.get(obj)
                    if g_r is None:
                        continue
                    in_goal_fact_id = self.in_id_by_obj_room.get((obj, g_r))
                    is_at_goal = in_goal_fact_id is not None and self._is_true(state_bitwise, in_goal_fact_id)
                    if is_at_goal:
                        continue
                    # Skip if (somehow) holding this obj
                    hid = self.holding_id_by_obj.get(obj)
                    if hid is not None and self._is_true(state_bitwise, hid):
                        continue

                    obj_room = self._get_object_room(state_bitwise, obj)
                    if obj_room is None:
                        # Could be currently held (already filtered) or inconsistent state; skip
                        continue

                    d1 = self._dist(robot_room, obj_room)
                    d2 = self._dist(obj_room, g_r)
                    if d1 < math.inf and d2 < math.inf:
                        cand = d1 + d2
                        if cand < best:
                            best = cand
                if best < math.inf:
                    move_lb += best

        # Final heuristic: manipulation LB for all objects + one-segment movement LB
        h_val = baseline + move_lb
        if h_val < 0:
            # Safety
            h_val = 0
        return int(h_val)

    # ===================== Helper methods =====================
    def _extract_state_and_tn(self, *args, **kwargs) -> Tuple[Optional[int], Optional[List[Any]]]:
        """
        Flexible extraction of (state_bitwise, task_network) from various calling styles:
        - evaluate(state_bitwise, task_network)
        - evaluate(node) where node may have attributes like state, state_bitwise, task_network, tn, tasks
        - evaluate(state_bitwise)
        """
        state_bitwise = None
        task_network = None

        # Direct tuple
        if len(args) >= 2 and isinstance(args[0], int):
            state_bitwise = args[0]
            task_network = args[1]
            return state_bitwise, task_network

        # Single argument: try to interpret node or state
        if len(args) == 1:
            node = args[0]
            if isinstance(node, int):
                state_bitwise = node
            else:
                # Try common attributes
                for attr in ("state_bitwise", "state", "s"):
                    if hasattr(node, attr):
                        sb = getattr(node, attr)
                        if isinstance(sb, int):
                            state_bitwise = sb
                            break
                for attr in ("task_network", "tn", "tasks"):
                    if hasattr(node, attr):
                        task_network = getattr(node, attr)
                        break

        # kwargs fallbacks
        if state_bitwise is None:
            state_bitwise = kwargs.get("state_bitwise", kwargs.get("state", None))
        if task_network is None:
            task_network = kwargs.get("task_network", kwargs.get("tn", None))

        return state_bitwise, task_network

    def _parse_fact_name(self, name: str) -> Tuple[Optional[str], List[str]]:
        name = name.strip()
        # Format: +predicate[arg1,arg2] or -predicate[arg1,arg2]
        if name.startswith("+") or name.startswith("-"):
            name_ = name[1:]
            bracket_idx = name_.find("[")
            if bracket_idx > 0 and name_.endswith("]"):
                predicate = name_[:bracket_idx]
                args_str = name_[bracket_idx + 1 : -1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    def _is_true(self, state_bitwise: int, fact_global_id: int) -> bool:
        return ((state_bitwise >> fact_global_id) & 1) == 1

    def _scan_facts_and_build_maps(self):
        # Ensure we iterate facts in global_id order
        facts: List[Any] = self.model.facts

        rooms_set = set()
        objects_set = set()

        for f in facts:
            pred, args = self._parse_fact_name(f.name)
            if not pred:
                continue

            if pred == "rloc" and len(args) == 1:
                room = args[0]
                rooms_set.add(room)
                self.rloc_id_by_room[room] = f.global_id

            elif pred == "in" and len(args) == 2:
                obj, room = args
                objects_set.add(obj)
                rooms_set.add(room)
                self.in_id_by_obj_room[(obj, room)] = f.global_id
                self.in_ids_per_obj[obj].append((room, f.global_id))

            elif pred == "holding" and len(args) == 1:
                obj = args[0]
                objects_set.add(obj)
                self.holding_id_by_obj[obj] = f.global_id

            elif pred == "armempty":
                self.armempty_id = f.global_id

            elif pred == "door" and len(args) == 3:
                r1, r2, d = args
                rooms_set.add(r1)
                rooms_set.add(r2)
                # Directed edge r1 -> r2 (move has this direction)
                self.adj[r1].append(r2)
                # Note: do not implicitly add reverse; rely on explicit door facts

            elif pred == "closed" and len(args) == 1:
                d = args[0]
                self.closed_id_by_door[d] = f.global_id

        self.rooms = sorted(list(rooms_set))
        self.objects = sorted(list(objects_set))
        self.room_index = {r: i for i, r in enumerate(self.rooms)}

    def _parse_goals_from_model(self):
        # Goals are stored as a bitwise int. Convert to explicit repr.
        goal_facts = self.model.state_explicit_repr(self.model.goals)
        for gf in goal_facts:
            pred, args = self._parse_fact_name(gf)
            if pred == "in" and len(args) == 2:
                obj, room = args
                self.goal_room_of[obj] = room
        # Some domains might encode additional goal preds; we only use 'in'.

    def _precompute_all_pairs_shortest_paths(self) -> Dict[str, Dict[str, int]]:
        # Distances ignoring door closures; cost per directed edge = 1
        dists: Dict[str, Dict[str, int]] = {r: {s: math.inf for s in self.rooms} for r in self.rooms}
        for r in self.rooms:
            dists[r][r] = 0
            # BFS from r
            q = deque([r])
            while q:
                u = q.popleft()
                du = dists[r][u]
                for v in self.adj.get(u, []):
                    if dists[r][v] > du + 1:
                        dists[r][v] = du + 1
                        q.append(v)
        return dists

    def _dist(self, r1: str, r2: str) -> int:
        if r1 not in self._all_pairs_dist or r2 not in self._all_pairs_dist[r1]:
            return math.inf
        return self._all_pairs_dist[r1][r2]

    def _get_robot_room(self, state_bitwise: int) -> Optional[str]:
        # rloc is a functional fluent: exactly one true in consistent states
        for room, fid in self.rloc_id_by_room.items():
            if self._is_true(state_bitwise, fid):
                return room
        return None

    def _get_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # Find the unique room r where in[obj, r] is true (if any)
        for room, fid in self.in_ids_per_obj.get(obj, []):
            if self._is_true(state_bitwise, fid):
                return room
        return None


# Backwards-compat export name
__all__ = ["RobotGPT5Heuristic1"]
