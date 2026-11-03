from typing import Dict, List, Tuple, Optional, Set
import heapq

# Robust import of Heuristic base
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic3(Heuristic):
    """
    RobotGPT5Heuristic3 — Domain-dependent heuristic for the Robot HDDL domain.

    Design highlights (admissible, state-aware, hierarchy-aware via effects of primitives):
    - Computes a strict lower bound on remaining primitive actions by leveraging domain structure:
      * Action necessities per package: each unsatisfied package requires at least a putdown; if not currently held,
        it also requires a pickup. So: 2 per unsatisfied package, except the currently held one (if any) which
        needs only 1 remaining action (putdown). This respects the capacity constraint (single arm) and the
        hierarchical decomposition that ultimately must refine to these primitives.
      * Minimal movement/opening to accomplish at least one immediate delivery:
        - If holding an object H, the robot must first reach goal(H) to free the arm; add the shortest path cost
          (where crossing an open door costs 1 move; a closed door costs 2: open+move). This is a safe lower bound.
        - Else, if not holding, delivering at least one unsatisfied package requires moving from current room to some
          package's room, then to its goal room. We add the minimum such cost over all unsatisfied packages (again
          counting closed-door crossings as 2 and open as 1). We only charge movement for the first delivery to
          keep admissibility (subsequent moves could be zero in the best case).
    - All calculations use fast bitwise state checks via global_id; goal mapping is derived once from model.goals.
    - Graph distances are computed on-demand with Dijkstra using dynamic edge weights reflecting the current state's
      door openness.

    This heuristic is consistent and admissible, while providing strong guidance in large instances by combining
    per-package action necessities with a realistic first-delivery movement lower bound that accounts for door states.
    """

    def __init__(self, model, robot_gpt5_heuristic_3: bool = True):
        super().__init__(model)
        self.robot_gpt5_heuristic_3 = robot_gpt5_heuristic_3

        # Indices for fast lookup
        self.rooms: Set[str] = set()
        self.objects: Set[str] = set()

        # Facts indices (global_id for fast bit checks)
        self.armempty_gid: Optional[int] = None
        self.rloc_gid_by_room: Dict[str, int] = {}
        self.in_gid: Dict[str, Dict[str, int]] = {}         # obj -> room -> gid
        self.holding_gid_by_obj: Dict[str, int] = {}        # obj -> gid
        self.closed_gid_by_door: Dict[str, int] = {}        # door -> gid

        # Doors/graph: room -> List[(neighbor_room, door_id)]
        self.graph: Dict[str, List[Tuple[str, str]]] = {}

        # Goal mapping: object -> goal_room (parsed from model.goals which stores +in[obj,room] goals)
        self.goal_room_by_obj: Dict[str, str] = {}
        self._build_indices()

    def get_name(self) -> str:
        return "robot_gpt5_heuristic_3"

    # ---------------------------- Core Heuristic ---------------------------- #
    def evaluate(self, state_bitwise: int, task_network=None) -> int:
        # Identify current robot room
        current_room = self._get_current_room(state_bitwise)
        if current_room is None:
            # If unknown, provide neutral lower bound 0 to stay safe
            return 0

        # Identify currently held object (if any)
        held_obj = self._get_held_object(state_bitwise)

        # Gather unsatisfied objects and compute remaining pickup/putdown necessities
        unsatisfied_not_held: List[str] = []
        base_actions = 0

        # Held object, if any, always needs at least one putdown
        if held_obj is not None:
            base_actions += 1

        # For all other objects, check if already at goal room
        for obj in self.objects:
            if obj == held_obj:
                continue
            goal_room = self.goal_room_by_obj.get(obj)
            if goal_room is None:
                continue
            obj_room = self._get_object_room(state_bitwise, obj)
            # If object is held, obj_room is None; but we filtered held above
            if obj_room == goal_room:
                continue  # already satisfied
            # Each unsatisfied and not-held object needs at least pickup + putdown
            base_actions += 2
            unsatisfied_not_held.append(obj)

        # Early exit: if no more actions obviously needed
        if base_actions == 0:
            return 0

        # Movement/opening lower bound for the immediate next necessary delivery step
        movement_lb = 0
        if held_obj is not None:
            # Must first reach goal of held_obj to free arm
            goal_room = self.goal_room_by_obj.get(held_obj)
            if goal_room is not None and current_room is not None:
                dist_from_current = self._dists_from(state_bitwise, current_room)
                d = dist_from_current.get(goal_room, float('inf'))
                if d != float('inf'):
                    movement_lb = d
                else:
                    # If unreachable under current weighting, keep LB at 0 (admissible fallback)
                    movement_lb = 0
        else:
            # Not holding: at least one delivery implies: go to some package's room and then to its goal
            if unsatisfied_not_held:
                dist_from_current = self._dists_from(state_bitwise, current_room)
                best = float('inf')
                # Evaluate minimal over all candidate packages
                for obj in unsatisfied_not_held:
                    obj_room = self._get_object_room(state_bitwise, obj)
                    goal_room = self.goal_room_by_obj.get(obj)
                    if obj_room is None or goal_room is None:
                        continue
                    d1 = dist_from_current.get(obj_room, float('inf'))
                    if d1 == float('inf'):
                        continue
                    # Distance from object's room to its goal under current door states
                    d2 = self._dist_between(state_bitwise, obj_room, goal_room)
                    if d2 == float('inf'):
                        continue
                    total = d1 + d2
                    if total < best:
                        best = total
                if best != float('inf'):
                    movement_lb = int(best)
                else:
                    movement_lb = 0  # conservative fallback

        h_val = int(base_actions + movement_lb)
        if h_val < 0:
            h_val = 0
        return h_val

    # ---------------------------- Parsing and Indices ---------------------------- #
    def _build_indices(self):
        # 1) Goal mapping from model.goals (bitwise) -> explicit facts list
        goal_fact_names = self.model.state_explicit_repr(self.model.goals)
        for name in goal_fact_names:
            pred, args = self._parse_fact_name(name)
            if pred == 'in' and len(args) == 2:
                obj, room = args
                self.goal_room_by_obj[obj] = room
                self.objects.add(obj)
                self.rooms.add(room)

        # 2) Scan all facts to build indices
        # The model keeps facts in a list addressed by global_id
        for fact in self.model.facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue
            if pred == 'armempty':
                self.armempty_gid = fact.global_id
            elif pred == 'rloc' and len(args) == 1:
                room = args[0]
                self.rooms.add(room)
                self.rloc_gid_by_room[room] = fact.global_id
            elif pred == 'in' and len(args) == 2:
                obj, room = args
                self.rooms.add(room)
                if obj not in self.in_gid:
                    self.in_gid[obj] = {}
                self.in_gid[obj][room] = fact.global_id
                self.objects.add(obj)
            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.holding_gid_by_obj[obj] = fact.global_id
                self.objects.add(obj)
            elif pred == 'closed' and len(args) == 1:
                door = args[0]
                self.closed_gid_by_door[door] = fact.global_id
            elif pred == 'door' and len(args) == 3:
                r1, r2, d = args
                self.rooms.add(r1)
                self.rooms.add(r2)
                if r1 not in self.graph:
                    self.graph[r1] = []
                self.graph[r1].append((r2, d))
                # Note: the HDDL provides both directions explicitly in instances,
                # but we build adjacency per provided 'door' facts, which may include both.

        # Ensure every room appears in the graph dict
        for r in list(self.rooms):
            self.graph.setdefault(r, [])

        # Any object name found in goals or facts is in self.objects
        # done.

    @staticmethod
    def _parse_fact_name(name: str):
        name = name.strip()
        if name.startswith("+") or name.startswith("-"):
            name = name[1:]
            bracket_idx = name.find("[")
            if bracket_idx > 0 and name.endswith("]"):
                predicate = name[:bracket_idx]
                args_str = name[bracket_idx + 1:-1]
                args = args_str.split(",") if args_str else []
                return predicate, args
        return None, []

    # ---------------------------- State Queries ---------------------------- #
    def _bit_is_true(self, state_bitwise: int, gid: int) -> bool:
        return ((state_bitwise >> gid) & 1) == 1

    def _get_current_room(self, state_bitwise: int) -> Optional[str]:
        for room, gid in self.rloc_gid_by_room.items():
            if self._bit_is_true(state_bitwise, gid):
                return room
        return None

    def _get_held_object(self, state_bitwise: int) -> Optional[str]:
        for obj, gid in self.holding_gid_by_obj.items():
            if self._bit_is_true(state_bitwise, gid):
                return obj
        return None

    def _get_object_room(self, state_bitwise: int, obj: str) -> Optional[str]:
        # If held, there is no in[obj, room] fact true
        gid_h = self.holding_gid_by_obj.get(obj)
        if gid_h is not None and self._bit_is_true(state_bitwise, gid_h):
            return None
        room_map = self.in_gid.get(obj, {})
        for room, gid in room_map.items():
            if self._bit_is_true(state_bitwise, gid):
                return room
        return None

    # ---------------------------- Distances ---------------------------- #
    def _edge_cost(self, state_bitwise: int, door_id: str) -> int:
        # Crossing an open door costs 1 (a move); crossing a closed door costs 2 (open + move)
        gid = self.closed_gid_by_door.get(door_id)
        if gid is None:
            # If unknown closed-fact, be safely optimistic: assume move only
            return 1
        return 2 if self._bit_is_true(state_bitwise, gid) else 1

    def _dists_from(self, state_bitwise: int, src: str) -> Dict[str, int]:
        # Dijkstra with small graph; dynamic edge weights based on current state
        dist: Dict[str, int] = {r: float('inf') for r in self.rooms}
        if src not in dist:
            return dist
        dist[src] = 0
        heap: List[Tuple[int, str]] = [(0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for v, door_id in self.graph.get(u, []):
                w = self._edge_cost(state_bitwise, door_id)
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    def _dist_between(self, state_bitwise: int, src: str, dst: str) -> int:
        if src == dst:
            return 0
        dist = self._dists_from(state_bitwise, src)
        return dist.get(dst, float('inf'))
