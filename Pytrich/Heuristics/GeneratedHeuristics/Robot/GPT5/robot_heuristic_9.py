from typing import Dict, List, Tuple, Optional, Set
import heapq

# Flexible import to work within Pytrich
from Pytrich.Heuristics.heuristic import Heuristic
class RobotGPT5Heuristic9(Heuristic):
    """
    Domain-dependent heuristic for the Robot HDDL domain.

    Design summary (admissible lower bound):
    - Each misplaced package requires at least 2 primitive actions (pickup + putdown).
      If the robot is already holding a misplaced package, the pickup for that package
      is not needed, so we subtract 1 action.
    - Additionally, we add a conservative movement/opening lower bound for the immediate
      next progress step:
        * If holding an object x, we must at least travel from current location to x's
          goal room. Travel cost per edge is 1 for move plus 1 if the edge's door is
          currently closed (for an open action). This path cost is computed with Dijkstra.
        * If not holding, we consider the best next object: the minimal cost to reach its
          current location from the robot's current location (Dijkstra as above) plus the
          number of move actions (no open costs) required to go from that object's location
          to its goal. Ignoring opens on the second leg preserves admissibility because
          opens can only increase real cost.

    Hierarchical awareness:
    - The abstract tasks (achieve-goals, release, move_abstract, open_abstract, etc.) refine
      into the above primitive actions. Our cost estimate focuses on the minimal number of
      primitive operators that must still be executed to refine the task network to a goal plan.
      Decompositions are cost-free in HTN, so we do not count those steps.

    Efficiency:
    - We precompute a static, unweighted all-pairs shortest-path map between rooms (edge cost 1),
      ignoring door closed status. This is reused in all heuristic calls for the second-leg move
      estimates.
    - For state-dependent travel costs that include current closed-door penalties, we run a single
      Dijkstra from the current robot room per state evaluation.

    Notes:
    - Facts are accessed via bitwise checks using their global_id for performance.
    - Goals are retrieved via model.state_explicit_repr(model.goals), but this domain encodes goals
      as 'in[obj,goal_room]' facts in the classical goal. The 'goal_in' predicate is a static
      relation present in the initial state and used by methods.
    - This heuristic is admissible and designed to provide strong guidance without heavy per-node overhead.
    """

    def __init__(self, model, robot_gpt5_heuristic_9: Optional[dict] = None):
        super().__init__()
        self.model = model
        self.config = robot_gpt5_heuristic_9 or {}

        # Parsed structures
        self.rooms: Set[str] = set()
        self.objects: Set[str] = set()
        self.doors: Set[str] = set()

        # Adjacency: room -> List[(neighbor_room, door_id)]
        self.adj: Dict[str, List[Tuple[str, str]]] = {}

        # Fact id maps for bitwise access
        self.closed_fact_by_door: Dict[str, int] = {}
        self.rloc_fact_by_loc: Dict[str, int] = {}
        self.holding_fact_by_obj: Dict[str, int] = {}
        self.in_fact_by_obj_loc: Dict[str, Dict[str, int]] = {}
        self.goal_loc_by_obj: Dict[str, str] = {}
        self.in_goal_fact_id_by_obj: Dict[str, int] = {}

        # Precompute static info
        self._analyze_model()

        # Precompute unweighted all-pairs shortest path distances (moves only, no open costs)
        self._INF = 10 ** 9
        self._apsp_moves_only: Dict[str, Dict[str, int]] = self._precompute_unweighted_apsp()

        # Simple cache for last state's Dijkstra distances (closed-aware)
        self._last_state_key: Optional[int] = None
        self._last_source: Optional[str] = None
        self._last_dist_map: Optional[Dict[str, int]] = None

    # --------------------------- Core Heuristic API ---------------------------
    def h(self, state_bitwise: int, task_network: List = None) -> int:
        if task_network is None:
            task_network = []

        # 1) Quickly detect if all goal in-facts (in[obj, goal_room]) are already satisfied
        #    If yes, cost is 0 (decompositions are cost-free)
        unsatisfied_objs: List[str] = []
        for obj, in_goal_fact_id in self.in_goal_fact_id_by_obj.items():
            if not self._is_true(state_bitwise, in_goal_fact_id):
                unsatisfied_objs.append(obj)
        if not unsatisfied_objs:
            return 0

        # 2) Base lower bound: 2 actions per misplaced object (pickup + putdown)
        base_cost = 2 * len(unsatisfied_objs)

        # 3) Identify current robot location and whether holding an object
        current_room = self._get_current_room(state_bitwise)
        held_obj = self._get_holding_obj(state_bitwise)

        # If holding a misplaced object, pickup for that one is not needed
        if held_obj is not None and held_obj in unsatisfied_objs:
            base_cost -= 1

        # 4) Add conservative travel lower bound for the immediate next progress step
        #    Compute closed-aware travel costs from current room to all rooms
        dist_closed_aware = self._dijkstra_from(state_bitwise, current_room)

        travel_lb = 0
        if held_obj is not None and held_obj in unsatisfied_objs:
            # Must at least travel to the goal room of the held object
            goal_r = self.goal_loc_by_obj.get(held_obj)
            if goal_r is not None:
                d1 = dist_closed_aware.get(goal_r, self._INF)
                if d1 < self._INF:
                    travel_lb = d1
                else:
                    travel_lb = 0  # keep admissible if disconnected
        else:
            # Not holding: choose best next object
            best = self._INF
            for obj in unsatisfied_objs:
                obj_loc = self._get_object_location(state_bitwise, obj)
                if obj_loc is None:
                    # If object is currently held, skip (should not happen in this branch)
                    continue
                goal_r = self.goal_loc_by_obj.get(obj)
                if goal_r is None:
                    continue
                # First leg: closed-aware cost from current room to the object's location
                c1 = dist_closed_aware.get(obj_loc, self._INF)
                # Second leg: minimal number of moves (edges) from object to its goal, ignoring opens
                c2 = self._apsp_moves_only.get(obj_loc, {}).get(goal_r, self._INF)
                if c1 >= self._INF or c2 >= self._INF:
                    continue
                cand = c1 + c2
                if cand < best:
                    best = cand
            if best < self._INF:
                travel_lb = best
            else:
                travel_lb = 0

        h_val = base_cost + travel_lb
        if h_val < 0:
            h_val = 0
        return int(h_val)

    # Allow callable usage if the planner expects it
    __call__ = h

    # --------------------------- Model Analysis ---------------------------
    def _analyze_model(self):
        # Parse all facts and build indices/mappings
        facts = getattr(self.model, 'facts', [])

        # Temporary storage for door edges (directed as listed)
        directed_edges: List[Tuple[str, str, str]] = []  # (loc1, loc2, door_id)

        # First pass: collect predicates
        for fact in facts:
            pred, args = self._parse_fact_name(fact.name)
            if pred is None:
                continue

            if pred == 'door' and len(args) == 3:
                loc1, loc2, door_id = args
                self.rooms.add(loc1)
                self.rooms.add(loc2)
                self.doors.add(door_id)
                directed_edges.append((loc1, loc2, door_id))

            elif pred == 'closed' and len(args) == 1:
                d = args[0]
                self.closed_fact_by_door[d] = fact.global_id
                self.doors.add(d)

            elif pred == 'rloc' and len(args) == 1:
                loc = args[0]
                self.rooms.add(loc)
                self.rloc_fact_by_loc[loc] = fact.global_id

            elif pred == 'holding' and len(args) == 1:
                obj = args[0]
                self.objects.add(obj)
                self.holding_fact_by_obj[obj] = fact.global_id

            elif pred == 'in' and len(args) == 2:
                obj, loc = args
                self.rooms.add(loc)
                self.objects.add(obj)
                self.in_fact_by_obj_loc.setdefault(obj, {})[loc] = fact.global_id

            elif pred == 'goal_in' and len(args) == 2:
                obj, loc = args
                self.rooms.add(loc)
                self.objects.add(obj)
                # 'goal_in' is static and known from init; store target room
                self.goal_loc_by_obj[obj] = loc

        # Build undirected adjacency (if only one direction was provided for some edge,
        # we still add that direction as available since moves rely on pairing door facts;
        # in given instances, both directions are typically listed.)
        for r in self.rooms:
            self.adj.setdefault(r, [])
        for (a, b, d) in directed_edges:
            self.adj[a].append((b, d))
            # If the reverse door fact is also present, it will be added during loop
            # But to ensure undirected exploration, we also add reverse if not present
            # under the same door id (safe as lower-bound traversal abstraction)
            # Check whether reverse was explicitly given; if not, still add reverse using same door id.
            # This is safe for a lower bound because adding potential paths cannot overestimate required cost.
            if not any(nbr == a and dd == d for (nbr, dd) in self.adj[b]):
                self.adj[b].append((a, d))

        # Map goal in -> corresponding 'in' fact id for fast satisfied check
        for obj, goal_loc in self.goal_loc_by_obj.items():
            fid = self.in_fact_by_obj_loc.get(obj, {}).get(goal_loc, None)
            if fid is not None:
                self.in_goal_fact_id_by_obj[obj] = fid

    # --------------------------- Distances ---------------------------
    def _precompute_unweighted_apsp(self) -> Dict[str, Dict[str, int]]:
        # BFS from each room, counting edges only (move cost 1), ignoring closed status
        apsp: Dict[str, Dict[str, int]] = {}
        for src in self.rooms:
            dist: Dict[str, int] = {r: self._INF for r in self.rooms}
            dist[src] = 0
            q: List[str] = [src]
            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                du = dist[u]
                for (v, _door) in self.adj.get(u, []):
                    if dist[v] == self._INF:
                        dist[v] = du + 1
                        q.append(v)
            apsp[src] = dist
        return apsp

    def _dijkstra_from(self, state_bitwise: int, src: Optional[str]) -> Dict[str, int]:
        if src is None:
            return {}

        # Simple last-state cache to reduce recomputation in local expansions
        if (
            self._last_state_key == state_bitwise and
            self._last_source == src and
            self._last_dist_map is not None
        ):
            return self._last_dist_map

        dist: Dict[str, int] = {r: self._INF for r in self.rooms}
        dist[src] = 0
        pq: List[Tuple[int, str]] = [(0, src)]

        while pq:
            d_u, u = heapq.heappop(pq)
            if d_u != dist[u]:
                continue
            for (v, door_id) in self.adj.get(u, []):
                # Edge cost: 1 move + 1 open if currently closed
                closed_cost = 1 if self._is_door_closed(state_bitwise, door_id) else 0
                w = d_u + 1 + closed_cost
                if w < dist[v]:
                    dist[v] = w
                    heapq.heappush(pq, (w, v))

        # Update cache
        self._last_state_key = state_bitwise
        self._last_source = src
        self._last_dist_map = dist
        return dist

    # --------------------------- State Queries ---------------------------
    def _get_current_room(self, state_bitwise: int) -> Optional[str]:
        # rloc is unique
        for loc, fid in self.rloc_fact_by_loc.items():
            if self._is_true(state_bitwise, fid):
                return loc
        return None

    def _get_holding_obj(self, state_bitwise: int) -> Optional[str]:
        for obj, fid in self.holding_fact_by_obj.items():
            if self._is_true(state_bitwise, fid):
                return obj
        return None

    def _get_object_location(self, state_bitwise: int, obj: str) -> Optional[str]:
        # If the object is being held, there will be no 'in[obj, loc]' true
        for loc, fid in self.in_fact_by_obj_loc.get(obj, {}).items():
            if self._is_true(state_bitwise, fid):
                return loc
        return None

    def _is_door_closed(self, state_bitwise: int, door_id: str) -> bool:
        fid = self.closed_fact_by_door.get(door_id, None)
        if fid is None:
            return False
        return bool((state_bitwise >> fid) & 1)

    def _is_true(self, state_bitwise: int, fact_global_id: int) -> bool:
        return bool((state_bitwise >> fact_global_id) & 1)

    # --------------------------- Utilities ---------------------------
    def _parse_fact_name(self, name: str):
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


# Alias required by some dynamic loaders
robot_gpt5_heuristic_9 = RobotGPT5Heuristic9
